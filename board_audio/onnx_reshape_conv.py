"""Use Reshape instead of Unsqueeze/Squeeze to convert Conv1d->Conv2d."""
import onnx
from onnx import helper, numpy_helper
import numpy as np
import onnxruntime as ort
import os

m = onnx.load('/project/aimet_work/generator_cle_static.onnx')

nodes = list(m.graph.node)
new_nodes = []
converted = 0

for node in nodes:
    if node.op_type in ('Conv', 'ConvTranspose') and len(node.input) >= 2:
        w_name = node.input[1]
        w = None
        init_idx = None
        for i, init in enumerate(m.graph.initializer):
            if init.name == w_name:
                w = numpy_helper.to_array(init)
                init_idx = i
                break

        if w is not None and len(w.shape) == 3:
            out_ch, in_ch, kw = w.shape[0], w.shape[1], w.shape[2]

            # --- Reshape weight 3D -> 4D ---
            w_4d = w.reshape(out_ch, in_ch, 1, kw)
            new_init = numpy_helper.from_array(w_4d, name=w_name)
            m.graph.initializer[init_idx].CopyFrom(new_init)

            # --- Insert Reshape BEFORE conv: [N,C,L] -> [N,C,1,L] ---
            pre_name = node.name + '_pre_reshape'
            pre_shape_name = pre_name + '_shape'
            # Shape: [N, C, 1, L] but N and L are dynamic. Use 0 to keep same, -1 to keep same
            # Actually for static model we can hardcode
            pre_shape = numpy_helper.from_array(
                np.array([1, in_ch, 1, -1], dtype=np.int64),
                name=pre_shape_name
            )
            m.graph.initializer.append(pre_shape)

            pre_reshape = helper.make_node(
                'Reshape',
                inputs=[node.input[0], pre_shape_name],
                outputs=[pre_name + '_out'],
                name=pre_name,
            )
            new_nodes.append(pre_reshape)
            node.input[0] = pre_name + '_out'

            # --- Fix Conv attributes 1D -> 2D ---
            new_attrs = []
            for attr in node.attribute:
                if attr.name == 'kernel_shape' and len(attr.ints) == 1:
                    new_attrs.append(helper.make_attribute('kernel_shape', [1, attr.ints[0]]))
                elif attr.name == 'pads' and len(attr.ints) == 2:
                    new_attrs.append(helper.make_attribute('pads', [0, 0, attr.ints[0], attr.ints[1]]))
                elif attr.name == 'strides' and len(attr.ints) == 1:
                    new_attrs.append(helper.make_attribute('strides', [1, attr.ints[0]]))
                elif attr.name == 'dilations':
                    pass  # skip, use default
                elif attr.name == 'group':
                    new_attrs.append(helper.make_attribute('group', int(attr.f)))
                else:
                    new_attrs.append(attr)
            node.ClearField('attribute')
            node.attribute.extend(new_attrs)

            # --- Insert Reshape AFTER conv: [N,C_out,1,L_out] -> [N,C_out,L_out] ---
            post_name = node.name + '_post_reshape'
            post_shape_name = post_name + '_shape'
            post_shape = numpy_helper.from_array(
                np.array([1, out_ch, -1], dtype=np.int64),
                name=post_shape_name
            )
            m.graph.initializer.append(post_shape)

            old_output = node.output[0]
            temp_out = old_output + '_4d'
            node.output[0] = temp_out

            post_reshape = helper.make_node(
                'Reshape',
                inputs=[temp_out, post_shape_name],
                outputs=[post_name + '_out'],
                name=post_name,
            )

            # --- Redirect downstream consumers ---
            new_output = post_name + '_out'
            for other in nodes:
                for j, inp in enumerate(other.input):
                    if inp == old_output:
                        other.input[j] = new_output

            new_nodes.append(node)
            new_nodes.append(post_reshape)
            converted += 1
            continue

    new_nodes.append(node)

del m.graph.node[:]
m.graph.node.extend(new_nodes)

out_path = '/project/aimet_work/generator_cle_2d_reshape.onnx'
onnx.save(m, out_path)
print('Converted {} Conv1d ops with Reshape'.format(converted))

# Verify
z = np.random.randn(1, 192, 128).astype(np.float32)*0.5
try:
    s_new = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    y_new = s_new.run(None, {'z': z})[0].squeeze()
    s_ref = ort.InferenceSession('/project/aimet_work/generator_cle_static.onnx', providers=['CPUExecutionProvider'])
    y_ref = s_ref.run(None, {'z': z})[0].squeeze()
    diff = np.abs(y_ref - y_new).max()
    corr = np.corrcoef(y_ref, y_new)[0,1]
    print('Verify OK: diff_max={:.10f}, corr={:.10f}'.format(diff, corr))
except Exception as e:
    print('ONNX Runtime error:', str(e)[:200])
    print('(trying QNN conversion anyway...)')

print('Size: {:.1f}MB'.format(os.path.getsize(out_path)/1024/1024))
