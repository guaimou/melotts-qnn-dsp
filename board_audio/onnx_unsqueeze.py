"""Insert Unsqueeze/Squeeze around Conv1d to make them Conv2d."""
import onnx
from onnx import helper, numpy_helper
import numpy as np
import onnxruntime as ort
from onnx import shape_inference
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
            # --- Reshape weight 3D -> 4D ---
            w_4d = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
            new_init = numpy_helper.from_array(w_4d, name=w_name)
            m.graph.initializer[init_idx].CopyFrom(new_init)

            # --- Insert Unsqueeze BEFORE conv ---
            unsq_name = node.name + '_unsq'
            # Create constant axes=[2] as initializer
            axes_name = unsq_name + '_axes'
            axes_init = numpy_helper.from_array(np.array([2], dtype=np.int64), name=axes_name)
            m.graph.initializer.append(axes_init)
            unsq = helper.make_node(
                'Unsqueeze',
                inputs=[node.input[0], axes_name],
                outputs=[unsq_name + '_out'],
                name=unsq_name,
            )
            new_nodes.append(unsq)

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
                    pass
                elif attr.name == 'group':
                    new_attrs.append(helper.make_attribute('group', int(attr.f)))
                else:
                    new_attrs.append(attr)

            node.ClearField('attribute')
            node.attribute.extend(new_attrs)

            # --- Update Conv input ---
            node.input[0] = unsq_name + '_out'

            # --- Insert Squeeze AFTER conv ---
            sqz_name = node.name + '_sqz'
            old_output = node.output[0]
            temp_out = old_output + '_4d'
            node.output[0] = temp_out

            axes_name_sqz = sqz_name + '_axes'
            axes_init_sqz = numpy_helper.from_array(np.array([2], dtype=np.int64), name=axes_name_sqz)
            m.graph.initializer.append(axes_init_sqz)
            sqz = helper.make_node(
                'Squeeze',
                inputs=[temp_out, axes_name_sqz],
                outputs=[sqz_name + '_out'],
                name=sqz_name,
            )

            # --- Redirect downstream consumers ---
            new_output = sqz_name + '_out'
            for other in nodes:
                for j, inp in enumerate(other.input):
                    if inp == old_output:
                        other.input[j] = new_output

            new_nodes.append(node)
            new_nodes.append(sqz)
            converted += 1
            continue

    new_nodes.append(node)

del m.graph.node[:]
m.graph.node.extend(new_nodes)

out_path = '/project/aimet_work/generator_cle_2d_us.onnx'
onnx.save(m, out_path)
print('Inserted Unsqueeze/Squeeze for {} Conv1d ops'.format(converted))

# Verify
z = np.random.randn(1, 192, 128).astype(np.float32)*0.5
s_ref = ort.InferenceSession('/project/aimet_work/generator_cle_static.onnx', providers=['CPUExecutionProvider'])
y_ref = s_ref.run(None, {'z': z})[0].squeeze()
s_new = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
y_new = s_new.run(None, {'z': z})[0].squeeze()
diff = np.abs(y_ref - y_new).max()
corr = np.corrcoef(y_ref, y_new)[0,1]
print('Verify: diff_max={:.10f}, corr={:.10f}'.format(diff, corr))
print('Size: {:.1f}MB'.format(os.path.getsize(out_path)/1024/1024))
