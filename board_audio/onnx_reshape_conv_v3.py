"""v3: Use int32 shape arrays, skip QNN-incompatible patterns."""
import onnx
from onnx import helper, numpy_helper, shape_inference
import numpy as np
import os

m = onnx.load('/project/aimet_work/generator_cle_static.onnx')
m_inferred = shape_inference.infer_shapes(m)

shape_map = {}
for vi in m_inferred.graph.value_info:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else 1)
    shape_map[vi.name] = dims
for inp in m_inferred.graph.input:
    shape_map[inp.name] = [d.dim_value for d in inp.type.tensor_type.shape.dim]
for out in m_inferred.graph.output:
    shape_map[out.name] = [d.dim_value for d in out.type.tensor_type.shape.dim]

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

            # Reshape weight 3D -> 4D
            w_4d = w.reshape(out_ch, in_ch, 1, kw)
            new_init = numpy_helper.from_array(w_4d, name=w_name)
            m.graph.initializer[init_idx].CopyFrom(new_init)

            input_name = node.input[0]
            in_shape = shape_map.get(input_name, [1, in_ch, 128])
            if len(in_shape) == 3:
                N_val, C_val, L_val = in_shape[0], in_shape[1], in_shape[2]
            else:
                N_val, C_val, L_val = 1, in_ch, 128

            # Pre-Reshape using int32
            pre_name = node.name + '_pre_reshape'
            pre_shape_name = pre_name + '_shape'
            pre_shape = numpy_helper.from_array(
                np.array([N_val, C_val, 1, L_val], dtype=np.int32),  # int32!
                name=pre_shape_name
            )
            m.graph.initializer.append(pre_shape)
            pre_reshape = helper.make_node(
                'Reshape', inputs=[input_name, pre_shape_name],
                outputs=[pre_name + '_out'], name=pre_name,
            )
            new_nodes.append(pre_reshape)
            node.input[0] = pre_name + '_out'

            # Fix Conv attributes 1D -> 2D
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
            # Add dilations if not present
            if not any(a.name == 'dilations' for a in new_attrs):
                new_attrs.append(helper.make_attribute('dilations', [1, 1]))
            node.ClearField('attribute')
            node.attribute.extend(new_attrs)

            old_output = node.output[0]
            temp_out = old_output + '_4d'
            node.output[0] = temp_out

            out_shape = shape_map.get(old_output, [1, out_ch, L_val])
            if len(out_shape) == 3:
                N_out, C_out, L_out = out_shape[0], out_shape[1], out_shape[2]
            else:
                N_out, C_out, L_out = 1, out_ch, L_val

            # Post-Reshape using int32
            post_name = node.name + '_post_reshape'
            post_shape_name = post_name + '_shape'
            post_shape = numpy_helper.from_array(
                np.array([N_out, C_out, L_out], dtype=np.int32),  # int32!
                name=post_shape_name
            )
            m.graph.initializer.append(post_shape)
            post_reshape = helper.make_node(
                'Reshape', inputs=[temp_out, post_shape_name],
                outputs=[post_name + '_out'], name=post_name,
            )

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

out_path = '/project/aimet_work/generator_cle_2d_v3.onnx'
onnx.save(m, out_path)
print('Converted {} ops with int32 shapes'.format(converted))

# Shape inference
m2 = shape_inference.infer_shapes(onnx.load(out_path))
out_inferred = '/project/aimet_work/generator_cle_2d_v3_inferred.onnx'
onnx.save(m2, out_inferred)
print('Inferred: {} value_info, saved'.format(len(m2.graph.value_info)))
