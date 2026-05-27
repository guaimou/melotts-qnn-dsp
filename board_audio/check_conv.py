import onnx
m = onnx.load('/project/aimet_work/generator_cle_static.onnx')
total = sum(1 for n in m.graph.node if n.op_type in ('Conv','ConvTranspose'))
d1 = sum(1 for n in m.graph.node if n.op_type in ('Conv','ConvTranspose') and len(n.input)>=2
         for init in m.graph.initializer if init.name==n.input[1] and len(init.dims)==3)
print(f'Total Conv: {total}, 1D: {d1}, 2D: {total-d1}')
