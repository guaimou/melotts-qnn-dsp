"""Test static-shape 2.26.2 W8A16 vs ONNX reference."""
import sys, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py
import onnxruntime as ort

SO = "/home/fibo/melotts_qnn/lib_w8a16_static/aarch64-ubuntu-gcc9.4/libmelotts_w8a16_static.so"
z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# ONNX reference
sess = ort.InferenceSession("/home/fibo/melotts_qnn/generator_cle.onnx", providers=['CPUExecutionProvider'])
y_onnx = sess.run(None, {'z': z})[0].squeeze()

# W8A16 DSP
params = api_infer_py.InferParams(SO, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)
api.Execute_float({"z": z.flatten().tolist()})
raw = np.array(api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
y_dsp = (raw.view(np.uint16).astype(np.float32) - 32768) * 0.000030517578125

# Compare
a = y_onnx[:len(y_dsp)]
b = y_dsp[:len(a)]
corr = np.corrcoef(a, b)[0,1]
sp = np.var(a)
npw = np.var(a-b)
snr = 10*np.log10(sp/npw) if npw>0 else float('inf')
print(f"corr={corr:.6f}, SNR={snr:.1f}dB, max_diff={np.abs(a-b).max():.6f}")
print(f"ONNX first5: {np.round(a[:5],6)}")
print(f"DSP  first5: {np.round(b[:5],6)}")
api.Release()
