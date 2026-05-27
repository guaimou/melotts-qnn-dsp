"""Compare INT8 DSP vs W8A16 DSP vs ONNX FP32 (same z)."""
import sys, time, numpy as np, types

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py
import onnxruntime as ort

z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# 1. ONNX FP32 reference
print("=== ONNX FP32 ===")
sess = ort.InferenceSession("/home/fibo/melotts_qnn/generator_cle.onnx", providers=['CPUExecutionProvider'])
y_onnx = sess.run(None, {'z': z})[0].squeeze()

# 2. W8A16 DSP
print("=== W8A16 DSP ===")
p_w8a16 = api_infer_py.InferParams(
    "/home/fibo/melotts_qnn/lib_cle_w8a16/aarch64-ubuntu-gcc9.4/libmelotts_cle_w8a16.so",
    'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api_w8 = api_infer_py.InferAPI()
api_w8.Init(p_w8a16)

api_w8.Execute_float({"z": z.flatten().tolist()})
y_u8_w8 = np.array(api_w8.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
y_u16 = y_u8_w8.view(np.uint16)
y_w8a16 = (y_u16.astype(np.float32) - 32768) * 0.0000305175781250
print(f"W8A16: std={y_w8a16.std():.6f}")

# 3. INT8 DSP (from new .so)
print("=== INT8 DSP ===")
p_int8 = api_infer_py.InferParams(
    "/home/fibo/melotts_qnn/lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so",
    'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api_i8 = api_infer_py.InferAPI()
api_i8.Init(p_int8)

IN_SCALE = 0.0635628774762154
IN_OFFSET = -133
OUT_SCALE = 0.0016634501516819
OUT_OFFSET = -117

z_nhwc = np.transpose(z, (0, 2, 1))
z_u8 = np.clip(np.round(z_nhwc / IN_SCALE - IN_OFFSET), 0, 255).astype(np.uint8)
api_i8.Execute_uint8({"z": z_u8.flatten().tolist()})
y_u8_i8 = np.array(api_i8.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
y_int8 = (y_u8_i8.astype(np.float32) + OUT_OFFSET) * OUT_SCALE
print(f"INT8: std={y_int8.std():.6f}")

# 4. Compare both vs ONNX
print("\n=== Results ===")
for label, y in [("W8A16", y_w8a16), ("INT8", y_int8)]:
    a = y_onnx[:len(y)]
    b = y[:len(a)]
    corr = np.corrcoef(a, b)[0, 1]
    sp = np.var(a)
    npower = np.var(a - b)
    snr = 10 * np.log10(sp / npower) if npower > 0 else float('inf')
    max_diff = np.abs(a - b).max()
    print(f"{label}: corr={corr:.6f}, SNR={snr:.1f}dB, max_diff={max_diff:.6f}, std_onnx={a.std():.4f}, std_dsp={b.std():.4f}")

api_w8.Release()
api_i8.Release()
