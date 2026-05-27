"""Test QNN W8A16 DSP inference."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

SO_PATH = "/home/fibo/melotts_qnn/lib_cle_w8a16/aarch64-ubuntu-gcc9.4/libmelotts_cle_w8a16.so"
IN_SCALE = 0.0002481628616806
IN_OFFSET = -34127
OUT_SCALE = 0.0000305175781250
OUT_OFFSET = -32768

print("=== QNN W8A16 ===")
print(f"Model: {SO_PATH}")

# Init
t0 = time.time()
params = api_infer_py.InferParams(SO_PATH, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)
print(f"Init: {time.time()-t0:.1f}s")

z_fp32 = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# Try 1: Execute_float (let SDK quantize)
print("\n[Try 1] Execute_float...")
t0 = time.time()
ret = api.Execute_float({"z": z_fp32.flatten().tolist()})
t_exec = time.time() - t0
print(f"  {t_exec*1000:.0f}ms, ret={ret}")

if ret == 0:
    y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
    print(f"  Output: shape={y.shape}, std={y.std():.6f}")
    print(f"  WORKS! RTF={(t_exec/(65536/44100)):.3f}")
else:
    # Try 2: Manual INT16 encoding
    print("\n[Try 2] Manual UFIXED_POINT_16...")
    # NHWC: [1, 128, 192]
    z_nhwc = np.transpose(z_fp32, (0, 2, 1))
    z_u16 = np.clip(np.round(z_nhwc / IN_SCALE - IN_OFFSET), 0, 65535).astype(np.uint16)

    t0 = time.time()
    ret = api.Execute_uint16({"z": z_u16.flatten().tolist()})
    t_exec = time.time() - t0
    print(f"  Execute_uint16: {t_exec*1000:.0f}ms, ret={ret}")

    if ret == 0:
        y_u16 = np.array(api.FetchOutputs_uint16(["y"])["y"], dtype=np.uint16)
        y = (y_u16.astype(np.float32) + OUT_OFFSET) * OUT_SCALE
        print(f"  Output: shape={y.shape}, std={y.std():.6f}")
    else:
        # Try 3: Execute_uint8 with 16-bit encoding split into two uint8 arrays
        print(f"\n[Try 3] Execute_float z quantized externally...")
        # The QNN runtime might accept float and internally convert
        # Try treating the model as accepting uint16 via float fallback
        print(f"  All attempts failed. API may not support UFIXED_POINT_16 directly.")

api.Release()
