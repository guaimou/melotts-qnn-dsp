"""Test W8A16 with correct output decoding."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

SO_PATH = "/home/fibo/melotts_qnn/lib_cle_w8a16/aarch64-ubuntu-gcc9.4/libmelotts_cle_w8a16.so"
OUT_SCALE = 0.0000305175781250
OUT_OFFSET = -32768

print("=== QNN W8A16 v2 ===")
params = api_infer_py.InferParams(SO_PATH, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)

z_fp32 = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# Execute_float works (ret=0), just need to fix output decoding
api.Execute_float({"z": z_fp32.flatten().tolist()})

# Try FetchOutputs_uint8 and reinterpret as uint16
try:
    y_u8 = np.array(api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
    print(f"FetchOutputs_uint8: shape={y_u8.shape}, len={len(y_u8)}")

    # Reinterpret bytes as uint16 (little-endian)
    if len(y_u8) >= 65536 * 2:
        y_u16 = y_u8[:65536*2].view(np.uint16)
        y_f32 = (y_u16.astype(np.float32) + OUT_OFFSET) * OUT_SCALE
        print(f"uint16 decode: shape={y_u16.shape}, min={y_f32.min():.6f}, max={y_f32.max():.6f}, std={y_f32.std():.6f}")
    elif len(y_u8) == 65536:
        y_u16 = y_u8.astype(np.uint16)
        y_f32 = (y_u16.astype(np.float32) + OUT_OFFSET) * OUT_SCALE
        print(f"uint8→uint16 decode: shape={y_u16.shape}, min={y_f32.min():.6f}, max={y_f32.max():.6f}, std={y_f32.std():.6f}")
    else:
        print(f"Unexpected size: {len(y_u8)}, raw first 10: {y_u8[:10]}")
except Exception as e:
    print(f"FetchOutputs_uint8 error: {e}")

# Speed test
print("\nSpeed test (3 runs)...")
times = []
for i in range(3):
    z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
    t0 = time.time()
    ret = api.Execute_float({"z": z.flatten().tolist()})
    api.FetchOutputs_uint8(["y"])
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed*1000:.0f}ms")

avg = np.mean(times)
print(f"\nAverage: {avg*1000:.0f}ms, RTF={avg/(65536/44100):.3f}")

api.Release()
