"""Test SNPE INT16 DLC inference via fiboaisdk InferAPI."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

DLC_PATH = "/home/fibo/melotts_qnn/generator_cle_int16.dlc"

print("=== SNPE DSP INT16 Test ===")
print(f"Model: {DLC_PATH}")

# Init
print("\n[1] Initializing SNPE DSP...")
t0 = time.time()
params = api_infer_py.InferParams(DLC_PATH, 'QUALCOMM', 'SNPE', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)
print(f"  Init done in {time.time()-t0:.1f}s")

# Test inference
print("\n[2] Testing INT16 inference...")
z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# Check what input format SNPE expects
# QNN INT8 uses NHWC uint8. SNPE INT16 may use different format.
# Try float input first (Execute_float)
try:
    t0 = time.time()
    ret = api.Execute_float({"z": z.flatten().tolist()})
    print(f"  Execute_float: {time.time()-t0:.3f}s, ret={ret}")

    if ret == 0:
        y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
        print(f"  Output: shape={y.shape}, min={y.min():.6f}, max={y.max():.6f}, std={y.std():.6f}")
    else:
        print(f"  Execute_float failed with ret={ret}")
except Exception as e:
    print(f"  Execute_float error: {e}")

# Speed test
if ret == 0:
    print("\n[3] Speed test (5 runs)...")
    times = []
    for i in range(5):
        z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
        t0 = time.time()
        api.Execute_float({"z": z.flatten().tolist()})
        api.FetchOutputs_float(["y"])
        times.append(time.time() - t0)
        print(f"  Run {i+1}: {times[-1]*1000:.0f}ms")

    avg = np.mean(times[1:])
    print(f"\n  Average (warm): {avg*1000:.0f}ms, RTF={avg/(65536/44100):.3f}")

api.Release()
