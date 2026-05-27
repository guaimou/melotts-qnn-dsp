"""Test SDK 2.46 INT16 DLC via SNPE HTP."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

DLC_PATH = "/home/fibo/melotts_qnn/generator_cle_246_int16.dlc"

print("=== SDK 2.46 INT16 + SNPE HTP ===")

t0 = time.time()
params = api_infer_py.InferParams(DLC_PATH, 'QUALCOMM', 'SNPE', 'HTP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)
init_t = time.time() - t0
print(f"Init: {init_t:.1f}s")

# Speed test
print("Speed test (5 runs)...")
times = []
for i in range(5):
    z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
    t0 = time.time()
    ret = api.Execute_float({"z": z.flatten().tolist()})
    if ret == 0:
        y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed*1000:.0f}ms, ret={ret}, out_std={y.std():.6f}")

avg = np.mean(times[1:])
print(f"\nAverage (warm): {avg*1000:.0f}ms, RTF={avg/(65536/44100):.3f}")

api.Release()
