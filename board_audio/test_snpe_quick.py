"""Quick SNPE INT16 test."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

DLC_PATH = "/home/fibo/melotts_qnn/generator_cle_int16_v2.dlc"

print("Init SNPE DSP...")
t0 = time.time()
params = api_infer_py.InferParams(DLC_PATH, 'QUALCOMM', 'SNPE', 'HTP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)
print(f"Init: {time.time()-t0:.1f}s")

z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
print("Execute...")
t0 = time.time()
ret = api.Execute_float({"z": z.flatten().tolist()})
print(f"Execute: {time.time()-t0:.3f}s, ret={ret}")

if ret == 0:
    y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
    print(f"Output: shape={y.shape}, std={y.std():.6f}")
else:
    print(f"Execute failed: ret={ret}")

api.Release()
