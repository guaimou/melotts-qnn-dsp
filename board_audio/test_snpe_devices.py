"""Test SNPE with different device strings."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

DLC_PATH = "/home/fibo/melotts_qnn/generator_cle_int16_v2.dlc"

for dev in ["HTP", "DSP", "CPU"]:
    print(f"\n=== SNPE + {dev} ===")
    try:
        params = api_infer_py.InferParams(DLC_PATH, 'QUALCOMM', 'SNPE', dev, 'ERROR', 5)
        api = api_infer_py.InferAPI()
        t0 = time.time()
        api.Init(params)
        init_t = time.time() - t0

        z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
        t0 = time.time()
        ret = api.Execute_float({"z": z.flatten().tolist()})
        exec_t = time.time() - t0

        if ret == 0:
            y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
            print(f"  Init={init_t:.1f}s, Exec={exec_t*1000:.0f}ms, out_std={y.std():.6f}")
        else:
            print(f"  Init={init_t:.1f}s, Exec failed ret={ret}")
        api.Release()
    except Exception as e:
        print(f"  FAIL: {str(e)[:100]}")
