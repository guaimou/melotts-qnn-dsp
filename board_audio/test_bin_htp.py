"""Test: load .bin directly via QNN HTP."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

# Try INT8 .bin first (known to be correct)
BIN_PATH = "/home/fibo/melotts_qnn/gen_cle_int8_v2.bin"

print("=== Test: .bin via QNN + HTP ===")
print(f"Model: {BIN_PATH}")

# Try loading .bin directly with HTP
for dev in ["HTP", "DSP"]:
    print(f"\n--- {dev} ---")
    try:
        params = api_infer_py.InferParams(BIN_PATH, 'QUALCOMM', 'QNN', dev, 'ERROR', 5)
        api = api_infer_py.InferAPI()
        t0 = time.time()
        api.Init(params)
        print(f"Init OK: {time.time()-t0:.1f}s")

        # Try inference
        z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
        # Encode as uint8 (INT8 encoding)
        z_nhwc = np.transpose(z, (0, 2, 1))
        z_u8 = np.clip(np.round(z_nhwc / 0.06356 + 133), 0, 255).astype(np.uint8)

        t0 = time.time()
        ret = api.Execute_uint8({"z": z_u8.flatten().tolist()})
        print(f"Execute: {(time.time()-t0)*1000:.0f}ms, ret={ret}")

        if ret == 0:
            y = np.array(api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
            print(f"Output: len={len(y)}, first10={y[:10]}")
        api.Release()
    except Exception as e:
        print(f"FAIL: {str(e)[:200]}")
