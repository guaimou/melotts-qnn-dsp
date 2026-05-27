"""Test GPU backend and QNN quantized paths."""
import time, numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

SO_PATH = "/home/fibo/melotts_qnn/lib_arm/aarch64-ubuntu-gcc9.4/libmelotts_gen.so"
DLC_INT16 = "/home/fibo/AI model/tts_models/output/generator_option_b_int16.dlc"

def test(path, fw, rt, label):
    print(f"  [{label}] fw={fw} rt={rt}")
    try:
        params = api_infer_py.InferParams(path, "QUALCOMM", fw, rt, "ERROR", 5)
    except Exception as e:
        print(f"    Params: {e}")
        return None
    api = api_infer_py.InferAPI()
    t0 = time.perf_counter()
    ret = api.Init(params)
    init_t = time.perf_counter() - t0
    if ret != 0:
        print(f"    Init FAILED: {ret}")
        return None
    print(f"    Init: {init_t:.1f}s")

    z = np.random.randn(1, 192, 128).astype(np.float32)
    ret = api.Execute_float({"z": z.flatten().tolist()})
    if ret != 0:
        print(f"    Execute FAILED: {ret}")
        api.Release()
        return None

    times = []
    for i in range(3):
        t0 = time.perf_counter()
        ret = api.Execute_float({"z": z.flatten().tolist()})
        if ret != 0: break
        times.append(time.perf_counter() - t0)

    if times:
        avg_ms = np.mean(times) * 1000
        rtf = (avg_ms / 1000) / (65536 / 44100)
        print(f"    {avg_ms:.0f}ms | RTF {rtf:.3f}")
    api.Release()
    return {"rtf": rtf, "ms": avg_ms}

print("=" * 55)
print("GPU / QNN Quantized Test")
print("=" * 55)

# Test GPU with .so (FP32 model)
test(SO_PATH, "QNN", "GPU", "QNN GPU (FP32 .so)")

# Test DSP with INT16 DLC (already worked before)
test(DLC_INT16, "SNPE", "DSP", "SNPE DSP (INT16 DLC)")

# Test GPU with INT16 DLC
test(DLC_INT16, "SNPE", "GPU", "SNPE GPU (INT16 DLC)")

# Test GPU with INT16 DLC + QNN
test(DLC_INT16, "QNN", "GPU", "QNN GPU (INT16 DLC)")

print("\nDone")
