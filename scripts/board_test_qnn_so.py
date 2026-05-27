"""Test QNN framework with aarch64 .so model on board."""
import time, numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

SO_PATH = "/home/fibo/melotts_qnn/lib_arm/aarch64-ubuntu-gcc9.4/libmelotts_gen.so"

print("=" * 55)
print("QNN Framework + .so Test")
print("=" * 55)

# Test QNN framework with .so
for fw in ["QNN", "SNPE"]:
    for rt in ["DSP", "CPU"]:
        print(f"\n  [{fw} {rt}] libmelotts_gen.so")
        try:
            params = api_infer_py.InferParams(SO_PATH, "QUALCOMM", fw, rt, "ERROR", 5)
        except Exception as e:
            print(f"    InferParams FAILED: {e}")
            continue

        api = api_infer_py.InferAPI()
        t0 = time.perf_counter()
        ret = api.Init(params)
        init_t = time.perf_counter() - t0
        if ret != 0:
            print(f"    Init FAILED: {ret}")
            continue
        print(f"    Init: {init_t:.1f}s")

        z = np.random.randn(1, 192, 128).astype(np.float32)
        ret = api.Execute_float({"z": z.flatten().tolist()})
        if ret != 0:
            print(f"    Execute FAILED: {ret}")
            api.Release()
            continue

        times = []
        for i in range(3):
            t0 = time.perf_counter()
            ret = api.Execute_float({"z": z.flatten().tolist()})
            if ret != 0: break
            times.append(time.perf_counter() - t0)

        if times:
            avg_ms = np.mean(times) * 1000
            rtf = (avg_ms / 1000) / (65536 / 44100)
            print(f"    Avg: {avg_ms:.0f}ms | RTF: {rtf:.3f}")

        api.Release()

print("\nDone")
