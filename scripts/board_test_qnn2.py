"""Test QNN framework with .bin (QNN format) models."""
import os, time
import numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

MODEL_DIR = "/home/fibo/AI model/tts_models/output"

def test(model_file, framework, runtime, label):
    path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(path):
        print(f"  SKIP: {path}")
        return None
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [{label}] {model_file} ({size_mb:.0f}MB) fw={framework} rt={runtime}")

    try:
        params = api_infer_py.InferParams(path, "QUALCOMM", framework, runtime, "ERROR", 5)
    except Exception as e:
        print(f"    InferParams FAILED: {e}")
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
    input_feed = {"z": z.flatten().tolist()}

    ret = api.Execute_float(input_feed)
    if ret != 0:
        print(f"    Execute FAILED: {ret}")
        api.Release()
        return None

    times = []
    for i in range(3):
        t0 = time.perf_counter()
        ret = api.Execute_float(input_feed)
        if ret != 0:
            print(f"    Execute FAILED at {i}: {ret}")
            break
        times.append(time.perf_counter() - t0)

    if times:
        avg_ms = np.mean(times) * 1000
        rtf = (avg_ms / 1000) / (65536 / 44100)
        print(f"    Avg: {avg_ms:.0f}ms | RTF: {rtf:.3f}")

    api.Release()
    return {"avg_ms": avg_ms, "rtf": rtf, "init_s": init_t, "size_mb": size_mb}

print("=" * 60)
print("QNN Framework (.bin format) vs SNPE Framework (.dlc format)")
print("=" * 60)

results = {}

# Test .bin files with QNN
for fw in ["QNN", "SNPE"]:
    for model, label in [
        ("generator_option_b.bin", "Option B FP32 bin"),
        ("generator_option_b_int16.dlc", "Option B INT16 dlc"),
    ]:
        for rt in ["DSP", "CPU"]:
            r = test(model, fw, rt, f"{label} {fw} {rt}")
            if r:
                results[f"{label} {fw} {rt}"] = r

print(f"\n{'='*60}")
print(f"{'Test':<35s} {'Size':>5s} {'RTF':>7s} {'Time':>8s}")
print("-" * 60)
for name, r in results.items():
    print(f"{name:<35s} {r['size_mb']:>4.0f}MB {r['rtf']:>6.3f} {r['avg_ms']:>7.0f}ms")
