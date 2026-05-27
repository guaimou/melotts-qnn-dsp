"""Test INT16-quantized MeloTTS Generator DLC on board DSP."""
import os, time
import numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

DLC_DIR = "/home/fibo/AI model/tts_models/output"

def test_dlc(dlc_name, label, runtime="DSP"):
    dlc_path = os.path.join(DLC_DIR, dlc_name)
    if not os.path.exists(dlc_path):
        print(f"  SKIP: {dlc_path} not found")
        return None

    size_mb = os.path.getsize(dlc_path) / 1024 / 1024
    print(f"\n  [{label}] {dlc_name} ({size_mb:.0f}MB) runtime={runtime}")

    t0 = time.perf_counter()
    params = api_infer_py.InferParams(dlc_path, "QUALCOMM", "SNPE", runtime, "ERROR", 5)
    api = api_infer_py.InferAPI()
    ret = api.Init(params)
    init_t = time.perf_counter() - t0
    if ret != 0:
        print(f"    Init FAILED: {ret}")
        return None
    print(f"    Init: {init_t:.1f}s")

    z = np.random.randn(1, 192, 128).astype(np.float32)
    input_feed = {"z": z.flatten().tolist()}

    # Warmup
    ret = api.Execute_float(input_feed)
    if ret != 0:
        print(f"    Execute FAILED: {ret}")
        api.Release()
        return None

    # Benchmark
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        ret = api.Execute_float(input_feed)
        if ret != 0:
            print(f"    Execute FAILED at iter {i}: {ret}")
            break
        times.append(time.perf_counter() - t0)

    if times:
        avg_ms = np.mean(times) * 1000
        audio_dur = 65536 / 44100  # ~1.486s
        rtf = (avg_ms / 1000) / audio_dur
        print(f"    Avg: {avg_ms:.0f}ms | RTF: {rtf:.3f} | Speedup vs CPU: {5.385/(avg_ms/1000):.1f}x")

    api.Release()
    return {"avg_ms": avg_ms, "rtf": rtf, "init_s": init_t, "size_mb": size_mb}

print("=" * 55)
print("INT16 Generator DLC — Board DSP Test")
print("=" * 55)

results = {}
for dlc, label in [
    ("generator_int16.dlc", "Original INT16"),
    ("generator_option_b_int16.dlc", "Option B INT16"),
]:
    for rt in ["DSP", "CPU"]:
        r = test_dlc(dlc, f"{label} {rt}", runtime=rt)
        if r:
            results[f"{label} {rt}"] = r

print(f"\n{'='*55}")
print(f"{'Model':<25s} {'Size':>6s} {'RTF':>7s} {'Time':>8s} {'Init':>6s}")
print("-" * 55)
for name, r in results.items():
    print(f"{name:<25s} {r['size_mb']:>5.0f}MB {r['rtf']:>6.3f} {r['avg_ms']:>7.0f}ms {r['init_s']:>5.1f}s")

# Compare with previous float32 CPU result
print(f"\n  Reference (float32 CPU): 55MB, RTF 3.624, 5385ms")
print(f"  Official FiboTTS DSP:    116MB .fmodel, RTF ~0.38")
