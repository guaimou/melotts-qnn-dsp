"""Benchmark MeloTTS Generator ONNX models on board (ORT CPU)."""
import os, time
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/home/fibo/melotts_qnn"
Z_SHAPE = (1, 192, 128)
HOP = 512
SR = 44100
WARMUP = 3
BENCH_RUNS = 10

def benchmark(model_path, label):
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path}")
        return None

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  [{label}] {os.path.basename(model_path)} ({size_mb:.0f}MB)")

    # Create session
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    init_t = time.perf_counter() - t0
    print(f"    Init: {init_t:.2f}s")

    # Prepare input
    z = np.random.randn(*Z_SHAPE).astype(np.float32)
    feed = {"z": z}

    # Warmup
    for _ in range(WARMUP):
        sess.run(None, feed)

    # Benchmark
    times = []
    for i in range(BENCH_RUNS):
        t0 = time.perf_counter()
        out = sess.run(None, feed)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000
    audio_s = out[0].shape[-1] / SR
    rtf = (avg_ms / 1000) / audio_s

    print(f"    Output: {out[0].shape} | Audio: {audio_s:.2f}s")
    print(f"    Avg: {avg_ms:.0f}ms | Min: {min_ms:.0f}ms | RTF: {rtf:.3f}")
    return {"avg_ms": avg_ms, "min_ms": min_ms, "rtf": rtf, "init_s": init_t, "size_mb": size_mb}

print("=" * 60)
print("MeloTTS Generator ONNX Benchmark (ORT CPU)")
print("=" * 60)

results = {}
for fname, label in [
    ("generator_original.onnx", "Original"),
    ("generator_option_b.onnx", "Option B"),
]:
    r = benchmark(os.path.join(MODEL_DIR, fname), label)
    if r:
        results[label] = r

# Summary
print(f"\n{'='*60}")
print(f"{'Model':<15s} {'Size':>6s} {'RTF':>7s} {'Time':>8s} {'Speedup':>8s}")
print("-" * 60)
baseline_rtf = results.get("Original", {}).get("rtf", 1)
for name, r in results.items():
    speedup = baseline_rtf / r["rtf"] if r["rtf"] > 0 else 0
    print(f"{name:<15s} {r['size_mb']:>5.0f}MB {r['rtf']:>6.3f} {r['avg_ms']:>7.0f}ms {speedup:>7.2f}x")

# Comparison with other methods
print(f"\n  Reference: SNPE DSP INT16 Option B  RTF 1.92")
print(f"  Reference: Official FiboTTS DSP    RTF 0.38")
