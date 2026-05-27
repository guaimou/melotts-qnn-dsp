"""Test MeloTTS Generator DLC on board via fiboaisdk InferAPI."""
import os, sys, time
import numpy as np

from fiboaisdk.api_aisdk_py import api_infer_py

DLC_DIR = "/home/fibo/AI model/tts_models/output"

def test_dlc(dlc_name, label, framework="SNPE", runtime="CPU"):
    dlc_path = os.path.join(DLC_DIR, dlc_name)
    if not os.path.exists(dlc_path):
        print(f"  SKIP: {dlc_path} not found")
        return None

    print(f"\n  [{label}] {dlc_name} ({os.path.getsize(dlc_path)/1024/1024:.0f}MB)")
    print(f"    Framework={framework}, Runtime={runtime}")

    # Init
    t0 = time.perf_counter()
    params = api_infer_py.InferParams(dlc_path, "QUALCOMM", framework, runtime, "ERROR", 5)
    api = api_infer_py.InferAPI()
    ret = api.Init(params)
    init_t = time.perf_counter() - t0
    if ret != 0:
        print(f"    Init FAILED: {ret}")
        return None
    print(f"    Init: {init_t:.2f}s")

    # Prepare input (random z latent)
    z = np.random.randn(1, 192, 128).astype(np.float32)
    input_feed = {"z": z.flatten().tolist()}

    # Warmup
    api.Execute_float(input_feed)

    # Benchmark
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        ret = api.Execute_float(input_feed)
        elapsed = time.perf_counter() - t0
        if ret != 0:
            print(f"    Execute FAILED: {ret}")
            break
        times.append(elapsed)

    if times:
        avg_t = np.mean(times) * 1000
        min_t = np.min(times) * 1000
        # Fetch output to verify shape
        result = api.FetchOutputs_float(["y"])
        out_shape = "?"
        for k, v in result.items():
            out_shape = str(np.array(v).shape)

        audio_samples = 65536  # T=128 * hop=512
        audio_dur = audio_samples / 44100
        rtf = (avg_t / 1000) / audio_dur

        print(f"    Output shape: {out_shape}")
        print(f"    Avg time: {avg_t:.1f}ms | Min: {min_t:.1f}ms")
        print(f"    RTF: {rtf:.3f} ({audio_dur:.1f}s audio in {avg_t:.1f}ms)")

    api.Release()
    return {"avg_ms": avg_t, "min_ms": min_t, "rtf": rtf, "init_s": init_t}

# Test both models
print("=" * 50)
print("MeloTTS Generator DLC Board Test")
print("=" * 50)

results = {}
for dlc, label in [("generator_original.dlc", "Original"), ("generator_option_b.dlc", "Option B")]:
    for runtime in ["CPU", "DSP"]:
        result = test_dlc(dlc, f"{label} {runtime}", runtime=runtime)
        if result:
            results[f"{label}_{runtime}"] = result

# Summary
print(f"\n{'='*50}")
print("Summary")
print(f"{'='*50}")
for name, r in results.items():
    print(f"  {name:25s}: {r['rtf']:.3f} RTF | {r['avg_ms']:.0f}ms avg | init {r['init_s']:.1f}s")
