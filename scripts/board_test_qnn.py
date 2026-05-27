"""Test QNN framework (not SNPE) for MeloTTS Generator on board DSP."""
import os, time
import numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

DLC_DIR = "/home/fibo/AI model/tts_models/output"

def test(framework, dlc_name, runtime, label):
    dlc_path = os.path.join(DLC_DIR, dlc_name)
    if not os.path.exists(dlc_path):
        print(f"  SKIP: {dlc_path}")
        return None

    size_mb = os.path.getsize(dlc_path) / 1024 / 1024
    print(f"  [{label}] fw={framework} rt={runtime} ({size_mb:.0f}MB)")

    try:
        params = api_infer_py.InferParams(dlc_path, "QUALCOMM", framework, runtime, "ERROR", 5)
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
        audio_dur = 65536 / 44100
        rtf = (avg_ms / 1000) / audio_dur
        print(f"    Avg: {avg_ms:.0f}ms | RTF: {rtf:.3f}")

    api.Release()
    return {"avg_ms": avg_ms, "rtf": rtf, "init_s": init_t}

print("=" * 55)
print("QNN Framework Test")
print("=" * 55)

for fw in ["QNN", "SNPE"]:
    for dlc, label in [
        ("generator_int16.dlc", "Original INT16"),
        ("generator_option_b_int16.dlc", "Option B INT16"),
    ]:
        for rt in ["DSP", "CPU"]:
            test(fw, dlc, rt, f"{label} {fw} {rt}")

print("\nDone")
