"""Test full MeloTTS pipeline on board via ONNX (sherpa-onnx)."""
import os, sys, time
import numpy as np

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from melotts_onnx import MeloTTSOnnx

print("=" * 60)
print("MeloTTS Full Pipeline Benchmark (ONNX CPU)")
print("=" * 60)

# Test with default voice (uses sherpa-onnx)
tts = MeloTTSOnnx(voice="default")
print(f"Model loaded, sample_rate={tts.sample_rate}")

texts = [
    ("short", "你好世界"),
    ("medium", "今天天气真不错适合出去散步"),
    ("long", "人工智能技术正在改变我们的生活方式让生活变得更加便捷"),
]

total_chars = 0
total_time = 0

for label, text in texts:
    t0 = time.perf_counter()
    audio = tts.synthesize(text)
    elapsed = time.perf_counter() - t0

    audio_len_s = len(audio) / tts.sample_rate
    rtf = elapsed / audio_len_s if audio_len_s > 0 else 0
    chars = len(text)

    print(f"  [{label:6s}] {elapsed:.2f}s | audio {audio_len_s:.1f}s | RTF {rtf:.3f} | {chars} chars | {chars/elapsed:.1f} chars/s")

    total_chars += chars
    total_time += elapsed

print(f"\n  Total: {total_time:.2f}s for {total_chars} chars, avg {total_chars/total_time:.1f} chars/s")
print(f"\n  Per char breakdown (est.):")
print(f"    Encoder+DP: ~0.1-0.2s total")
print(f"    Generator:  {total_time - 0.15:.2f}s (est. from total - encoder)")

# Now also benchmark with A2 voice (if available)
print(f"\n--- A2 voice ---")
try:
    tts2 = MeloTTSOnnx(voice="a2")
    for label, text in texts[:2]:
        t0 = time.perf_counter()
        audio = tts2.synthesize(text)
        elapsed = time.perf_counter() - t0
        print(f"  [{label:6s}] {elapsed:.2f}s | RTF {elapsed/(len(audio)/tts2.sample_rate):.3f}")
except Exception as e:
    print(f"  A2 failed: {e}")

# Estimate with Option B generator
print(f"\n--- Projection with Option B (1.74x generator speedup) ---")
print(f"  If generator is ~80% of total time:")
gen_time = total_time * 0.80
new_gen = gen_time / 1.74
new_total = (total_time - gen_time) + new_gen
print(f"  Full pipeline: {total_time:.2f}s -> {new_total:.2f}s ({total_time/new_total:.1f}x faster)")
