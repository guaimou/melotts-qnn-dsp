"""Benchmark full MeloTTS pipeline on board."""
import os, sys, time
sys.path.insert(0, "/home/fibo/AI model/tts_models")

print("=" * 60)
print("MeloTTS Full Pipeline Benchmark")
print("=" * 60)

# Use synthesize_to_file from melotts_onnx
import melotts_onnx

texts = [
    ("short4", "你好世界"),
    ("medium13", "今天天气真不错适合出去散步"),
    ("long29", "人工智能技术正在改变我们的生活方式让生活变得更加便捷"),
]

results = []

for voice in ["default", "a2", "a13"]:
    print(f"\n--- {voice} ---")
    try:
        for label, text in texts:
            t0 = time.perf_counter()
            out_path = melotts_onnx.synthesize_to_file(text, f"/tmp/bench_{voice}_{label}.wav", voice=voice)
            elapsed = time.perf_counter() - t0

            # Get audio duration
            import wave
            with wave.open(out_path, 'rb') as w:
                frames = w.getnframes()
                sr = w.getframerate()
                audio_s = frames / sr if sr > 0 else 1

            rtf = elapsed / audio_s if audio_s > 0 else 0
            chars = len(text)
            results.append({
                'voice': voice, 'label': label, 'chars': chars,
                'time_s': elapsed, 'audio_s': audio_s, 'rtf': rtf
            })
            print(f"  [{label:10s}] {elapsed:.2f}s | audio {audio_s:.1f}s | RTF {rtf:.3f} | {chars/elapsed:.0f} ch/s")

            # Clean up
            os.remove(out_path)
    except Exception as e:
        print(f"  Error: {e}")

# Summary
print(f"\n{'='*60}")
print("Summary (RTF)")
print("-" * 40)
for voice in ["default", "a2", "a13"]:
    voice_results = [r for r in results if r['voice'] == voice]
    if voice_results:
        avg_rtf = sum(r['rtf'] for r in voice_results) / len(voice_results)
        avg_time = sum(r['time_s'] for r in voice_results) / len(voice_results)
        print(f"  {voice:10s}: avg RTF {avg_rtf:.3f}, avg {avg_time:.1f}s")

# Compare with DSP FiboTTS baseline (from earlier test)
print(f"\n  Reference: DSP FiboTTS (official) ~0.38 RTF")
print(f"  Reference: Option B generator alone ~0.55 RTF")
