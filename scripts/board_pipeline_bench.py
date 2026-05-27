"""Full pipeline: ONNX CPU (audio samples) + QNN DSP INT8 benchmark projection."""
import os, sys, time, wave
import numpy as np
sys.path.insert(0, "/home/fibo/AI model/tts_models")
import melotts_onnx

OUT = "/home/fibo/melotts_qnn/output"
os.makedirs(OUT, exist_ok=True)

texts = [
    ("short_4", "你好世界"),
    ("intro_9", "你好，欢迎使用本地语音合成"),
    ("news_20", "今天下午三点在北京人民大会堂举行新闻发布会"),
    ("daily_13", "明天天气不错，我们可以一起去公园散步"),
    ("long_27", "人工智能技术正在深刻改变我们的生活方式让生活变得更加便捷"),
]

print("=" * 65)
print("MeloTTS Full Pipeline: ONNX CPU + QNN DSP INT8 Projection")
print("=" * 65)

# QNN DSP INT8 generator benchmark result (from actual board test)
QNN_GEN_PER_CHUNK = 0.202  # seconds per 65536-sample chunk
QNN_GEN_RTF = 0.136
CHUNK_SIZE = 65536
SR = 44100

for voice in ["default", "a2", "a13"]:
    print(f"\n{'='*65}")
    print(f"  {voice.upper()}")
    print(f"{'='*65}")
    print(f"{'Text':<12s} {'Chars':>5s} {'Audio':>6s} {'ONNX':>7s} {'ONNX RTF':>8s} {'QNN est':>8s} {'QNN RTF':>8s} {'Speedup':>7s}")
    print("-" * 65)

    for label, text in texts:
        t0 = time.perf_counter()
        wav_path = os.path.join(OUT, f"{voice}_{label}.wav")
        melotts_onnx.synthesize_to_file(text, wav_path, voice=voice)
        onnx_time = time.perf_counter() - t0

        with wave.open(wav_path, 'rb') as w:
            samples = w.getnframes()
            audio_s = samples / SR

        chunks = max(1, int(np.ceil(samples / CHUNK_SIZE)))
        qnn_time = (onnx_time - chunks * 0.82) + chunks * QNN_GEN_PER_CHUNK
        # 0.82 = Option B ONNX CPU gen per chunk (from benchmark)
        # For default voice which is smaller, use 0.50 (proportional)
        if voice == "default":
            qnn_time = (onnx_time - chunks * 0.50) + chunks * QNN_GEN_PER_CHUNK

        qnn_time = max(qnn_time, 0.15)  # minimum overhead

        rtf_onnx = onnx_time / audio_s if audio_s > 0 else 0
        rtf_qnn = qnn_time / audio_s if audio_s > 0 else 0
        speedup = onnx_time / qnn_time if qnn_time > 0 else 0

        print(f"{label:<12s} {len(text):>5d} {audio_s:>5.1f}s {onnx_time:>6.2f}s {rtf_onnx:>7.3f} {qnn_time:>7.2f}s {rtf_qnn:>7.3f} {speedup:>6.1f}x")

print(f"\nAudio samples saved to: {OUT}/")
print(f"QNN DSP INT8 Generator: {QNN_GEN_PER_CHUNK*1000:.0f}ms/chunk, RTF {QNN_GEN_RTF}")
print("Done")
