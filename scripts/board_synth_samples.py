"""Synthesize sample audio for voice quality evaluation."""
import os, sys, time
sys.path.insert(0, "/home/fibo/AI model/tts_models")
import melotts_onnx

OUT = "/home/fibo/melotts_qnn/output"
os.makedirs(OUT, exist_ok=True)

# Chinese test sentences covering different phonetic patterns
texts = [
    ("intro", "你好，欢迎使用本地语音合成系统"),
    ("news", "今天下午三点在北京人民大会堂举行新闻发布会"),
    ("daily", "明天天气不错，我们可以一起去公园散步"),
    ("story", "从前有一座山，山上有座庙，庙里住着一个老和尚"),
    ("tech", "人工智能技术正在深刻改变我们的生活方式"),
]

for voice in ["default", "a2", "a13"]:
    print(f"\n=== {voice} ===")
    for label, text in texts:
        out_path = os.path.join(OUT, f"{voice}_{label}.wav")
        t0 = time.perf_counter()
        melotts_onnx.synthesize_to_file(text, out_path, voice=voice)
        elapsed = time.perf_counter() - t0
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  {voice}_{label}.wav: {elapsed:.1f}s, {size_kb:.0f}KB - {text}")

print(f"\nAll files saved to {OUT}/")
print("Done")
