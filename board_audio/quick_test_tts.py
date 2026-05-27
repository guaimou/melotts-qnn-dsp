"""Quick TTS test: diagnose audio quality issue."""
import sys, os
sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
import numpy as np
from scipy.io import wavfile

text = "你好测试"
out = "/tmp/quick_tts_test.wav"

print(f"Mode: {tts.get_mode()}")
print(f"Voice: {tts.get_voice()}")
print(f"SpecSub: enabled={tts.QNN_SPECSUB_ENABLED}, alpha={tts.QNN_SPECSUB_ALPHA}")
print(f"Voices: {list(tts.QNN_VOICE_MAP.keys())}")

print(f"\nSynthesizing: '{text}'")
try:
    result, elapsed = tts.synthesize_to_wav(text, out)
    print(f"Result: {result}, elapsed: {elapsed:.1f}s")

    if os.path.exists(out):
        sr, audio = wavfile.read(out)
        print(f"Audio: sr={sr}, len={len(audio)}, dur={len(audio)/sr:.2f}s")
        print(f"Stats: min={audio.min()}, max={audio.max()}, mean={audio.mean():.1f}, std={audio.std():.1f}")

        # Check for obvious problems
        af = audio.astype(float)
        if af.std() < 10:
            print("WARNING: audio nearly silent!")
        if np.max(np.abs(af)) < 100:
            print("WARNING: very low amplitude!")
        if len(audio) < 1000:
            print("WARNING: audio too short!")

        # Quick noise check
        win = int(sr * 0.01)
        n = len(af) // win
        if n > 2:
            e = [np.std(af[i*win:(i+1)*win]) for i in range(n)]
            quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
            print(f"Quiet RMS: {quiet:.0f}")
    else:
        print("ERROR: output file not found!")

except Exception as e:
    import traceback
    traceback.print_exc()
