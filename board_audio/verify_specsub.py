"""Verify spectral subtraction in TTS pipeline."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts

text = "你好，这是一个测试语音合成效果。"
VOICE = "default_cle"

print(f"SpecSub enabled: {tts.QNN_SPECSUB_ENABLED}")
print(f"alpha: {tts.QNN_SPECSUB_ALPHA}, beta: {tts.QNN_SPECSUB_BETA}")

# Encode
z_saved, T_saved = tts._qnn_encode(text, voice=VOICE, speed=1.0)
print(f"z shape={z_saved.shape}")

# Patch encode for fair comparison
orig = tts._qnn_encode
tts._qnn_encode = lambda *a, **kw: (z_saved, T_saved)

# With specsub
print("\nWith specsub...")
tts.QNN_SPECSUB_ENABLED = True
tts._qnn_synthesize_to_wav(text, "/tmp/verify_specsub_on.wav", voice=VOICE)

# Without
print("Without specsub...")
tts.QNN_SPECSUB_ENABLED = False
tts._qnn_synthesize_to_wav(text, "/tmp/verify_specsub_off.wav", voice=VOICE)

tts._qnn_encode = orig
tts.QNN_SPECSUB_ENABLED = True

# Analyze
from scipy.io import wavfile
import os

for path, label in [("/tmp/verify_specsub_on.wav", "SpecSub ON"), ("/tmp/verify_specsub_off.wav", "SpecSub OFF")]:
    sr, a = wavfile.read(path)
    af = a.astype(float)
    win = int(sr * 0.01)
    n = len(af) // win
    e = [np.std(af[i*win:(i+1)*win]) for i in range(n)]
    quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
    loud = np.mean([x for x in e if x > np.percentile(e, 90)])
    print(f"\n{label}:")
    print(f"  Quiet={quiet:.0f}, Loud={loud:.0f}, DR={20*np.log10(loud/quiet):.1f}dB")
    print(f"  Total={af.std():.0f}, Peak={max(abs(a.min()), abs(a.max()))}")

os.system("cp /tmp/verify_specsub_on.wav /home/fibo/melotts_qnn/verify_specsub_on.wav")
os.system("cp /tmp/verify_specsub_off.wav /home/fibo/melotts_qnn/verify_specsub_off.wav")
print("\nSaved: verify_specsub_on.wav, verify_specsub_off.wav")
