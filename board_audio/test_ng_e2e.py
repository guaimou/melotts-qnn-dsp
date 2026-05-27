"""End-to-end test: same z_latent, gate integrated in _qnn_synthesize_to_wav."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts

text = "你好，这是一个测试语音合成效果。"
VOICE = "default_cle"

# Encode once
print("Encoding...")
z_saved, T_saved = tts._qnn_encode(text, voice=VOICE, speed=1.0)
print(f"z shape={z_saved.shape}")

# Monkey-patch _qnn_encode to return saved z
original_encode = tts._qnn_encode
tts._qnn_encode = lambda text, voice=None, speed=1.0: (z_saved, T_saved)

# Test with gate ON
print("\nWith gate ON...")
tts.QNN_NOISE_GATE_ENABLED = True
tts._qnn_synthesize_to_wav(text, "/tmp/e2e_gate_on.wav", voice=VOICE, speed=1.0)

# Test with gate OFF
print("With gate OFF...")
tts.QNN_NOISE_GATE_ENABLED = False
tts._qnn_synthesize_to_wav(text, "/tmp/e2e_gate_off.wav", voice=VOICE, speed=1.0)

# Restore
tts._qnn_encode = original_encode

# Analyze
from scipy.io import wavfile
import os

def analyze(path, label):
    sr, a = wavfile.read(path)
    af = a.astype(float)
    win = int(sr * 0.01)
    n = len(af) // win
    e = [np.std(af[i*win:(i+1)*win]) for i in range(n)]
    quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
    loud = np.mean([x for x in e if x > np.percentile(e, 90)])
    return {"label": label, "quiet": quiet, "loud": loud, "total": af.std()}

r_on = analyze("/tmp/e2e_gate_on.wav", "Gate ON")
r_off = analyze("/tmp/e2e_gate_off.wav", "Gate OFF")

print(f"\n{'Method':<15} {'Quiet RMS':>12} {'Loud RMS':>12} {'DR':>8} {'Total':>10}")
for r in [r_on, r_off]:
    dr = 20*np.log10(r["loud"]/r["quiet"]) if r["quiet"]>0 else 0
    print(f"{r['label']:<15} {r['quiet']:>12.0f} {r['loud']:>12.0f} {dr:>7.1f}dB {r['total']:>10.0f}")

imp = (1 - r_on["quiet"] / r_off["quiet"]) * 100
print(f"\nNoise gate improvement: {imp:+.1f}%")

os.system("cp /tmp/e2e_gate_on.wav /home/fibo/melotts_qnn/e2e_gate_on.wav")
os.system("cp /tmp/e2e_gate_off.wav /home/fibo/melotts_qnn/e2e_gate_off.wav")
print("Saved: e2e_gate_on.wav, e2e_gate_off.wav")
