"""Fair comparison: same z_latent, gate ON vs OFF."""
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
z_latent, T_enc = tts._qnn_encode(text, voice=VOICE, speed=1.0)
print(f"z shape={z_latent.shape}, T={T_enc}")

# Run DSP without gate
print("\nWithout gate...")
old_gate = tts.QNN_NOISE_GATE_ENABLED
tts.QNN_NOISE_GATE_ENABLED = False
audio_off = tts._qnn_generate(z_latent, voice=VOICE)

# Run DSP with gate
print("With gate...")
tts.QNN_NOISE_GATE_ENABLED = True
audio_on = tts._qnn_generate(z_latent, voice=VOICE)

# Apply gate manually to the "off" audio for comparison
audio_manual = tts._apply_noise_gate(audio_off, sr=44100, strength=0.5)

# Restore
tts.QNN_NOISE_GATE_ENABLED = old_gate

# Analyze all three at raw float level (no normalization)
def analyze(a, label):
    win = int(44100 * 0.01)
    n = len(a) // win
    e = [np.std(a[i*win:(i+1)*win]) for i in range(n)]
    quiet = np.mean([x for x in e if x < np.percentile(e, 10)])
    loud = np.mean([x for x in e if x > np.percentile(e, 90)])
    return {"label": label, "quiet": quiet, "loud": loud, "total": np.std(a),
            "dr": 20*np.log10(loud/quiet) if quiet>0 else 0}

results = [
    analyze(audio_off, "No Gate (raw DSP)"),
    analyze(audio_manual, "Gate Post (manual)"),
    analyze(audio_on, "Gate Integrated"),
]

print(f"\n{'Method':<25} {'Quiet RMS':>12} {'Loud RMS':>12} {'DR':>8} {'Total':>10}")
for r in results:
    print(f"{r['label']:<25} {r['quiet']:>12.6f} {r['loud']:>12.4f} {r['dr']:>7.1f}dB {r['total']:>10.4f}")

base_q = results[0]["quiet"]
for r in results[1:]:
    imp = (1 - r["quiet"] / base_q) * 100
    print(f"\n{r['label']}: quiet RMS {imp:+.1f}% vs no-gate")

# Check correlation between gate methods
corr = np.corrcoef(audio_manual, audio_on)[0, 1]
print(f"\nGate correlation (manual vs integrated): {corr:.6f}")
