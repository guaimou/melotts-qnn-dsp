"""Quick comparison: CLE vs SQNR vs PCH."""
import types, sys, time, numpy as np

m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_synthesize_to_wav
from scipy.io import wavfile

text = "你好，这是一个测试语音合成效果。"
results = {}
import os

for voice, label in [("default_cle", "CLE+SQNR"), ("default_ada_sqnr", "AdaRound+SQNR"), ("default", "PCH")]:
    print(f"=== {label} ({voice}) ===")
    t0 = time.time()
    out = _qnn_synthesize_to_wav(text, f"/tmp/test_{voice}.wav", voice=voice, speed=1.0)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    sr, audio = wavfile.read(f"/tmp/test_{voice}.wav")
    audio_f = audio.astype(float)

    # Noise analysis
    win = int(sr * 0.01)
    n_win = len(audio_f) // win
    energies = [np.std(audio_f[i*win:(i+1)*win]) for i in range(n_win)]
    quiet_thresh = np.percentile(energies, 10)
    quiet_rms = np.mean([e for e in energies if e < quiet_thresh])

    results[label] = {
        "quiet_rms": quiet_rms,
        "duration": len(audio_f)/sr,
        "time": elapsed,
        "rms": np.std(audio_f),
    }

    os.system(f"cp /tmp/test_{voice}.wav /home/fibo/melotts_qnn/test_{voice}.wav")
    print(f"  Quiet RMS: {quiet_rms:.0f}, RMS: {results[label]['rms']:.0f}")

print("\n=== Summary ===")
print(f"{'Version':<20} {'Quiet RMS':>10} {'Dyn Range':>10} {'Time':>8}")
for label, r in results.items():
    dr = 20 * np.log10(r["rms"] / r["quiet_rms"]) if r["quiet_rms"] > 0 else 0
    print(f"{label:<20} {r['quiet_rms']:>10.0f} {dr:>9.1f}dB {r['time']:>7.1f}s")
