import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m
sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_synthesize_to_wav
from scipy.io import wavfile

text = "你好，这是一个测试语音合成效果。"

print("Warm run comparison:")
for voice, label in [("default_cle", "CLE"), ("default_ada_sqnr", "ASQNR"), ("default", "PCH")]:
    t0 = time.time()
    _qnn_synthesize_to_wav(text, f"/tmp/r1_{voice}.wav", voice=voice, speed=1.0)
    elapsed = time.time() - t0
    sr, a = wavfile.read(f"/tmp/r1_{voice}.wav")
    af = a.astype(float)
    win = int(sr*0.01)
    nw = len(af)//win
    e = [np.std(af[i*win:(i+1)*win]) for i in range(nw)]
    q = np.mean([x for x in e if x < np.percentile(e,10)])
    print(f"  {label:6s}: {elapsed:.1f}s, quiet_rms={q:.0f}, len={len(a)}, dur={len(a)/sr:.2f}s")
