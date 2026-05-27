import numpy as np
from scipy.io import wavfile

sr, base = wavfile.read("/home/fibo/melotts_qnn/test_base_norm.wav")
sr2, ng = wavfile.read("/home/fibo/melotts_qnn/test_noise_gate.wav")
sr3, ss = wavfile.read("/home/fibo/melotts_qnn/test_spectral_sub.wav")

def noise_floor(a):
    win = 441
    n = len(a) // win
    e = [np.std(a[i*win:(i+1)*win]) for i in range(n)]
    return np.mean([x for x in e if x < np.percentile(e, 10)])

print("Int16-level noise floor:")
print(f"  Base:         {noise_floor(base.astype(float)):.0f}")
print(f"  Noise Gate:   {noise_floor(ng.astype(float)):.0f}")
print(f"  Spectral Sub: {noise_floor(ss.astype(float)):.0f}")

print(f"\nTotal RMS:")
print(f"  Base:         {base.astype(float).std():.0f}")
print(f"  Noise Gate:   {ng.astype(float).std():.0f}")
print(f"  Spectral Sub: {ss.astype(float).std():.0f}")

# Compare quiet segment sample values
for label, a in [("Base", base), ("Gate", ng), ("SpecSub", ss)]:
    af = a.astype(float)
    win = 441
    n = len(af) // win
    e = [np.std(af[i*win:(i+1)*win]) for i in range(n)]
    q = np.mean([x for x in e if x < np.percentile(e, 10)])
    print(f"{label:10s}: quiet_std={q:.0f}, min={a.min()}, max={a.max()}")
