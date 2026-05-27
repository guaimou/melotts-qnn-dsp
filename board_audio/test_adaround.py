"""Test AdaRound TTS on board."""
import sys, time, numpy as np, os
sys.path.insert(0, '/home/fibo/AI model/tts_models')
from tts import _qnn_synthesize_to_wav
from scipy.io import wavfile

text = '你好，这是一个测试语音合成效果。'
print(f"Testing: '{text}' with voice=default_ada")
t0 = time.time()

tmp_wav = '/tmp/test_adaround.wav'
_qnn_synthesize_to_wav(text, tmp_wav, voice='default_ada', speed=1.0)
elapsed = time.time() - t0
print(f'TTS done in {elapsed:.1f}s')

sr, audio = wavfile.read(tmp_wav)
print(f'Audio: len={len(audio)}, sr={sr}, duration={len(audio)/sr:.2f}s')
audio_f = audio.astype(np.float32) / 32768.0
print(f'Stats: min={audio_f.min():.4f}, max={audio_f.max():.4f}, mean={audio_f.mean():.6f}, std={audio_f.std():.6f}')

# Copy to stable location
os.system(f'cp {tmp_wav} /home/fibo/melotts_qnn/test_adaround.wav')
print('Saved to /home/fibo/melotts_qnn/test_adaround.wav')

# Also test with original pch for comparison
print(f"\nComparing with original default (pch)...")
t0 = time.time()
tmp_pch = '/tmp/test_pch.wav'
_qnn_synthesize_to_wav(text, tmp_pch, voice='default', speed=1.0)
elapsed_pch = time.time() - t0
print(f'TTS done in {elapsed_pch:.1f}s')
sr2, audio_pch = wavfile.read(tmp_pch)
print(f'Audio: len={len(audio_pch)}, duration={len(audio_pch)/sr2:.2f}s')
os.system(f'cp {tmp_pch} /home/fibo/melotts_qnn/test_pch_compare.wav')
print('Saved to /home/fibo/melotts_qnn/test_pch_compare.wav')
