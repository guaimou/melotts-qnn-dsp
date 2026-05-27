"""Test FP32 DSP inference directly via fiboaisdk."""
import sys, time, numpy as np

sys.path.insert(0, "/home/fibo/AI model/tts_models")
sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")

from fiboaisdk.api_aisdk_py import api_infer_py

so_path = "/home/fibo/melotts_qnn/lib_cle_fp32/aarch64-ubuntu-gcc9.4/libmelotts_cle_fp32.so"

print("=== QNN DSP FP32 Test ===")
print(f"Model: {so_path}")

# Init API
print("\n[1] Initializing DSP...")
t0 = time.time()
params = api_infer_py.InferParams(so_path, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)
print(f"  Init done in {time.time()-t0:.1f}s")

# Test with random z_latent
print("\n[2] Testing FP32 inference...")
z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
z_nhwc = np.transpose(z, (0, 2, 1))  # [1, 128, 192]

t0 = time.time()
ret = api.Execute_float({"z": z_nhwc.flatten().tolist()})
print(f"  Execute_float: {time.time()-t0:.3f}s, ret={ret}")

t0 = time.time()
y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
print(f"  FetchOutputs: {time.time()-t0:.3f}s, shape={y.shape}")

print(f"  Output: len={len(y)}, min={y.min():.6f}, max={y.max():.6f}, std={y.std():.6f}")

# Speed test - multiple runs
print("\n[3] Speed test (5 runs)...")
times = []
for i in range(5):
    z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
    z_nhwc = np.transpose(z, (0, 2, 1))
    t0 = time.time()
    api.Execute_float({"z": z_nhwc.flatten().tolist()})
    y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.3f}s")

avg = np.mean(times[1:])  # skip first (cold)
print(f"\n  Average (warm): {avg*1000:.1f}ms/chunk, RTF={avg/(65536/44100):.3f}")

# Compare with INT8 PCH speed
print("\n[4] Running full TTS with FP32 via custom path...")
import types
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_encode

text = "你好，测试。"
print(f"  Encoding: '{text}'")
t0 = time.time()
z_latent, T_enc = _qnn_encode(text, voice="default", speed=1.0)
print(f"  Encoder: {time.time()-t0:.1f}s, z shape={z_latent.shape}, T={T_enc}")

# Run FP32 vocoder manually on the encoded z
print(f"\n  Running FP32 vocoder (T={T_enc})...")
QNN_CHUNK_T = 128
HOP_LENGTH = 512
overlap_t = 16
hop_t = QNN_CHUNK_T - overlap_t

# Pad to chunk size if needed
if T_enc <= QNN_CHUNK_T:
    pad_len = QNN_CHUNK_T - T_enc
    z_pad = np.pad(z_latent[0], ((0, 0), (0, pad_len)), mode="constant")
    z_nhwc = np.transpose(z_pad[np.newaxis, :, :], (0, 2, 1))
    t0 = time.time()
    api.Execute_float({"z": z_nhwc.flatten().tolist()})
    y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
    y = y.reshape(1, 1, -1)[0, 0]
    audio = y[:T_enc * HOP_LENGTH]
    print(f"  Single chunk: {time.time()-t0:.3f}s")
else:
    # Multi-chunk
    t0 = time.time()
    audio_parts = []
    fade_in = np.linspace(0, 1, overlap_t * HOP_LENGTH)
    fade_out = np.linspace(1, 0, overlap_t * HOP_LENGTH)
    pos = 0
    while pos < T_enc:
        end = min(pos + QNN_CHUNK_T, T_enc)
        chunk_len = end - pos
        z_chunk = z_latent[0, :, pos:end]
        if chunk_len < QNN_CHUNK_T:
            z_chunk = np.pad(z_chunk, ((0, 0), (0, QNN_CHUNK_T - chunk_len)), mode="constant")
        z_nhwc = np.transpose(z_chunk[np.newaxis, :, :], (0, 2, 1))
        api.Execute_float({"z": z_nhwc.flatten().tolist()})
        y = np.array(api.FetchOutputs_float(["y"])["y"], dtype=np.float32)
        y_f = y.reshape(1, 1, -1)[0, 0]
        valid_out = y_f[:chunk_len * HOP_LENGTH]

        if pos > 0:
            valid_out[:len(fade_in)] *= fade_in
            audio_parts[-1][-len(fade_out):] *= fade_out
            valid_out[:len(fade_in)] += audio_parts[-1][-len(fade_out):]

        audio_parts.append(valid_out)
        pos += hop_t

    audio = np.concatenate(audio_parts)

    total_time = time.time() - t0
    chunks = (T_enc + hop_t - 1) // hop_t
    print(f"  {chunks} chunks: {total_time:.3f}s total, {total_time/chunks*1000:.1f}ms/chunk")

from scipy.io import wavfile
audio = audio - np.mean(audio)
peak = np.max(np.abs(audio))
if peak > 0:
    audio = audio / peak * 0.95
samples = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
wavfile.write("/home/fibo/melotts_qnn/test_fp32.wav", 44100, samples)
print(f"\n  Saved test_fp32.wav, len={len(audio)}, dur={len(audio)/44100:.2f}s")
api.Release()
