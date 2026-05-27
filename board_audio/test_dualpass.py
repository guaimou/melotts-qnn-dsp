"""Dual-pass INT8 with half-step offset — effectively ~9-bit resolution."""
import types, sys, time, numpy as np

m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
from tts import _qnn_get_api, _QNN_PARAMS, QNN_CHUNK_T, HOP_LENGTH
from scipy.io import wavfile

VOICE = "default_cle"
SO_PATH = "/home/fibo/melotts_qnn/lib_cle_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_cle_sqnr.so"

print("=== Dual-Pass Residual INT8 ===")
print(f"Voice: {VOICE}")

# Get API and params
_api = _qnn_get_api(VOICE)
if _api is None:
    print("Need to init API via _qnn_get_api... loading model")
    # Force init through the standard path
    sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
    from fiboaisdk.api_aisdk_py import api_infer_py
    params = api_infer_py.InferParams(SO_PATH, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
    _api = api_infer_py.InferAPI()
    _api.Init(params)

p = _QNN_PARAMS[VOICE]
in_scale, in_offset = p["in_scale"], p["in_offset"]
out_scale, out_offset = p["out_scale"], p["out_offset"]

print(f"Params: in_scale={in_scale:.6f}, in_offset={in_offset}")
print(f"        out_scale={out_scale:.6f}, out_offset={out_offset}")

# Helper: single DSP inference
def dsp_infer(z_nchw, api, in_scl, in_off, out_scl, out_off):
    """Run one DSP INT8 inference. z_nchw: [1, C, T]"""
    z_nhwc = np.transpose(z_nchw, (0, 2, 1))  # NCHW -> NHWC [1,T,C]
    z_u8 = np.clip(np.round(z_nhwc / in_scl - in_off), 0, 255).astype(np.uint8)
    api.Execute_uint8({"z": z_u8.flatten().tolist()})
    y_u8 = np.array(api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
    return (y_u8.astype(np.float32) + out_off) * out_scl

def dsp_infer_dual(z_nchw, api, in_scl, in_off, out_scl, out_off):
    """Dual-pass: standard + half-step offset, then average."""
    # Pass 1: standard encoding
    audio_a = dsp_infer(z_nchw, api, in_scl, in_off, out_scl, out_off)

    # Pass 2: half-step offset (0.5 LSB shift)
    audio_b = dsp_infer(z_nchw, api, in_scl, in_off - 0.5, out_scl, out_off)

    # Average
    return (audio_a + audio_b) / 2.0, audio_a, audio_b

# Test with random z_latent
print("\n[1] Single chunk test (T=128)...")
z_test = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

t0 = time.time()
audio_dual, audio_a, audio_b = dsp_infer_dual(z_test, _api, in_scale, in_offset, out_scale, out_offset)
elapsed = time.time() - t0
print(f"  Dual-pass: {elapsed:.3f}s (2x DSP calls)")
print(f"  Pass A std={audio_a.std():.6f}, Pass B std={audio_b.std():.6f}")
print(f"  Dual std={audio_dual.std():.6f}")
print(f"  Pass A vs B correlation: {np.corrcoef(audio_a, audio_b)[0,1]:.6f}")

# Test: single pass for timing comparison
t0 = time.time()
audio_single = dsp_infer(z_test, _api, in_scale, in_offset, out_scale, out_offset)
single_time = time.time() - t0
print(f"  Single-pass: {single_time:.3f}s")
print(f"  Overhead: {elapsed - 2*single_time:.3f}s ({(elapsed/2/single_time - 1)*100:.0f}% per pass)")

# Full TTS test
print("\n[2] Full TTS test...")
from tts import _qnn_encode
text = "你好，这是一个测试语音合成效果。"

t0 = time.time()
z_latent, T_enc = _qnn_encode(text, voice=VOICE, speed=1.0)
enc_time = time.time() - t0
print(f"  Encoder: {enc_time:.1f}s, T={T_enc}")

# Chunked dual-pass processing
print(f"  Running dual-pass vocoder (T={T_enc})...")
overlap_t = 16
hop_t = QNN_CHUNK_T - overlap_t
audio_parts_a = []
audio_parts_b = []
fade_in = np.linspace(0, 1, overlap_t * HOP_LENGTH)
fade_out = np.linspace(1, 0, overlap_t * HOP_LENGTH)

t0 = time.time()
pos = 0
n_chunks = 0
while pos < T_enc:
    end = min(pos + QNN_CHUNK_T, T_enc)
    chunk_len = end - pos
    z_chunk = z_latent[0:1, :, pos:end]

    if chunk_len < QNN_CHUNK_T:
        pad_len = QNN_CHUNK_T - chunk_len
        z_chunk = np.pad(z_chunk, ((0, 0), (0, 0), (0, pad_len)), mode="constant")

    # Dual-pass for this chunk
    valid_samples = chunk_len * HOP_LENGTH
    y_dual, y_a, y_b = dsp_infer_dual(z_chunk, _api, in_scale, in_offset, out_scale, out_offset)
    y_dual = y_dual.reshape(1, 1, -1)[0, 0][:valid_samples]
    y_a = y_a.reshape(1, 1, -1)[0, 0][:valid_samples]

    if pos > 0:
        y_dual[:len(fade_in)] *= fade_in
        y_dual[:len(fade_in)] += audio_parts_a[-1][-len(fade_out):]

    audio_parts_a.append(y_dual)
    pos += hop_t
    n_chunks += 1

audio = np.concatenate(audio_parts_a)
voc_time = time.time() - t0
print(f"  {n_chunks} chunks: {voc_time:.1f}s ({voc_time/n_chunks*1000:.0f}ms/chunk)")
print(f"  Total TTS: {enc_time + voc_time:.1f}s")

# Save
audio = audio - np.mean(audio)
peak = np.max(np.abs(audio))
if peak > 0:
    audio = audio / peak * 0.95
samples = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
wavfile.write("/home/fibo/melotts_qnn/test_dualpass.wav", 44100, samples)

# Noise analysis
af = audio
win = int(44100 * 0.01)
n_win = len(af) // win
energies = [np.std(af[i*win:(i+1)*win]) for i in range(n_win)]
quiet_rms = np.mean([e for e in energies if e < np.percentile(energies, 10)])

# Compare with single-pass reference
print(f"\n[3] Comparing with single-pass...")
t0 = time.time()
pos = 0
audio_sp = []
while pos < T_enc:
    end = min(pos + QNN_CHUNK_T, T_enc)
    chunk_len = end - pos
    z_chunk = z_latent[0:1, :, pos:end]
    if chunk_len < QNN_CHUNK_T:
        z_chunk = np.pad(z_chunk, ((0, 0), (0, 0), (0, QNN_CHUNK_T - chunk_len)), mode="constant")
    y = dsp_infer(z_chunk, _api, in_scale, in_offset, out_scale, out_offset)
    y = y.reshape(1, 1, -1)[0, 0][:chunk_len * HOP_LENGTH]
    if pos > 0:
        y[:len(fade_in)] *= fade_in
        y[:len(fade_in)] += audio_sp[-1][-len(fade_out):]
    audio_sp.append(y)
    pos += hop_t

audio_s = np.concatenate(audio_sp)
sp_time = time.time() - t0
audio_s = audio_s - np.mean(audio_s)
peak_s = np.max(np.abs(audio_s))
if peak_s > 0:
    audio_s = audio_s / peak_s * 0.95
samples_s = np.clip(audio_s * 32767, -32768, 32767).astype(np.int16)
wavfile.write("/home/fibo/melotts_qnn/test_single.wav", 44100, samples_s)

# Compare noise floors
af_s = audio_s
e_s = [np.std(af_s[i*win:(i+1)*win]) for i in range(len(af_s)//win)]
quiet_s = np.mean([e for e in e_s if e < np.percentile(e_s, 10)])

print(f"\n=== Results ===")
print(f"Dual-pass: quiet_rms={quiet_rms:.4f}, voc_time={voc_time:.1f}s")
print(f"Single:    quiet_rms={quiet_s:.4f}, voc_time={sp_time:.1f}s")
print(f"Improvement: {(1 - quiet_rms/quiet_s)*100:.1f}% quieter")

# Also save non-normalized versions for comparison
wavfile.write("/home/fibo/melotts_qnn/test_dualpass_raw.wav", 44100, (audio * 32767 * 0.95).astype(np.int16))
wavfile.write("/home/fibo/melotts_qnn/test_single_raw.wav", 44100, (audio_s * 32767 * 0.95).astype(np.int16))
print("Saved raw versions for blind comparison")
