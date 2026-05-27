"""Generate clean INT8 and ONNX reference audio for listening."""
import types, sys, time, numpy as np
m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")
import tts
import onnxruntime as ort
from scipy.io import wavfile

text = "你好，这是一个测试语音合成效果。"
QNN_CHUNK_T = 128
HOP_LENGTH = 512

# Encode once
print("Encoding...")
z, T_enc = tts._qnn_encode(text, voice="default", speed=1.0)
print(f"z shape={z.shape}, T={T_enc}")

# 1. ONNX FP32 reference
print("\n=== ONNX FP32 ===")
sess = ort.InferenceSession("/home/fibo/melotts_qnn/generator_cle.onnx", providers=['CPUExecutionProvider'])
audio_onnx_parts = []
pos = 0
overlap_t = 16
hop_t = QNN_CHUNK_T - overlap_t
while pos < T_enc:
    end = min(pos + QNN_CHUNK_T, T_enc)
    chunk_len = end - pos
    z_chunk = z[0:1, :, pos:end]
    if chunk_len < QNN_CHUNK_T:
        z_chunk = np.pad(z_chunk, ((0,0),(0,0),(0,QNN_CHUNK_T-chunk_len)), mode='constant')
    y = sess.run(None, {'z': z_chunk.astype(np.float32)})[0]
    y_f = y.reshape(1, 1, -1)[0, 0]
    valid = y_f[:chunk_len * HOP_LENGTH]
    # Simple concat (no overlap-add for simplicity)
    audio_onnx_parts.append(valid)
    pos += QNN_CHUNK_T  # Non-overlapping chunks for simplicity

audio_onnx = np.concatenate(audio_onnx_parts)
print(f"ONNX FP32: len={len(audio_onnx)}")

# 2. INT8 DSP
print("\n=== INT8 DSP ===")
tts.QNN_SPECSUB_ENABLED = False

# Switch default to INT8 temporarily
# Actually let me use the INT8 .so directly
sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

so_int8 = "/home/fibo/melotts_qnn/lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so"
api_i8 = api_infer_py.InferAPI()
api_i8.Init(api_infer_py.InferParams(so_int8, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5))

IN_SCALE = 0.0635628774762154
IN_OFFSET = -133
OUT_SCALE = 0.0016634501516819
OUT_OFFSET = -117

audio_int8_parts = []
pos = 0
while pos < T_enc:
    end = min(pos + QNN_CHUNK_T, T_enc)
    chunk_len = end - pos
    z_chunk = z[0:1, :, pos:end]
    if chunk_len < QNN_CHUNK_T:
        z_chunk = np.pad(z_chunk, ((0,0),(0,0),(0,QNN_CHUNK_T-chunk_len)), mode='constant')

    z_nhwc = np.transpose(z_chunk, (0, 2, 1))
    z_u8 = np.clip(np.round(z_nhwc / IN_SCALE - IN_OFFSET), 0, 255).astype(np.uint8)
    api_i8.Execute_uint8({"z": z_u8.flatten().tolist()})
    y_u8 = np.array(api_i8.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
    y_f = (y_u8.astype(np.float32) + OUT_OFFSET) * OUT_SCALE
    y_f = y_f.reshape(1, 1, -1)[0, 0]
    valid = y_f[:chunk_len * HOP_LENGTH]
    audio_int8_parts.append(valid)
    pos += QNN_CHUNK_T

audio_int8 = np.concatenate(audio_int8_parts)
print(f"INT8 DSP: len={len(audio_int8)}")

# 3. Save both with DC removal + peak norm only
def save(audio, path):
    a = audio - np.mean(audio)
    peak = np.max(np.abs(a))
    if peak > 0:
        a = a / peak * 0.95
    samples = np.clip(a * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(path, 44100, samples)

save(audio_onnx, "/home/fibo/melotts_qnn/final_onnx.wav")
save(audio_int8, "/home/fibo/melotts_qnn/final_int8.wav")

# 4. Quick compare
min_len = min(len(audio_onnx), len(audio_int8))
a = audio_onnx[:min_len]
b = audio_int8[:min_len]
corr = np.corrcoef(a, b)[0, 1]
sp = np.var(a)
npw = np.var(a - b)
snr = 10 * np.log10(sp / npw) if npw > 0 else float('inf')
print(f"\nCorrelation: {corr:.6f}, SNR: {snr:.1f}dB")
print(f"ONNX first10: {np.round(a[:10], 4)}")
print(f"INT8 first10: {np.round(b[:10], 4)}")
print("\nSaved: final_onnx.wav, final_int8.wav")

api_i8.Release()
