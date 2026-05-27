"""Patch _qnn_generate in tts.py: NHWC layout + INT8 dequant."""
import re

PATH = "/home/fibo/AI model/tts_models/tts.py"

with open(PATH, "r") as f:
    content = f.read()

old_start = "def _qnn_generate(z_latent):"
old_end = "def _qnn_synthesize_to_wav(text, output_path, voice=None, speed=1.0):"

start_idx = content.index(old_start)
end_idx = content.index(old_end)

new_function = """# QNN INT8 quantization parameters (from gen262_int8.cpp)
_QNN_IN_SCALE = 0.0197250004857779
_QNN_IN_OFFSET = -126
_QNN_OUT_SCALE = 0.0006642817752436
_QNN_OUT_OFFSET = -152


def _qnn_generate(z_latent):
    _api = _qnn_get_api()
    if _api is None:
        raise RuntimeError("QNN DSP not initialized")

    _, C, T_enc = z_latent.shape

    def _run_chunk(z_nchw_chunk, valid_T):
        z_nhwc = np.transpose(z_nchw_chunk, (0, 2, 1))
        ret = _api.Execute_float({"z": z_nhwc.flatten().tolist()})
        if ret != 0:
            raise RuntimeError(f"QNN Execute_float failed: {ret}")
        y_int8 = np.array(_api.FetchOutputs_int8(["y"])["y"], dtype=np.int8)
        y_float = (y_int8.astype(np.float32) - _QNN_OUT_OFFSET) * _QNN_OUT_SCALE
        y_float = y_float.reshape(1, 1, -1)[0, 0]
        return y_float[:valid_T * HOP_LENGTH]

    if T_enc <= QNN_CHUNK_T:
        pad_len = QNN_CHUNK_T - T_enc
        z_pad = np.pad(z_latent[0], ((0, 0), (0, pad_len)), mode="constant")
        z_pad = z_pad[np.newaxis, :, :]
        return _run_chunk(z_pad, T_enc)

    overlap_t = 16
    hop_t = QNN_CHUNK_T - overlap_t
    audio_parts = []
    fade_in = np.linspace(0, 1, overlap_t * HOP_LENGTH)
    fade_out = np.linspace(1, 0, overlap_t * HOP_LENGTH)
    prev_tail = None

    pos = 0
    while pos < T_enc:
        end = min(pos + QNN_CHUNK_T, T_enc)
        chunk_len = end - pos

        if chunk_len < QNN_CHUNK_T:
            pad_len = QNN_CHUNK_T - chunk_len
            z_chunk = np.pad(z_latent[0, :, pos:end], ((0, 0), (0, pad_len)), mode="constant")
            z_chunk = z_chunk[np.newaxis, :, :]
        else:
            z_chunk = z_latent[:, :, pos:end]

        audio_chunk = _run_chunk(z_chunk, min(chunk_len, QNN_CHUNK_T))

        if prev_tail is not None and overlap_t > 0:
            cross_len = overlap_t * HOP_LENGTH
            if cross_len <= len(prev_tail) and cross_len <= len(audio_chunk):
                prev_tail[-cross_len:] = (
                    prev_tail[-cross_len:] * fade_out[:cross_len] +
                    audio_chunk[:cross_len] * fade_in[:cross_len]
                )
                audio_parts.append(prev_tail[:-cross_len])
                prev_tail = audio_chunk
            else:
                audio_parts.append(prev_tail)
                prev_tail = audio_chunk
        else:
            if prev_tail is not None:
                audio_parts.append(prev_tail)
            prev_tail = audio_chunk

        pos += hop_t

    if prev_tail is not None:
        audio_parts.append(prev_tail)

    return np.concatenate(audio_parts)


"""

new_content = content[:start_idx] + new_function + content[end_idx:]

with open(PATH, "w") as f:
    f.write(new_content)

# Verify
with open(PATH, "r") as f:
    for i, line in enumerate(f, 1):
        if "_QNN_OUT_SCALE" in line or "_run_chunk" in line:
            print(f"  L{i}: {line.rstrip()}")

print("Patch applied successfully")
