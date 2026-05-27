"""Fix tts.py: v2 model + correct QNN encoding + uint8 I/O."""
PATH = "/home/fibo/AI model/tts_models/tts.py"

with open(PATH, "r") as f:
    content = f.read()

# 1. Fix .so path
content = content.replace(
    'QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_int8/aarch64-ubuntu-gcc9.4/libmelotts_int8.so"',
    'QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so"'
)
content = content.replace(
    'QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_ifixed/aarch64-ubuntu-gcc9.4/libmelotts_ifixed.so"',
    'QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so"'
)

# 2. Fix I/O encoding params (v2 model, from gen262_int8_v2.cpp)
# Input: UFIXED_POINT_8, scale=0.0637778565287590, offset=-133
# Output: UFIXED_POINT_8, scale=0.0010843767086044, offset=-145
# QNN formula: float = (uint8 + offset) * scale
# Encoding: uint8 = clamp(round(float/scale - offset), 0, 255)
old_qnn_params = """# QNN INT8 quantization parameters (from gen262_int8.cpp)
_QNN_IN_SCALE = 0.0197250004857779
_QNN_IN_OFFSET = -126
_QNN_OUT_SCALE = 0.0006642817752436
_QNN_OUT_OFFSET = -152"""

new_qnn_params = """# QNN INT8 quantization parameters (from gen262_int8_v2.cpp, real z_latent calibration)
# QNN formula: float = (uint8 + offset) * scale
# Encode: uint8 = clamp(round(float/scale - offset), 0, 255)
_QNN_IN_SCALE = 0.0637778565287590
_QNN_IN_OFFSET = -133
_QNN_OUT_SCALE = 0.0010843767086044
_QNN_OUT_OFFSET = -145"""

content = content.replace(old_qnn_params, new_qnn_params)

# Also fix the variant that was patched earlier
content = content.replace(
    '_QNN_IN_SCALE = 0.06376',
    '_QNN_IN_SCALE = 0.0637778565287590'
)

# 3. Replace _qnn_generate with correct uint8 I/O + QNN formula
old_func_marker = "def _qnn_generate(z_latent):"
end_marker = "def _qnn_synthesize_to_wav(text, output_path, voice=None, speed=1.0):"

idx_start = content.index(old_func_marker)
idx_end = content.index(end_marker)

new_func = """def _qnn_generate(z_latent):
    _api = _qnn_get_api()
    if _api is None:
        raise RuntimeError("QNN DSP not initialized")

    _, C, T_enc = z_latent.shape

    def _run_chunk(z_nchw_chunk, valid_T):
        z_nhwc = np.transpose(z_nchw_chunk, (0, 2, 1))  # NCHW->NHWC [1,T,192]
        # QNN encoding: uint8 = clamp(round(float/scale - offset), 0, 255)
        z_uint8 = np.clip(np.round(z_nhwc / _QNN_IN_SCALE - _QNN_IN_OFFSET), 0, 255).astype(np.uint8)
        ret = _api.Execute_uint8({"z": z_uint8.flatten().tolist()})
        if ret != 0:
            raise RuntimeError(f"QNN Execute_uint8 failed: {ret}")
        y_uint8 = np.array(_api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
        # QNN decoding: float = (uint8 + offset) * scale
        y_float = (y_uint8.astype(np.float32) + _QNN_OUT_OFFSET) * _QNN_OUT_SCALE
        y_float = y_float.reshape(1, 1, -1)[0, 0]
        return y_float[:valid_T * HOP_LENGTH]

    if T_enc <= QNN_CHUNK_T:
        pad_len = QNN_CHUNK_T - T_enc
        z_pad = np.pad(z_latent[0], ((0, 0), (0, pad_len)), mode="constant")
        z_pad = z_pad[np.newaxis, :, :]
        return _run_chunk(z_pad, T_enc)

    # Multi-chunk inference with overlap-add
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

content = content[:idx_start] + new_func + content[idx_end:]

with open(PATH, "w") as f:
    f.write(content)

# Verify
with open(PATH, "r") as f:
    for i, line in enumerate(f, 1):
        if "_QNN_IN_SCALE" in line or "_QNN_OUT_SCALE" in line or "Execute_uint8" in line or "FetchOutputs_uint8" in line or "lib_int8_v2" in line:
            print(f"  L{i}: {line.rstrip()}")

print("Done - tts.py fully updated for v2 model")
