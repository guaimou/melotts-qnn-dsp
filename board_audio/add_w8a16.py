"""Add W8A16 voice to TTS module and patch _qnn_generate."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# 1. Add SO_MAP entry
old_so = '"a13":     "/home/fibo/melotts_qnn/lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so",'
new_so = old_so + '\n    "default_w8a16": "/home/fibo/melotts_qnn/lib_cle_w8a16/aarch64-ubuntu-gcc9.4/libmelotts_cle_w8a16.so",'
content = content.replace(old_so, new_so)

# 2. Add VOICE_MAP entry
old_vm = '"a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),'
new_vm = old_vm + '\n    "default_w8a16": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),'
content = content.replace(old_vm, new_vm)

# 3. Add QNN_PARAMS with W8A16 encoding + mode flag
old_p = '"a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021189446561038, "out_offset": -119},'
new_p = old_p + '\n    "default_w8a16": {"w8a16": True, "in_scale": 0.0002481628616806, "in_offset": -34127, "out_scale": 0.0000305175781250, "out_offset": -32768},'
content = content.replace(old_p, new_p)

# 4. Modify _qnn_generate to support W8A16 mode
old_gen = '''    p = _QNN_PARAMS[voice]
    in_scale, in_offset = p["in_scale"], p["in_offset"]
    out_scale, out_offset = p["out_scale"], p["out_offset"]

    _, C, T_enc = z_latent.shape

    def _run_chunk(z_nchw_chunk, valid_T):
        z_nhwc = np.transpose(z_nchw_chunk, (0, 2, 1))  # NCHW->NHWC [1,T,192]
        # QNN encoding: uint8 = clamp(round(float/scale - offset), 0, 255)
        z_uint8 = np.clip(np.round(z_nhwc / in_scale - in_offset), 0, 255).astype(np.uint8)
        ret = _api.Execute_uint8({"z": z_uint8.flatten().tolist()})
        if ret != 0:
            raise RuntimeError(f"QNN Execute_uint8 failed: {ret}")
        y_uint8 = np.array(_api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
        # QNN decoding: float = (uint8 + offset) * scale
        y_float = (y_uint8.astype(np.float32) + out_offset) * out_scale
        y_float = y_float.reshape(1, 1, -1)[0, 0]
        return y_float[:valid_T * HOP_LENGTH]'''

is_w8a16 = 'p.get("w8a16", False)'

new_gen = '''    p = _QNN_PARAMS[voice]
    is_w8a16 = p.get("w8a16", False)
    in_scale, in_offset = p["in_scale"], p["in_offset"]
    out_scale, out_offset = p["out_scale"], p["out_offset"]

    _, C, T_enc = z_latent.shape

    def _run_chunk(z_nchw_chunk, valid_T):
        if is_w8a16:
            # W8A16: Execute_float input (auto-quantize), uint8→uint16 output
            ret = _api.Execute_float({"z": z_nchw_chunk.flatten().tolist()})
            if ret != 0:
                raise RuntimeError(f"QNN Execute_float failed: {ret}")
            y_u8 = np.array(_api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
            y_u16 = y_u8.view(np.uint16)  # reinterpret bytes as uint16
            y_float = (y_u16.astype(np.float32) + out_offset) * out_scale
            y_float = y_float.reshape(1, 1, -1)[0, 0]
            return y_float[:valid_T * HOP_LENGTH]
        else:
            # INT8: manual quantize, Execute_uint8, dequantize
            z_nhwc = np.transpose(z_nchw_chunk, (0, 2, 1))  # NCHW->NHWC [1,T,192]
            z_uint8 = np.clip(np.round(z_nhwc / in_scale - in_offset), 0, 255).astype(np.uint8)
            ret = _api.Execute_uint8({"z": z_uint8.flatten().tolist()})
            if ret != 0:
                raise RuntimeError(f"QNN Execute_uint8 failed: {ret}")
            y_uint8 = np.array(_api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
            y_float = (y_uint8.astype(np.float32) + out_offset) * out_scale
            y_float = y_float.reshape(1, 1, -1)[0, 0]
            return y_float[:valid_T * HOP_LENGTH]'''

content = content.replace(old_gen, new_gen)

# 5. Ensure spectral subtraction code exists (may have been lost in restore)
if '_apply_spectral_subtraction' not in content:
    specsub_fn = '''
def _apply_spectral_subtraction(audio, sr=44100, alpha=2.0, beta=0.01):
    """频谱减法降噪"""
    n_fft = 2048
    hop = 512
    win = np.hanning(n_fft)
    n_frames = (len(audio) - n_fft) // hop + 1
    if n_frames < 3:
        return audio
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        frame = audio[i*hop:i*hop+n_fft] * win
        stft[:, i] = np.fft.rfft(frame)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    frame_energies = np.sum(magnitude**2, axis=0)
    quiet_thresh = np.percentile(frame_energies, 20)
    quiet_mask = frame_energies < quiet_thresh
    noise_mag = np.median(magnitude[:, quiet_mask], axis=1) if np.sum(quiet_mask) > 0 else np.zeros(n_fft//2+1)
    clean_mag = np.maximum(magnitude - alpha * noise_mag[:, np.newaxis], beta * magnitude)
    clean_stft = clean_mag * np.exp(1j * phase)
    clean_frames = np.fft.irfft(clean_stft, n=n_fft, axis=0)
    out = np.zeros(len(audio))
    for i in range(n_frames):
        out[i*hop:i*hop+n_fft] += clean_frames[:, i] * win
    return out[:len(audio)]
'''
    # Insert before _qnn_synthesize_to_wav
    old_fn = 'def _qnn_synthesize_to_wav(text, output_path, voice=None, speed=1.0):'
    content = content.replace(old_fn, specsub_fn + '\n' + old_fn)

# Add specsub config if missing
if 'QNN_SPECSUB_ENABLED' not in content:
    old_cfg = 'SAMPLE_RATE = 44100    # 模型采样率'
    new_cfg = 'SAMPLE_RATE = 44100    # 模型采样率\nQNN_SPECSUB_ENABLED = True   # 频谱减法降噪\nQNN_SPECSUB_ALPHA = 2.0      # 过减因子\nQNN_SPECSUB_BETA = 0.01      # 频谱底'
    content = content.replace(old_cfg, new_cfg)

# Add specsub call in pipeline if missing
if 'QNN_SPECSUB_ENABLED' in content and 'Step 2b' not in content:
    old_step = '    # Step 3: Remove DC offset and normalize, then write WAV (44100Hz, S16_LE)\n    audio = audio - np.mean(audio)'
    new_step = '    # Step 2b: 频谱减法降噪\n    if QNN_SPECSUB_ENABLED:\n        audio = _apply_spectral_subtraction(audio, sr=SAMPLE_RATE, alpha=QNN_SPECSUB_ALPHA, beta=QNN_SPECSUB_BETA)\n\n    # Step 3: Remove DC offset and normalize, then write WAV (44100Hz, S16_LE)\n    audio = audio - np.mean(audio)'
    content = content.replace(old_step, new_step)

with open(tts_path, 'w') as f:
    f.write(content)

# Verify
print("Verification:")
for key in ['default_w8a16', 'is_w8a16', '_apply_spectral_subtraction', 'QNN_SPECSUB_ENABLED', 'Execute_float']:
    count = content.count(key)
    print(f"  {key}: {count}")
