"""Add per-voice QNN model routing to tts.py."""
PATH = "/home/fibo/AI model/tts_models/tts.py"

with open(PATH, "r") as f:
    c = f.read()

# 1. Replace single QNN_SO_PATH with voice map
old_so = 'QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_default/aarch64-ubuntu-gcc9.4/libmelotts_default_dsp.so"'
new_so = '''# QNN DSP .so per voice
QNN_SO_MAP = {
    "default": "/home/fibo/melotts_qnn/lib_default/aarch64-ubuntu-gcc9.4/libmelotts_default_dsp.so",
    "a2":      "/home/fibo/melotts_qnn/lib_a2/aarch64-ubuntu-gcc9.4/libmelotts_a2.so",
    "a13":     "/home/fibo/melotts_qnn/lib_a13/aarch64-ubuntu-gcc9.4/libmelotts_a13.so",
}
# For backward compat
QNN_SO_PATH = QNN_SO_MAP["default"]'''

c = c.replace(old_so, new_so)

# 2. Add per-voice QNN encoding params
old_params = '''_QNN_IN_SCALE = 0.0637778565287590
_QNN_IN_OFFSET = -133
_QNN_OUT_SCALE = 0.0016496082535014
_QNN_OUT_OFFSET = -115'''

new_params = '''# QNN encoding params per voice (QNN formula: float = (uint8 + offset) * scale)
_QNN_PARAMS = {
    "default": {"in_scale": 0.0637778565287590, "in_offset": -133, "out_scale": 0.0016496082535014, "out_offset": -115},
    "a2":      {"in_scale": 0.0986579582095146, "in_offset": -125, "out_scale": 0.0022140776272863, "out_offset": -100},
    "a13":     {"in_scale": 0.0508260242640972, "in_offset": -133, "out_scale": 0.0021459185518324, "out_offset": -121},
}
_QNN_IN_SCALE = _QNN_PARAMS["default"]["in_scale"]
_QNN_IN_OFFSET = _QNN_PARAMS["default"]["in_offset"]
_QNN_OUT_SCALE = _QNN_PARAMS["default"]["out_scale"]
_QNN_OUT_OFFSET = _QNN_PARAMS["default"]["out_offset"]'''

c = c.replace(old_params, new_params)

# 3. Update _qnn_generate to use voice-specific params and SO
old_gen = '''def _qnn_generate(z_latent):
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
        return y_float[:valid_T * HOP_LENGTH]'''

new_gen = '''def _qnn_generate(z_latent, voice="default"):
    _api = _qnn_get_api(voice)
    if _api is None:
        raise RuntimeError("QNN DSP not initialized")

    p = _QNN_PARAMS[voice]
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

c = c.replace(old_gen, new_gen)

# 4. Update _qnn_get_api to accept voice parameter
old_api = '''def _qnn_get_api():
    \"\"\"获取或初始化 QNN DSP 声码器（持久化单例）\"\"\"
    global _qnn_api, _qnn_loaded_so
    if _qnn_api is not None and _qnn_loaded_so == QNN_SO_PATH:
        return _qnn_api'''

new_api = '''def _qnn_get_api(voice="default"):
    \"\"\"获取或初始化 QNN DSP 声码器（持久化单例，按音色路由）\"\"\"
    global _qnn_api, _qnn_loaded_so
    so_path = QNN_SO_MAP.get(voice, QNN_SO_MAP["default"])
    if _qnn_api is not None and _qnn_loaded_so == so_path:
        return _qnn_api'''

c = c.replace(old_api, new_api)

# 5. Update SO path usage inside _qnn_get_api
old_so_ref = 'QNN_SO_PATH, "QUALCOMM", "QNN", "DSP", "ERROR", 5'
new_so_ref = 'so_path, "QUALCOMM", "QNN", "DSP", "ERROR", 5'
c = c.replace(old_so_ref, new_so_ref)

# 6. Update _qnn_loaded_so assignment
old_loaded = '_qnn_loaded_so = QNN_SO_PATH'
new_loaded = '_qnn_loaded_so = so_path'
c = c.replace(old_loaded, new_loaded)

# 7. Update _qnn_synthesize_to_wav to pass voice to _qnn_generate
old_synth = 'audio = _qnn_generate(z_latent)  # float32'
new_synth = 'audio = _qnn_generate(z_latent, voice=voice)  # float32'
c = c.replace(old_synth, new_synth)

with open(PATH, "w") as f:
    f.write(c)

# Verify
for i, line in enumerate(c.split('\n'), 1):
    if 'QNN_SO_MAP' in line and '{' in line:
        print("L%d: %s" % (i, line.strip()))
    if '_QNN_PARAMS' in line and '{' in line:
        print("L%d: %s" % (i, line.strip()))
    if 'def _qnn_get_api' in line:
        print("L%d: %s" % (i, line.strip()))
    if 'def _qnn_generate' in line:
        print("L%d: %s" % (i, line.strip()))
    if 'so_path, "QUALCOMM"' in line:
        print("L%d: %s" % (i, line.strip()))
    if '_qnn_generate(z_latent, voice=' in line:
        print("L%d: %s" % (i, line.strip()))

print("Done")
