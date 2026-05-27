"""Update tts.py to use ifixed model with proper input quantization."""
PATH = "/home/fibo/AI model/tts_models/tts.py"

with open(PATH, "r") as f:
    content = f.read()

# 1. Update .so path
old_so = 'QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_int8/aarch64-ubuntu-gcc9.4/libmelotts_int8.so"'
new_so = 'QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_ifixed/aarch64-ubuntu-gcc9.4/libmelotts_ifixed.so"'
content = content.replace(old_so, new_so)

# 2. Update _QNN_OUT_SCALE and _QNN_OUT_OFFSET (they're correct from last fix)
# 3. Replace _qnn_generate to add input pre-quantization

old_func = """def _qnn_generate(z_latent):
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
        return y_float[:valid_T * HOP_LENGTH]"""

new_func = """def _qnn_generate(z_latent):
    _api = _qnn_get_api()
    if _api is None:
        raise RuntimeError("QNN DSP not initialized")

    _, C, T_enc = z_latent.shape

    def _run_chunk(z_nchw_chunk, valid_T):
        z_nhwc = np.transpose(z_nchw_chunk, (0, 2, 1))  # NCHW -> NHWC [1, T, 192]
        # Pre-quantize float -> uint8 using model input encoding
        z_uint8 = np.clip(np.round(z_nhwc / _QNN_IN_SCALE + _QNN_IN_OFFSET), 0, 255).astype(np.uint8)
        ret = _api.Execute_uint8({"z": z_uint8.flatten().tolist()})
        if ret != 0:
            raise RuntimeError(f"QNN Execute_uint8 failed: {ret}")
        y_int8 = np.array(_api.FetchOutputs_int8(["y"])["y"], dtype=np.int8)
        y_float = (y_int8.astype(np.float32) - _QNN_OUT_OFFSET) * _QNN_OUT_SCALE
        y_float = y_float.reshape(1, 1, -1)[0, 0]
        return y_float[:valid_T * HOP_LENGTH]"""

if old_func in content:
    content = content.replace(old_func, new_func)
    print("_qnn_generate replaced")
else:
    print("WARNING: old function not found - checking what exists")
    idx = content.find("def _qnn_generate(z_latent):")
    if idx >= 0:
        print(content[idx:idx+300])

# 4. Also fix speed=0 bug (might have been fixed, but ensure)
old_speed = 'speed=1.0 if speed is None else speed'
new_speed = 'speed=1.0 if speed in (None, 0) else speed'
if old_speed in content:
    content = content.replace(old_speed, new_speed)
    print("speed bug fix applied")

with open(PATH, "w") as f:
    f.write(content)

# Verify
with open(PATH, "r") as f:
    for i, line in enumerate(f, 1):
        if "_QNN_IN_SCALE" in line or "Execute_uint8" in line or "lib_ifixed" in line:
            print(f"  L{i}: {line.rstrip()}")

print("Done")
