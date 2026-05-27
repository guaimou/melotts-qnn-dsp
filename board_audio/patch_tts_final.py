"""Final patch for tts.py: best model + cache fix."""
PATH = "/home/fibo/AI model/tts_models/tts.py"

with open(PATH, "r") as f:
    content = f.read()

# Fix 1: Ensure correct SO path (baseline 30-sample model, best correlation)
for old_path in [
    'lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so',
    'lib_ifixed/aarch64-ubuntu-gcc9.4/libmelotts_ifixed.so',
    'lib_int8/aarch64-ubuntu-gcc9.4/libmelotts_int8.so',
    'lib_p99/aarch64-ubuntu-gcc9.4/libmelotts_p99.so',
    'lib_100/aarch64-ubuntu-gcc9.4/libmelotts_100.so',
]:
    if old_path in content:
        content = content.replace(old_path, 'lib_default/aarch64-ubuntu-gcc9.4/libmelotts_default_dsp.so')
        print("Replaced: %s" % old_path.split('/')[-3])

# Fix 2: Ensure correct QNN params for baseline model
content = content.replace(
    '_QNN_IN_SCALE = 0.0751',
    '_QNN_IN_SCALE = 0.0637778565287590'
)
content = content.replace(
    '_QNN_IN_OFFSET = -121',
    '_QNN_IN_OFFSET = -133'
)
content = content.replace(
    '_QNN_OUT_SCALE = 0.0003921568568330',
    '_QNN_OUT_SCALE = 0.0016496082535014'
)
content = content.replace(
    '_QNN_OUT_OFFSET = 0',
    '_QNN_OUT_OFFSET = -115'
)
# Also ensure the ones from old patches are correct
content = content.replace(
    '_QNN_OUT_SCALE = 0.0010843767086044\n_QNN_OUT_OFFSET = -145',
    '_QNN_OUT_SCALE = 0.0016496082535014\n_QNN_OUT_OFFSET = -115'
)

# Fix 3: Add SO path tracking to prevent stale cache
old = "_qnn_api = None          # InferAPI 实例"
new = "_qnn_api = None          # InferAPI 实例\n_qnn_loaded_so = None     # 已加载的SO路径"
if old in content:
    content = content.replace(old, new)
    print("Added _qnn_loaded_so tracking")

# Fix 4: Update _qnn_get_api to check SO path
old_func = """def _qnn_get_api():
    \"\"\"获取或初始化 QNN DSP 声码器（持久化单例）\"\"\"
    global _qnn_api
    if _qnn_api is not None:
        return _qnn_api"""
new_func = """def _qnn_get_api():
    \"\"\"获取或初始化 QNN DSP 声码器（持久化单例）\"\"\"
    global _qnn_api, _qnn_loaded_so
    if _qnn_api is not None and _qnn_loaded_so == QNN_SO_PATH:
        return _qnn_api"""
if old_func in content:
    content = content.replace(old_func, new_func)
    print("Updated _qnn_get_api")

# Fix 5: Record SO path after successful init
old_init = '_qnn_api = api\n            print("[QNN-TTS] QNN DSP INT8'
# Find the exact pattern
for pattern in [
    '_qnn_api = api\n            print("[QNN-TTS] QNN DSP INT8 声码器初始化成功")',
    '_qnn_api = api\n            print("[QNN-TTS] QNN DSP INT8 声码器初始化成功")',
]:
    if pattern in content:
        content = content.replace(pattern,
            '_qnn_api = api\n            _qnn_loaded_so = QNN_SO_PATH\n            print("[QNN-TTS] QNN DSP INT8 声码器初始化成功")')
        print("Added SO tracking on init")
        break
else:
    # Try grep for the print line
    for i, line in enumerate(content.split('\n'), 1):
        if 'QNN DSP INT8 声码器初始化成功' in line:
            before = '\n'.join(content.split('\n')[:i-1])
            after = '\n'.join(content.split('\n')[i-1:])
            old_line = content.split('\n')[i-2] + '\n' + content.split('\n')[i-1]
            new_line = content.split('\n')[i-2] + '\n            _qnn_loaded_so = QNN_SO_PATH\n' + content.split('\n')[i-1]
            content = content.replace(old_line, new_line)
            print("Added SO tracking before init print")
            break

# Fix 6: release() clears _qnn_loaded_so
old_rel = 'global _qnn_api, _qnn_encoder'
new_rel = 'global _qnn_api, _qnn_encoder, _qnn_loaded_so'
if old_rel in content:
    content = content.replace(old_rel, new_rel)
    print("Added _qnn_loaded_so to release()")

old_rel2 = '_qnn_encoder = None'
new_rel2 = '_qnn_encoder = None\n    _qnn_loaded_so = None'
if old_rel2 in content and '_qnn_loaded_so = None' not in content.split(old_rel2)[1][:50]:
    content = content.replace(old_rel2, new_rel2, 1)  # Only first occurrence in release()
    print("Added _qnn_loaded_so clear in release()")

# Write back
with open(PATH, "w") as f:
    f.write(content)

# Verify
for i, line in enumerate(content.split('\n'), 1):
    if 'QNN_SO_PATH =' in line and 'lib_' in line:
        print("  L%d: %s" % (i, line.strip()))
    if line.strip().startswith('_QNN_IN_SCALE'):
        print("  L%d: %s" % (i, line.strip()))
    if line.strip().startswith('_QNN_OUT_SCALE'):
        print("  L%d: %s" % (i, line.strip()))
    if '_qnn_loaded_so' in line:
        print("  L%d: %s" % (i, line.strip()))

print("Done")
