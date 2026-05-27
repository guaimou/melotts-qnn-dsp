"""Patch TTS module to mock torchaudio before melo import."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# Find the _qnn_load_encoder function and add torchaudio mock before melo import
old = '    from melo import utils as melo_utils'
new = '''    # Mock torchaudio (not needed for encoder, only for audio I/O in utils)
    import types, sys as _sys
    for _mod in ('torchaudio',):
        _m = types.ModuleType(_mod)
        _m.load = lambda *a, **kw: (None, None)
        _m.info = lambda *a, **kw: None
        _sys.modules[_mod] = _m
    from melo import utils as melo_utils'''

if old in content:
    content = content.replace(old, new)
    with open(tts_path, 'w') as f:
        f.write(content)
    print("Patched torchaudio mock in TTS module")
else:
    print("Already patched or pattern not found")
    if 'torchaudio' in content and 'types.ModuleType' in content:
        print("  Mock already present")
    elif 'torchaudio' not in content:
        print("  torchaudio not mentioned in file")
