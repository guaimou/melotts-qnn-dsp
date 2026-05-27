"""Add torchaudio mock before melo import in tts.py."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

old = '''    from deploy import _install_local_download_utils
    _install_local_download_utils()

    from melo import utils as melo_utils'''

new = '''    from deploy import _install_local_download_utils
    _install_local_download_utils()

    # Mock torchaudio — not needed for encoder, melo/utils imports it at module level
    import types as _types
    _ta = _types.ModuleType("torchaudio")
    _ta.load = lambda *a, **kw: (None, None)
    _ta.info = lambda *a, **kw: None
    sys.modules["torchaudio"] = _ta

    from melo import utils as melo_utils'''

content = content.replace(old, new)

with open(tts_path, 'w') as f:
    f.write(content)

print("Mock added." if 'sys.modules["torchaudio"]' in content else "FAILED")
