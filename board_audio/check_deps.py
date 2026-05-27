import importlib
for mod in ['torchaudio', 'txtsplit', 'librosa', 'scipy', 'numpy']:
    try:
        importlib.import_module(mod)
        print(f"  {mod}: OK")
    except ImportError:
        print(f"  {mod}: MISSING")
