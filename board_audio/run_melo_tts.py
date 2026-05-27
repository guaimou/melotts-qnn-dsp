"""Run melo_tts.py synthesize_to_file, save to writable path."""
import sys, os
os.chdir('/home/fibo/melotts_qnn')

# Patch OUTPUT_DIR before importing melo_tts
sys.path.insert(0, '/home/fibo/AI model/tts_models/before')

# Direct import of the internal synthesize function (bypass the module-level OUTPUT_DIR creation)
# melo_tts uses a MeloTTS class internally
import importlib.util
spec = importlib.util.spec_from_file_location(
    'melo_tts_mod',
    '/home/fibo/AI model/tts_models/before/melo_tts.py',
    submodule_search_locations=[]
)

# We need to override OUTPUT_DIR before the module body runs
# Easiest: just run the internal functions directly
# The melo_tts module creates a MeloTTS class. Let's use it.

import torch
import numpy as np
from pathlib import Path

RUNTIME_DIR = Path('/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/runtime')
MODELS_DIR = Path('/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/models')

sys.path.insert(0, str(RUNTIME_DIR))
from melo.api import TTS

# Initialize TTS
speed = 1.0
device = 'cpu'

checkpoint_path = MODELS_DIR / 'default_zh' / 'G_default.pth'
config_path = MODELS_DIR / 'default_zh' / 'config.json'

model = TTS(
    language='ZH_MIX_EN',
    device=device,
    use_hifigan=False,
    config_path=str(config_path),
    ckpt_path=str(checkpoint_path),
)

speaker_ids = model.hps.data.spk2id
speaker_id = speaker_ids.get('ZH', list(speaker_ids.values())[0])

text = '你好，这是一个测试语音合成效果。'
print(f'Generating: {text}')

t0 = time.time()
audio = model.tts_to_audio(text, speaker_id=speaker_id, speed=speed)
elapsed = time.time() - t0
print(f'Done in {elapsed:.1f}s, len={len(audio)}')

# Save
import soundfile as sf
out_path = '/home/fibo/melotts_qnn/melo_fp32_test.wav'
sf.write(out_path, audio, 44100)
print(f'Saved to {out_path}')
