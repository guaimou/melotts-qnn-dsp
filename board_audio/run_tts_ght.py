"""Run tts_ght AudioAPI to generate audio file only (no playback)."""
import sys, os, json, tempfile, time

# Hardcoded config from get_output_from_log_method/config.py
TTS_MODEL = "/home/fibo/TTS/TTS"
LICENSE_DIR = "/home/fibo/qcom_6490_license"
CARD_NAME = "lahainayupikiot"
SAMPLE_RATE = 16000
CHANNELS = 1

# Make config importable before importing tts_ght
sys.path.insert(0, '/home/fibo/AI model/llm_models/get_output_from_log_method')
sys.path.insert(0, '/home/fibo/AI model/tts_models')

def read_file(path, mode="r"):
    with open(path, mode) as f:
        return f.read()

text = '你好，这是一个测试语音合成效果。'
out_path = '/home/fibo/melotts_qnn/tts_ght_output.wav'

print(f"Text: {text}")
print(f"Output: {out_path}")
print(f"Model: {TTS_MODEL}")

# Init license manually (tts_ght's init_license reads from its own config import)
from fiboaisdk.api_aisdk_py import license_py as license_api
ret = license_api.Init(
    read_file(os.path.join(LICENSE_DIR, "key1.pem"), "r"),
    read_file(os.path.join(LICENSE_DIR, "key2.pem"), "r"),
    read_file(os.path.join(LICENSE_DIR, "key3.pem"), "r"),
    read_file(os.path.join(LICENSE_DIR, "license.bin"), "rb"),
)
if ret != 0:
    raise RuntimeError(f"license init failed: {ret}")
print("License OK")

from fiboaisdk.api_aisdk_py import api_audio_py as au_api
api = au_api.AudioAPI()

ret = api.Init(TTS_MODEL)
if ret != 0:
    raise RuntimeError(f"AudioAPI init failed: {ret}")
print("API init OK")

# Synthesize
out_audio = au_api.FiboAudio()
out_audio.audio_sample_rate = SAMPLE_RATE
out_audio.audio_channel = CHANNELS
out_audio.audio_format = au_api.FiboAudioFormat.FIBO_AUDIO_FORMAT_WAV
out_audio.audio_path = out_path
out_audio.extra_params = json.dumps({"language": "zh"})

t0 = time.perf_counter()
syn_ret = api.SpeechSynthesisSync(text, out_audio)
elapsed = time.perf_counter() - t0

if syn_ret != 0:
    print(f"SpeechSynthesisSync failed: {syn_ret}")
else:
    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"Done in {elapsed:.1f}s, file size={size} bytes")

api.Release()
