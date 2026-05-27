"""Check fiboaisdk capabilities - packaging, model format, etc."""
from fiboaisdk.api_aisdk_py import api_audio_py, api_infer_py

print("=== api_audio_py (AudioAPI - TTS) ===")
audio_attrs = [x for x in dir(api_audio_py) if not x.startswith("_")]
print("  Classes/functions:", audio_attrs)

# Check AudioAPI methods
print("\n  AudioAPI methods:")
for attr in audio_attrs:
    obj = getattr(api_audio_py, attr)
    if "method" in str(type(obj)) or "function" in str(type(obj)):
        print(f"    {attr}")
    elif "type" in str(type(obj)):
        try:
            methods = [m for m in dir(obj) if not m.startswith("_")]
            print(f"    {attr}: {methods}")
        except:
            print(f"    {attr}: (class)")

print("\n=== api_infer_py (InferAPI - general) ===")
infer_attrs = [x for x in dir(api_infer_py) if not x.startswith("_")]
print("  Classes/functions:", infer_attrs)

# Check InferAPI methods
for attr in infer_attrs:
    obj = getattr(api_infer_py, attr)
    if "type" in str(type(obj)):
        try:
            methods = [m for m in dir(obj) if not m.startswith("_")]
            print(f"  {attr}: {methods}")
        except:
            print(f"  {attr}: (class)")

print("\n=== FiboAudioFormat options ===")
for fmt_name in dir(api_audio_py):
    if "FORMAT" in fmt_name or "FIBO" in fmt_name:
        print(f"  {fmt_name}")

# Check if there's model pack/create functionality
import fiboaisdk
print("\n=== fiboaisdk package info ===")
print("  path:", fiboaisdk.__path__)
print("  dsp_lib_path:", fiboaisdk.dsp_lib_path)
print("  qcom_dsp_version:", fiboaisdk.qcom_dsp_version)
