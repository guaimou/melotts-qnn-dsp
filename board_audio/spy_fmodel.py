"""Spy on .fmodel structure and API internals."""
import sys, os, json

sys.path.insert(0, "/home/fibo/AI model/tts_models")
sys.path.insert(0, "/home/fibo/AI model/llm_models/get_output_from_log_method")

FMODEL = "/home/fibo/AI model/tts_models/fibotts_1.0.0_qcom_6490-8550_qnn_2.26_dsp_02c30b45759c7583f61881af2e1d8c24.fmodel"

# 1. File format check
print("=== 1. File format ===")
with open(FMODEL, "rb") as f:
    header = f.read(16)
print(f"Header hex: {header.hex()}")
print(f"Header ascii: {header}")

# Check if ZIP
if header[:2] == b"PK":
    print("-> ZIP format detected!")
    import zipfile
    with zipfile.ZipFile(FMODEL) as zf:
        for name in zf.namelist()[:30]:
            info = zf.getinfo(name)
            print(f"  {name} ({info.file_size/1024/1024:.1f}MB)")
elif header[:4] == b"\x7fELF":
    print("-> ELF binary")
else:
    print("-> Unknown format, first 256 bytes:")
    with open(FMODEL, "rb") as f:
        data = f.read(256)
    # Try to find readable strings
    import re
    strings = re.findall(b'[ -~]{5,}', data)
    for s in strings[:20]:
        print(f"  '{s.decode('ascii', errors='replace')}'")

# 2. Load with AudioAPI and spy
print("\n=== 2. AudioAPI internals ===")
from fiboaisdk.api_aisdk_py import api_audio_py as au_api

api = au_api.AudioAPI()
# Spy on AudioAPI object
print("AudioAPI methods:")
for m in dir(api):
    if not m.startswith("_"):
        print(f"  {m}")

# 3. Init and spy on what happens
print("\n=== 3. Init and observe ===")
from config import LICENSE_DIR

def read_file(path, mode="r"):
    with open(path, mode) as f:
        return f.read()

from fiboaisdk.api_aisdk_py import license_py as license_api
license_api.Init(
    read_file(os.path.join(LICENSE_DIR, "key1.pem"), "r"),
    read_file(os.path.join(LICENSE_DIR, "key2.pem"), "r"),
    read_file(os.path.join(LICENSE_DIR, "key3.pem"), "r"),
    read_file(os.path.join(LICENSE_DIR, "license.bin"), "rb"),
)

ret = api.Init(FMODEL)
print(f"Init ret={ret}")

# Spy on what's available after init
print("\nAfter Init, attributes:")
for attr in dir(api):
    if not attr.startswith("_") and not callable(getattr(api, attr, None)):
        try:
            val = getattr(api, attr)
            print(f"  {attr}: {val}")
        except:
            pass

# Check internal structures
print("\nInternal type info:")
for attr in ["__class__", "__dict__"]:
    if hasattr(api, attr):
        d = getattr(api, attr)
        if isinstance(d, dict):
            print(f"  {attr}: {list(d.keys())[:20]}")

api.Release()
