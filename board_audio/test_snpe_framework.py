from fiboaisdk.api_aisdk_py import api_infer_py

# Check SNPE framework support
for fw in ["QNN", "SNPE"]:
    for dev in ["DSP", "CPU"]:
        try:
            p = api_infer_py.InferParams("/home/fibo/TTS/TTS", "QUALCOMM", fw, dev, "ERROR", 5)
            print(f"{fw}+{dev}: OK")
        except Exception as e:
            print(f"{fw}+{dev}: FAIL - {e}")
