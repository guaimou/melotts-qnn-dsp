"""Fix corrupted _QNN_PARAMS section in tts.py"""
PATH = "/home/fibo/AI model/tts_models/tts.py"

with open(PATH, "r") as f:
    content = f.read()

# Find the corrupted block from "# QNN INT8 quantization" to the last "out_offset" before "def _qnn_generate"
start_marker = "# QNN INT8 quantization parameters"
end_marker = "def _qnn_generate(z_latent, voice=\"default\"):"

start_idx = content.index(start_marker)
end_idx = content.index(end_marker)

correct_block = """# QNN INT8 quantization parameters (SQNR calibration, from gen_default_sqnr.cpp)
# QNN formula: float = (uint8 + offset) * scale
# Encode: uint8 = clamp(round(float/scale - offset), 0, 255)
_QNN_PARAMS = {
    "default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016485477099195, "out_offset": -115},
    "a2":      {"in_scale": 0.0986679643392563, "in_offset": -124, "out_scale": 0.0022215729113668, "out_offset": -100},
    "a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021487122867256, "out_offset": -121},
}
# Legacy accessors
_QNN_IN_SCALE = _QNN_PARAMS["default"]["in_scale"]
_QNN_IN_OFFSET = _QNN_PARAMS["default"]["in_offset"]
_QNN_OUT_SCALE = _QNN_PARAMS["default"]["out_scale"]
_QNN_OUT_OFFSET = _QNN_PARAMS["default"]["out_offset"]

"""

content = content[:start_idx] + correct_block + content[end_idx:]

with open(PATH, "w") as f:
    f.write(content)
print("Fixed")
