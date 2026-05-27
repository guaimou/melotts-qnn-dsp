"""Fix _QNN_PARAMS dict in tts.py"""
PATH = "/home/fibo/AI model/tts_models/tts.py"

with open(PATH, "r") as f:
    content = f.read()

# The broken block (no quotes from shell escaping)
old = """# QNN encoding params per voice: QNN formula float = (uint8 + offset) * scale
_QNN_PARAMS = {
    default: {in_scale: 0.0637778565287590, in_offset: -133, out_scale: 0.0016496082535014, out_offset: -115},
    a2:      {in_scale: 0.0986579582095146, in_offset: -125, out_scale: 0.0022140776272863, out_offset: -100},
    a13:     {in_scale: 0.0508260242640972, in_offset: -133, out_scale: 0.0021459185518324, out_offset: -121},
}
# Legacy accessors
_QNN_IN_SCALE = _QNN_PARAMS[default][in_scale]
_QNN_IN_OFFSET = _QNN_PARAMS[default][in_offset]
_QNN_OUT_SCALE = _QNN_PARAMS[default][out_scale]
_QNN_OUT_OFFSET = _QNN_PARAMS[default][out_offset]"""

new = """# QNN encoding params per voice: QNN formula float = (uint8 + offset) * scale
_QNN_PARAMS = {
    "default": {"in_scale": 0.0637778565287590, "in_offset": -133, "out_scale": 0.0016496082535014, "out_offset": -115},
    "a2":      {"in_scale": 0.0986579582095146, "in_offset": -125, "out_scale": 0.0022140776272863, "out_offset": -100},
    "a13":     {"in_scale": 0.0508260242640972, "in_offset": -133, "out_scale": 0.0021459185518324, "out_offset": -121},
}
# Legacy accessors (using dict keys)
_QNN_IN_SCALE = _QNN_PARAMS["default"]["in_scale"]
_QNN_IN_OFFSET = _QNN_PARAMS["default"]["in_offset"]
_QNN_OUT_SCALE = _QNN_PARAMS["default"]["out_scale"]
_QNN_OUT_OFFSET = _QNN_PARAMS["default"]["out_offset"]"""

if old in content:
    content = content.replace(old, new)
    print("Replaced")
else:
    print("Old block not found - checking...")
    idx = content.find("_QNN_PARAMS = {")
    if idx >= 0:
        print(content[idx:idx+500])

with open(PATH, "w") as f:
    f.write(content)
print("Done")
