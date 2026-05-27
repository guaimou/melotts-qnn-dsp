"""Update tts.py with per-channel models."""
PATH = "/home/fibo/AI model/tts_models/tts.py"
with open(PATH, "r") as f:
    c = f.read()

# Update SO paths
c = c.replace("lib_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_sqnr.so", "lib_pch/aarch64-ubuntu-gcc9.4/libmelotts_pch.so")
c = c.replace("lib_a2_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_a2_sqnr.so", "lib_a2_pch/aarch64-ubuntu-gcc9.4/libmelotts_a2_pch.so")
c = c.replace("lib_a13_sqnr/aarch64-ubuntu-gcc9.4/libmelotts_a13_sqnr.so", "lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so")

# Update _QNN_PARAMS
idx = c.index("_QNN_PARAMS = {")
idx_end = c.index("}", idx) + 1

new_dict = '''_QNN_PARAMS = {
    "default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016914557199925, "out_offset": -117},
    "a2":      {"in_scale": 0.0986679643392563, "in_offset": -124, "out_scale": 0.0022239189129323, "out_offset": -99},
    "a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021189446561038, "out_offset": -119},
}'''

c = c[:idx] + new_dict + c[idx_end+1:]

# Also update legacy accessors
leg_idx = c.find("# Legacy accessors")
leg_end = c.find("def _qnn_generate", leg_idx)
c = c[:leg_idx] + c[leg_end:]

with open(PATH, "w") as f:
    f.write(c)
print("Updated to per-channel models")
