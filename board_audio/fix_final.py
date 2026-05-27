"""Clean replacement of QNN params section in tts.py"""
PATH = "/home/fibo/AI model/tts_models/tts.py"
with open(PATH, "r") as f:
    lines = f.readlines()

# Find the QNN section boundaries
start_line = None
end_line = None
for i, line in enumerate(lines):
    if "# QNN INT8 quantization parameters" in line or "# QNN formula" in line:
        if start_line is None:
            start_line = i
    if start_line is not None and line.startswith("def _qnn_generate"):
        end_line = i
        break
    if start_line is not None and line.strip().startswith("_api = _qnn_get_api"):
        # Missing 'def' line - include this as end
        end_line = i - 1
        break

if start_line is None:
    print("ERROR: Could not find QNN params start")
    exit(1)
if end_line is None:
    end_line = start_line + 20

print("Replacing lines %d to %d" % (start_line + 1, end_line + 1))

correct_block = [
    '# QNN INT8 quantization parameters (per-channel SQNR calibration)\n',
    '# QNN formula: float = (uint8 + offset) * scale\n',
    '# Encode: uint8 = clamp(round(float/scale - offset), 0, 255)\n',
    '_QNN_PARAMS = {\n',
    '    "default": {"in_scale": 0.0635628774762154, "in_offset": -133, "out_scale": 0.0016914557199925, "out_offset": -117},\n',
    '    "a2":      {"in_scale": 0.0986679643392563, "in_offset": -124, "out_scale": 0.0022239189129323, "out_offset": -99},\n',
    '    "a13":     {"in_scale": 0.0507732145488262, "in_offset": -133, "out_scale": 0.0021189446561038, "out_offset": -119},\n',
    '}\n',
    '\n',
    '\n',
]

# Replace
lines[start_line:end_line] = correct_block

with open(PATH, "w") as f:
    f.writelines(lines)

# Verify syntax
import py_compile
try:
    py_compile.compile(PATH, doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print("Syntax error:", e)
