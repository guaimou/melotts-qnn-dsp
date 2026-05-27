"""Fix _QNN_PARAMS dict quotes in tts.py"""
PATH = "/home/fibo/AI model/tts_models/tts.py"
with open(PATH, "r") as f:
    content = f.read()

# Fix unquoted dict keys
replacements = [
    ("    default: {in_scale:", '    "default": {"in_scale":'),
    ("    a2:      {in_scale:", '    "a2":      {"in_scale":'),
    ("    a13:     {in_scale:", '    "a13":     {"in_scale":'),
    (": {in_scale: 0.", ': {"in_scale": 0.'),
    (", in_offset:", ', "in_offset":'),
    (", out_scale:", ', "out_scale":'),
    (", out_offset:", ', "out_offset":'),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print("Fixed:", old[:40])

with open(PATH, "w") as f:
    f.write(content)

# Also fix the missing 'def' line if needed
lines = content.split("\n")
for i, line in enumerate(lines):
    if line.strip() == "}" and i > 650 and i < 670:
        # Check if def is missing after the closing brace
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and lines[j].startswith("    ") and not any(
            l.startswith("def ") for l in lines[i:j]
        ):
            lines.insert(j, "def _qnn_generate(z_latent, voice=\"default\"):")
            content = "\n".join(lines)
            with open(PATH, "w") as f:
                f.write(content)
            print("Inserted missing def _qnn_generate")
            break

print("Done")
