"""Generate HTP context binary ON the board using native tools."""
import subprocess, os, sys

# We need a .so model. Use the INT8 .so we built earlier
SO_PATH = "/home/fibo/melotts_qnn/lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so"
SDK = "/home/fibo/qnn_sdk_262"
CTX_BIN = "/home/fibo/melotts_qnn/htp_ctx_onboard.bin"
BACKEND = f"{SDK}/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so"
GENERATOR = f"{SDK}/bin/aarch64-ubuntu-gcc9.4/qnn-context-binary-generator"

print(f"SO: {SO_PATH}")
print(f"Backend: {BACKEND}")
print(f"Output: {CTX_BIN}")

# First verify SO exists
if not os.path.exists(SO_PATH):
    print(f"ERROR: SO not found at {SO_PATH}")
    sys.exit(1)

# Generate context binary
cmd = [
    GENERATOR,
    f"--model={SO_PATH}",
    f"--backend={BACKEND}",
    f"--binary_file={CTX_BIN}",
    f"--output_dir=/home/fibo/melotts_qnn/",
]
print(f"Running: {' '.join(cmd)}")

result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print("STDOUT:", result.stdout[-1000:] if result.stdout else "")
print("STDERR:", result.stderr[-1000:] if result.stderr else "")
print(f"Return code: {result.returncode}")

# Check output
for f in os.listdir("/home/fibo/melotts_qnn/"):
    if "htp_ctx" in f:
        fpath = os.path.join("/home/fibo/melotts_qnn/", f)
        print(f"  {f}: {os.path.getsize(fpath)/1024/1024:.1f}MB")
