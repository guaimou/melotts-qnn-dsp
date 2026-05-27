"""Create input and run qnn-net-run with context binary."""
import numpy as np, os, subprocess

z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
# Encode as uint8 for INT8 model
z_nhwc = np.transpose(z, (0, 2, 1))  # [1, 128, 192]
z_u8 = np.clip(np.round(z_nhwc / 0.06356 + 133), 0, 255).astype(np.uint8)

# Save as raw
z_u8.tofile("/tmp/qnn_input.raw")
print(f"Input saved: {len(z_u8)} uint8 values")

# Create input list file
with open("/tmp/input_list.txt", "w") as f:
    f.write("/tmp/qnn_input.raw\n")

SDK = "/home/fibo/qnn_sdk_262"
ctx = "/home/fibo/melotts_qnn/htp_context.bin"
backend = f"{SDK}/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so"

cmd = [
    f"{SDK}/bin/aarch64-ubuntu-gcc9.4/qnn-net-run",
    f"--retrieve_context={ctx}",
    f"--backend={backend}",
    "--input_list=/tmp/input_list.txt",
    "--output_dir=/tmp/qnn_output",
    "--use_native_input_files",
    "--use_native_output_files",
]

print(f"Running: {' '.join(cmd)}")
os.makedirs("/tmp/qnn_output", exist_ok=True)
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print("STDOUT:", result.stdout[-500:] if result.stdout else "")
print("STDERR:", result.stderr[-500:] if result.stderr else "")
print(f"Return code: {result.returncode}")

# Check output
out_dir = "/tmp/qnn_output"
if os.path.exists(out_dir):
    for f in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f}: {size} bytes")
        if f.endswith(".raw") and size > 0:
            data = np.fromfile(fpath, dtype=np.uint8)
            print(f"    uint8: len={len(data)}, min={data.min()}, max={data.max()}")
