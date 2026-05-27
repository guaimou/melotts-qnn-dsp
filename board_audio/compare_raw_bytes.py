"""Compare raw output bytes between INT8 and W8A16 for SAME input."""
import sys, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py

z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# INT8 DSP
print("=== INT8 DSP ===")
p_i8 = api_infer_py.InferParams(
    "/home/fibo/melotts_qnn/lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so",
    'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api_i8 = api_infer_py.InferAPI()
api_i8.Init(p_i8)

# Manual INT8 encoding
z_nhwc = np.transpose(z, (0, 2, 1))
z_u8 = np.clip(np.round(z_nhwc / 0.0635628774762154 + 133), 0, 255).astype(np.uint8)
api_i8.Execute_uint8({"z": z_u8.flatten().tolist()})
raw_i8 = np.array(api_i8.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
print(f"INT8 raw bytes: len={len(raw_i8)}, min={raw_i8.min()}, max={raw_i8.max()}, mean={raw_i8.mean():.1f}")

# W8A16 DSP
print("\n=== W8A16 DSP ===")
p_w8 = api_infer_py.InferParams(
    "/home/fibo/melotts_qnn/lib_cle_w8a16/aarch64-ubuntu-gcc9.4/libmelotts_cle_w8a16.so",
    'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api_w8 = api_infer_py.InferAPI()
api_w8.Init(p_w8)

# Manual INT16 encoding for W8A16
z_u16 = np.clip(np.round(z_nhwc / 0.0002481628616806 + 34127), 0, 65535).astype(np.uint16)
# Need to pass as bytes since there's no Execute_uint16
z_bytes = z_u16.view(np.uint8).flatten().tolist()
try:
    ret = api_w8.Execute_uint8({"z": z_bytes})
    print(f"Execute_uint8 (with uint16 input): ret={ret}")
    raw_w8 = np.array(api_w8.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
    print(f"W8A16 raw bytes: len={len(raw_w8)}, min={raw_w8.min()}, max={raw_w8.max()}, mean={raw_w8.mean():.1f}")
except Exception as e:
    print(f"Execute_uint8 with uint16 input failed: {e}")

    # Fall back to Execute_float
    print("\nFalling back to Execute_float...")
    api_w8.Execute_float({"z": z.flatten().tolist()})
    raw_w8 = np.array(api_w8.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
    print(f"W8A16 raw bytes (float input): len={len(raw_w8)}, min={raw_w8.min()}, max={raw_w8.max()}, mean={raw_w8.mean():.1f}")

# Compare raw bytes
print(f"\nComparison:")
print(f"INT8 output bytes: {raw_i8[:20]}")
print(f"W8A16 output bytes: {raw_w8[:20]}")
corr_bytes = np.corrcoef(raw_i8.astype(float), raw_w8.astype(float))[0, 1] if len(raw_i8) == len(raw_w8) else 0
print(f"Byte correlation: {corr_bytes:.6f}")

api_i8.Release()
api_w8.Release()
