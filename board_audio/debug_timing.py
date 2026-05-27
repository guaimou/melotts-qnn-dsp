"""Debug DSP timing breakdown."""
import types, sys, time, numpy as np

m = types.ModuleType("torchaudio")
m.load = lambda *a, **kw: (None, None)
m.info = lambda *a, **kw: None
sys.modules["torchaudio"] = m

sys.path.insert(0, "/home/fibo/AI model/tts_models")

from tts import _qnn_get_api, _QNN_PARAMS, QNN_CHUNK_T, HOP_LENGTH

# Get API for default_ada
print("Getting API...")
t0 = time.time()
_api = _qnn_get_api("default_ada")
print(f"API init: {time.time()-t0:.2f}s")

p = _QNN_PARAMS["default_ada"]
in_scale, in_offset = p["in_scale"], p["in_offset"]
out_scale, out_offset = p["out_scale"], p["out_offset"]
print(f"Params: in_scale={in_scale}, in_offset={in_offset}")
print(f"        out_scale={out_scale}, out_offset={out_offset}")

# Test with a single chunk of exactly 128
print("\n=== Single chunk test (T=128) ===")
z_test = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
z_nhwc = np.transpose(z_test, (0, 2, 1))  # [1, 128, 192]

print(f"z shape: {z_nhwc.shape}")
t0 = time.time()
z_uint8 = np.clip(np.round(z_nhwc / in_scale - in_offset), 0, 255).astype(np.uint8)
print(f"Quantize: {time.time()-t0:.4f}s, uint8 range=[{z_uint8.min()}, {z_uint8.max()}]")

flat_data = z_uint8.flatten().tolist()
print(f"Flatten: {len(flat_data)} elements")

t0 = time.time()
ret = _api.Execute_uint8({"z": flat_data})
print(f"Execute_uint8: {time.time()-t0:.3f}s, ret={ret}")

t0 = time.time()
y_uint8 = np.array(_api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
print(f"FetchOutputs: {time.time()-t0:.3f}s, shape={y_uint8.shape}")

t0 = time.time()
y_float = (y_uint8.astype(np.float32) + out_offset) * out_scale
print(f"Decode: {time.time()-t0:.4f}s")
print(f"Output: range=[{y_float.min():.6f}, {y_float.max():.6f}], len={len(y_float)}")

# Test 2 - same with default pch for comparison
print("\n=== Single chunk test (PCH, T=128) ===")
_api2 = _qnn_get_api("default")
p2 = _QNN_PARAMS["default"]
in_scale2, in_offset2 = p2["in_scale"], p2["in_offset"]
out_scale2, out_offset2 = p2["out_scale"], p2["out_offset"]

z_nhwc2 = np.transpose(z_test, (0, 2, 1))
z_uint82 = np.clip(np.round(z_nhwc2 / in_scale2 - in_offset2), 0, 255).astype(np.uint8)
flat2 = z_uint82.flatten().tolist()

t0 = time.time()
ret2 = _api2.Execute_uint8({"z": flat2})
et = time.time() - t0
print(f"Execute_uint8: {et:.3f}s, ret={ret2}")
y_uint82 = np.array(_api2.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
print(f"Output shape={y_uint82.shape}")
