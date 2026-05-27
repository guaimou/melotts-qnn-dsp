"""AIMET AdaRound v3 — weight rounding optimization for MeloTTS HiFiGAN decoder.

API: aimet-torch 2.31.0, Adaround.apply_adaround (correct API for v2.x).
Output: FP32 ONNX with AdaRound-optimized weights.
Then: QNN converter with per-channel SQNR for best INT8 results.
"""
import sys, os, torch, numpy as np
import time

# Mock torchaudio + librosa before melo import (not needed for model loading)
import types
for mod_name in ('torchaudio', 'librosa'):
    m = types.ModuleType(mod_name)
    m.__dict__.update({k: (lambda *a, **kw: None) for k in ('load', 'info', 'get_duration', 'resample')})
    sys.modules[mod_name] = m

sys.path.insert(0, '/project/aimet_work')

from melo.models import SynthesizerTrn
from melo import utils as melo_utils
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 4
NUM_CALIB = 30  # Use all 30 calibration samples
NUM_ITER = 2000  # AdaRound iterations (enough for convergence)

print("=" * 60)
print("AIMET AdaRound v3 — MeloTTS HiFiGAN Decoder")
print("=" * 60)

# 1. Load model and extract decoder
print("\n[1/5] Loading model...")
ckpt_path = '/project/aimet_work/G_default.pth'
config_path = '/project/aimet_work/config.json'
hps = melo_utils.get_hparams_from_file(config_path)
hps.data.disable_bert = True

model = SynthesizerTrn(
    len(hps.symbols), hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers, num_tones=hps.num_tones,
    num_languages=hps.num_languages, **hps.model,
)
model.eval()
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model'], strict=True)
print("  Model loaded: %d params" % sum(p.numel() for p in model.parameters()))

dec = model.dec
dec.eval()

# Remove weight_norm — required for AIMET deepcopy
# Use model's built-in method, then catch any remaining via recursive cleanup
from torch.nn.utils import remove_weight_norm
if hasattr(dec, 'remove_weight_norm'):
    dec.remove_weight_norm()
# Also handle conv_pre, conv_post and any stray weight_norm
for name, module in dec.named_modules():
    try:
        remove_weight_norm(module)
    except ValueError:
        pass
print("  weight_norm removed from decoder")

class GeneratorWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z):
        return self.dec(z, g=None)

wrapper = GeneratorWrapper(dec)
wrapper.eval()
print("  Wrapper ready")

# 2. Load calibration data
print("\n[2/5] Loading calibration data...")
calib_dir = '/project/cosyvoice_snpe_snpe/calib_real'
calib_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.raw')])
calib_files = calib_files[:NUM_CALIB]

all_z = []
for f in calib_files:
    z = np.fromfile(os.path.join(calib_dir, f), dtype=np.float32).reshape(1, 192, 128)
    all_z.append(torch.from_numpy(z).squeeze(0))  # [192, 128]

calib_tensor = torch.stack(all_z)  # [N, 192, 128]
dataset = TensorDataset(calib_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print("  %d samples, %d batches (batch_size=%d)" % (
    len(calib_tensor), len(dataloader), BATCH_SIZE))

# 3. Run AdaRound
print("\n[3/5] Running AdaRound optimization...")
print("  Iterations: %d" % NUM_ITER)
dummy_input = torch.randn(1, 192, 128)

adaround_params = AdaroundParameters(
    data_loader=dataloader,
    num_batches=len(dataloader),
    default_num_iterations=NUM_ITER,
)

t0 = time.time()
optimized_model = Adaround.apply_adaround(
    model=wrapper,
    dummy_input=dummy_input,
    params=adaround_params,
    path='/project/aimet_work',
    filename_prefix='generator_adaround',
    default_param_bw=8,
)
elapsed = time.time() - t0
print("  AdaRound complete in %.1f seconds" % elapsed)

# 4. Export AdaRound-optimized FP32 ONNX
print("\n[4/5] Exporting AdaRound-optimized ONNX...")
out_path = '/project/aimet_work/generator_adaround.onnx'
torch.onnx.export(
    optimized_model,
    (dummy_input,),
    out_path,
    input_names=['z'],
    output_names=['y'],
    dynamic_axes={'z': {0: 'N', 2: 'T'}, 'y': {0: 'N', 2: 'T_audio'}},
    opset_version=17,
    do_constant_folding=True,
)
size_mb = os.path.getsize(out_path) / 1024 / 1024
print("  Exported: %s (%.1f MB)" % (out_path, size_mb))

# 5. Validate — compare AdaRound model output vs original
print("\n[5/5] Validating...")
import onnxruntime as ort

# Original ONNX (Option B generator)
orig_onnx = '/project/cosyvoice_snpe_snpe/generator_default.onnx'
sess_orig = ort.InferenceSession(orig_onnx, providers=['CPUExecutionProvider'])
sess_ada = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])

# Test with calibration samples
corrs = []
diffs_max = []
for i in range(min(10, len(calib_tensor))):
    z_test = calib_tensor[i:i+1].numpy()
    out_orig = sess_orig.run(None, {'z': z_test})[0].squeeze()
    out_ada = sess_ada.run(None, {'z': z_test})[0].squeeze()
    diff = np.abs(out_orig - out_ada)
    corr = np.corrcoef(out_orig, out_ada)[0, 1]
    corrs.append(corr)
    diffs_max.append(diff.max())

print("  AdaRound vs Original FP32:")
print("    diff_max: mean=%.6f, max=%.6f" % (np.mean(diffs_max), np.max(diffs_max)))
print("    corr:     mean=%.6f, min=%.6f" % (np.mean(corrs), np.min(corrs)))

# Also test with random noise input
z_rand = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
out_orig = sess_orig.run(None, {'z': z_rand})[0].squeeze()
out_ada = sess_ada.run(None, {'z': z_rand})[0].squeeze()
diff_rand = np.abs(out_orig - out_ada)
corr_rand = np.corrcoef(out_orig, out_ada)[0, 1]
print("  Random input: diff_max=%.6f, corr=%.6f" % (diff_rand.max(), corr_rand))

print("\nDone — AdaRound-optimized ONNX saved to %s" % out_path)
