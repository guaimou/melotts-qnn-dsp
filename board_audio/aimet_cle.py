"""AIMET Cross-Layer Equalization for MeloTTS HiFiGAN decoder."""
import sys, os, torch, numpy as np, types

for mod_name in ('torchaudio', 'librosa'):
    m = types.ModuleType(mod_name)
    m.load = lambda *a, **kw: (None, None)
    m.info = lambda *a, **kw: None
    sys.modules[mod_name] = m

sys.path.insert(0, '/project/aimet_work')
from melo.models import SynthesizerTrn
from melo import utils as melo_utils

print("=" * 60)
print("AIMET CLE — Cross-Layer Equalization")
print("=" * 60)

# 1. Load model
print("\n[1/4] Loading model...")
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
ckpt = torch.load('/project/aimet_work/G_default.pth', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model'], strict=True)
print("  %d params" % sum(p.numel() for p in model.parameters()))

dec = model.dec
dec.eval()

# 2. Apply CLE (before removing weight_norm — CLE operates on Conv weights)
print("\n[2/4] Applying CLE...")
import aimet_torch.cross_layer_equalization as cle_mod
from aimet_torch.cross_layer_equalization import equalize_model

# LeakyReLU is scale-equivariant but not in whitelist. Add it.
original_activations = cle_mod.cls_supported_activations
cle_mod.cls_supported_activations = original_activations + (torch.nn.LeakyReLU,)
print("  Added LeakyReLU to CLE activation whitelist")

dummy_input = torch.randn(1, 192, 128)
equalize_model(dec, input_shapes=(1, 192, 128), dummy_input=dummy_input)
print("  CLE complete")

# Restore original whitelist
cle_mod.cls_supported_activations = original_activations

# 3. Remove weight_norm for ONNX export
print("\n[3/4] Removing weight_norm + exporting ONNX...")
from torch.nn.utils import remove_weight_norm
if hasattr(dec, 'remove_weight_norm'):
    dec.remove_weight_norm()
for _, module in dec.named_modules():
    try:
        remove_weight_norm(module)
    except ValueError:
        pass

class GeneratorWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
    def forward(self, z):
        return self.dec(z, g=None)

wrapper = GeneratorWrapper(dec)
wrapper.eval()

out_path = '/project/aimet_work/generator_cle.onnx'
torch.onnx.export(
    wrapper, (dummy_input,), out_path,
    input_names=['z'], output_names=['y'],
    dynamic_axes={'z': {0: 'N', 2: 'T'}, 'y': {0: 'N', 2: 'T_audio'}},
    opset_version=17, do_constant_folding=True,
)
print("  Exported: %s (%.1f MB)" % (out_path, os.path.getsize(out_path)/1024/1024))

# 4. Validate
print("\n[4/4] Validating...")
import onnxruntime as ort
sess_orig = ort.InferenceSession('/project/cosyvoice_snpe_snpe/generator_default.onnx', providers=['CPUExecutionProvider'])
sess_cle = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])

# Load a few calibration samples for testing
calib_dir = '/project/calib_real'
calib_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.raw')])[:10]
corrs, diffs = [], []
for f in calib_files:
    z = np.fromfile(os.path.join(calib_dir, f), dtype=np.float32).reshape(1, 192, 128)
    out_orig = sess_orig.run(None, {'z': z})[0].squeeze()
    out_cle = sess_cle.run(None, {'z': z})[0].squeeze()
    diffs.append(np.abs(out_orig - out_cle).max())
    corrs.append(np.corrcoef(out_orig, out_cle)[0, 1])

print("  CLE vs Original FP32:")
print("    diff_max: mean=%.6f, max=%.6f" % (np.mean(diffs), np.max(diffs)))
print("    corr:     mean=%.6f, min=%.6f" % (np.mean(corrs), np.min(corrs)))

# Random input
z_rand = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
out_orig = sess_orig.run(None, {'z': z_rand})[0].squeeze()
out_cle = sess_cle.run(None, {'z': z_rand})[0].squeeze()
diff_rand = np.abs(out_orig - out_cle).max()
corr_rand = np.corrcoef(out_orig, out_cle)[0, 1]
print("  Random: diff_max=%.6f, corr=%.6f" % (diff_rand, corr_rand))

print("\nDone: %s" % out_path)
