"""AIMET Bias Correction only — adjust biases to reduce quantization noise."""
import sys, os, torch, numpy as np, time, types

for mod_name in ('torchaudio', 'librosa'):
    m = types.ModuleType(mod_name)
    m.__dict__.update({k: (lambda *a, **kw: None) for k in ('load', 'info', 'get_duration', 'resample')})
    sys.modules[mod_name] = m

sys.path.insert(0, '/project/aimet_work')
from melo.models import SynthesizerTrn
from melo import utils as melo_utils
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("AIMET Bias Correction — MeloTTS HiFiGAN")
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

from torch.nn.utils import remove_weight_norm
if hasattr(dec, 'remove_weight_norm'):
    dec.remove_weight_norm()
for _, module in dec.named_modules():
    try:
        remove_weight_norm(module)
    except ValueError:
        pass
print("  weight_norm removed")

class GeneratorWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
    def forward(self, z):
        return self.dec(z, g=None)

wrapper = GeneratorWrapper(dec)
wrapper.eval()

# 2. Load calibration data
print("\n[2/4] Loading calibration data...")
calib_dir = '/project/calib_real'
calib_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.raw')])[:30]
all_z = []
for f in calib_files:
    z = np.fromfile(os.path.join(calib_dir, f), dtype=np.float32).reshape(1, 192, 128)
    all_z.append(torch.from_numpy(z).squeeze(0))
calib_tensor = torch.stack(all_z)
dataset = TensorDataset(calib_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
print("  %d samples" % len(calib_tensor))

# 3. Bias Correction
print("\n[3/4] Running Bias Correction...")
from aimet_torch.quantsim import QuantizationSimModel
dummy_input = torch.randn(1, 192, 128)

print("  Creating QuantSim + computing encodings...")

def calibration_cb(model, _):
    for batch in dataloader:
        model(batch[0])
        break

sim = QuantizationSimModel(
    model=wrapper,
    dummy_input=dummy_input,
    default_param_bw=8,
    default_output_bw=8,
)
sim.compute_encodings(calibration_cb, forward_pass_callback_args=None)
print("  Encodings computed")

# Correct biases
print("  Correcting biases...")
t0 = time.time()
try:
    from aimet_torch.bias_correction import module_level_bias_correction, BiasCorrectionParams
    params = BiasCorrectionParams(
        num_quant_samples=len(calib_tensor),
        num_bias_correct_samples=len(calib_tensor),
    )
    module_level_bias_correction(wrapper, sim, dataloader, params)
except (ImportError, AttributeError) as e:
    print(f"  API fallback: {e}")
    from aimet_torch.bias_correction import correct_bias
    correct_bias(wrapper, dummy_input, dataloader, num_quant_samples=min(10, len(calib_tensor)))

print("  Bias Correction done in %.1fs" % (time.time() - t0))

# 4. Export ONNX
print("\n[4/4] Exporting ONNX...")
out_path = '/project/aimet_work/generator_bc.onnx'
torch.onnx.export(
    wrapper,
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

# Quick validate
import onnxruntime as ort
sess_orig = ort.InferenceSession('/project/cosyvoice_snpe_snpe/generator_default.onnx', providers=['CPUExecutionProvider'])
sess_bc = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
z_test = calib_tensor[0:1].numpy()
out_orig = sess_orig.run(None, {'z': z_test})[0].squeeze()
out_bc = sess_bc.run(None, {'z': z_test})[0].squeeze()
diff = np.abs(out_orig - out_bc)
corr = np.corrcoef(out_orig, out_bc)[0, 1]
print("  vs original: diff_max=%.6f, corr=%.6f" % (diff.max(), corr))

print("\nDone: %s" % out_path)
