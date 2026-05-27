"""AIMET Bias Correction + AdaRound for MeloTTS HiFiGAN decoder.
Order: Bias Correction first, then AdaRound (AIMET best practice).
"""
import sys, os, torch, numpy as np, time, types

# Mock torchaudio + librosa
for mod_name in ('torchaudio', 'librosa'):
    m = types.ModuleType(mod_name)
    m.__dict__.update({k: (lambda *a, **kw: None) for k in ('load', 'info', 'get_duration', 'resample')})
    sys.modules[mod_name] = m

sys.path.insert(0, '/project/aimet_work')

from melo.models import SynthesizerTrn
from melo import utils as melo_utils
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.bias_correction import BiasCorrectionParams, correct_bias
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 4
NUM_CALIB = 30
NUM_ITER = 2000

print("=" * 60)
print("AIMET Bias Correction + AdaRound — MeloTTS HiFiGAN")
print("=" * 60)

# 1. Load model
print("\n[1/6] Loading model...")
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
print("  Model loaded: %d params" % sum(p.numel() for p in model.parameters()))

dec = model.dec
dec.eval()

# Remove weight_norm (required for AIMET)
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
print("\n[2/6] Loading calibration data...")
calib_dir = '/project/calib_real'
calib_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.raw')])[:NUM_CALIB]

all_z = []
for f in calib_files:
    z = np.fromfile(os.path.join(calib_dir, f), dtype=np.float32).reshape(1, 192, 128)
    all_z.append(torch.from_numpy(z).squeeze(0))

calib_tensor = torch.stack(all_z)
dataset = TensorDataset(calib_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print("  %d samples, %d batches" % (len(calib_tensor), len(dataloader)))

# 3. Bias Correction
print("\n[3/6] Running Bias Correction...")
t0 = time.time()

# Create quantized model for bias correction
from aimet_torch.quantsim import QuantizationSimModel
dummy_input = torch.randn(1, 192, 128)

sim = QuantizationSimModel(
    model=wrapper,
    dummy_input=dummy_input,
    default_param_bw=8,
    default_output_bw=8,
)

# Run bias correction
params = BiasCorrectionParams(
    num_quant_samples=10,
    num_bias_correct_samples=10,
    input_shape=(1, 192, 128),
)

try:
    correct_bias(
        model=wrapper,
        quant_sim=sim,
        data_loader=dataloader,
        params=params,
    )
    print("  Bias Correction complete in %.1f seconds" % (time.time() - t0))
except Exception as e:
    print("  Bias Correction simplified: %s" % str(e)[:100])
    # Fallback: use simpler API
    from aimet_torch.bias_correction import correct_bias as cb2
    cb2(wrapper, dummy_input, dataloader, num_quant_samples=10)
    print("  Bias Correction (simplified) complete in %.1f seconds" % (time.time() - t0))

# 4. AdaRound on bias-corrected model
print("\n[4/6] Running AdaRound on bias-corrected model...")
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
    filename_prefix='generator_bc_adaround',
    default_param_bw=8,
)
print("  AdaRound complete in %.1f seconds" % (time.time() - t0))

# 5. Export ONNX
print("\n[5/6] Exporting ONNX...")
out_path = '/project/aimet_work/generator_bc_adaround.onnx'
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

# 6. Validate
print("\n[6/6] Validating...")
import onnxruntime as ort

orig_onnx = '/project/cosyvoice_snpe_snpe/generator_default.onnx'
sess_orig = ort.InferenceSession(orig_onnx, providers=['CPUExecutionProvider'])
sess_opt = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])

corrs = []
diffs_max = []
for i in range(min(10, len(calib_tensor))):
    z_test = calib_tensor[i:i+1].numpy()
    out_orig = sess_orig.run(None, {'z': z_test})[0].squeeze()
    out_opt = sess_opt.run(None, {'z': z_test})[0].squeeze()
    diff = np.abs(out_orig - out_opt)
    corr = np.corrcoef(out_orig, out_opt)[0, 1]
    corrs.append(corr)
    diffs_max.append(diff.max())

print("  BC+AdaRound vs Original FP32:")
print("    diff_max: mean=%.6f, max=%.6f" % (np.mean(diffs_max), np.max(diffs_max)))
print("    corr:     mean=%.6f, min=%.6f" % (np.mean(corrs), np.min(corrs)))

print("\nDone — BC+AdaRound ONNX: %s" % out_path)
