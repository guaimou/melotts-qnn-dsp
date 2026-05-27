"""AIMET AdaRound optimization for MeloTTS HiFiGAN decoder - v2 (correct API)."""
import sys, os, torch, numpy as np
sys.path.insert(0, '/project/aimet_work')

from melo.models import SynthesizerTrn
from melo import utils as melo_utils
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.onnx_utils import OnnxExportApiArgs
from torch.utils.data import DataLoader, TensorDataset

# 1. Load model and extract decoder
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
print('Model loaded')

dec = model.dec
dec.eval()

class GeneratorWrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
    def forward(self, z):
        return self.dec(z, g=None)

wrapper = GeneratorWrapper(dec)
wrapper.eval()
print('Wrapper ready')

# 2. Load calibration data as DataLoader
calib_dir = '/project/cosyvoice_snpe_snpe/calib_real'
calib_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.raw')])
print('Found %d calibration files' % len(calib_files))

# Load z_latent samples into tensors
all_z = []
for f in calib_files[:20]:
    z = np.fromfile(os.path.join(calib_dir, f), dtype=np.float32).reshape(1, 192, 128)
    all_z.append(torch.from_numpy(z).squeeze(0))  # [192, 128]

calib_tensor = torch.stack(all_z)  # [N, 192, 128]
dataset = TensorDataset(calib_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
print('DataLoader ready: %d batches of size 4' % len(dataloader))

# 3. Run AdaRound (post-training weight optimization)
dummy_input = torch.randn(1, 192, 128)

print('Starting AdaRound...')
adaround_params = AdaroundParameters(
    data_loader=dataloader,
    num_batches=len(dataloader),
    default_num_iterations=1000,
)

optimized_model = Adaround.apply_adaround(
    model=wrapper,
    dummy_input=dummy_input,
    params=adaround_params,
    path='/project/aimet_work',
    filename_prefix='generator_adaround',
    default_param_bw=8,
)
print('AdaRound complete')

# 4. Export optimized ONNX
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
print('ONNX exported: %s (%.1fMB)' % (out_path, os.path.getsize(out_path)/1024/1024))

# 5. Compare FP32 output vs original for validation
import onnxruntime as ort
sess_orig = ort.InferenceSession('/project/cosyvoice_snpe_snpe/generator_default.onnx', providers=['CPUExecutionProvider'])
sess_opt = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])

z_test = np.random.randn(1, 192, 128).astype(np.float32)
out_orig = sess_orig.run(None, {'z': z_test})[0].squeeze()
out_opt = sess_opt.run(None, {'z': z_test})[0].squeeze()

diff = np.abs(out_orig - out_opt)
corr = np.corrcoef(out_orig, out_opt)[0, 1]
print('AdaRound vs Original: diff_max=%.6f, corr=%.6f' % (diff.max(), corr))

print('Done')
