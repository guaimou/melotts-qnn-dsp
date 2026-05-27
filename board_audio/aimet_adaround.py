"""AIMET AdaRound optimization for MeloTTS HiFiGAN decoder."""
import sys, os, torch, numpy as np
sys.path.insert(0, '/project/aimet_work')

from melo.models import SynthesizerTrn
from melo import utils as melo_utils
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

# 1. Load model
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

# 2. Extract decoder as standalone
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
print('Wrapper created')

# 3. Load calibration data
calib_dir = '/project/cosyvoice_snpe_snpe/calib_real'
calib_files = sorted([f for f in os.listdir(calib_dir) if f.endswith('.raw')])
print('Found %d calibration files' % len(calib_files))

# Load all calibration z_latent samples
calib_data = []
for f in calib_files[:10]:  # Use 10 samples for AdaRound
    z = np.fromfile(os.path.join(calib_dir, f), dtype=np.float32).reshape(1, 192, 128)
    calib_data.append(torch.from_numpy(z))

print('Loaded %d calibration samples, shape=%s' % (len(calib_data), calib_data[0].shape))

# 4. Create QuantizationSimModel
# Define the quantization scheme - matching QNN DSP INT8
dummy_input = torch.randn(1, 192, 128)

sim = QuantizationSimModel(
    model=wrapper,
    dummy_input=dummy_input,
    default_param_bw=8,
    default_output_bw=8,
    config_file=None,  # Use default config
)
print('QuantSim created')

# 5. Run AdaRound
# AdaRound optimizes weight rounding to minimize quantization error
adaround_params = AdaroundParameters(
    data_loader=calib_data,
    num_batches=len(calib_data),
    default_num_iterations=10000,
    default_reg_param=0.01,
    default_beta_range=(20, 2),
    default_warm_start=0.2,
)

print('Starting AdaRound optimization...')
# Adaround.optimize wraps the full flow
Adaround.optimize(
    sim=sim,
    params=adaround_params,
    path='/project/aimet_work/adaround_output',
    filename_prefix='generator_adaround',
    default_param_bw=8,
    default_quant_scheme=None,
)
print('AdaRound complete')

# 6. Export the optimized model
# Export encoding JSON for QNN converter
sim.export_encodings(
    encoding_file_path='/project/aimet_work/adaround_encodings.json',
    export_dir='/project/aimet_work/',
    filename_prefix='generator_adaround'
)
print('Encodings exported')

# Also export ONNX with Q/DQ nodes
torch.onnx.export(
    wrapper,
    (dummy_input,),
    '/project/aimet_work/generator_adaround.onnx',
    input_names=['z'],
    output_names=['y'],
    dynamic_axes={'z': {0: 'N', 2: 'T'}, 'y': {0: 'N', 2: 'T_audio'}},
    opset_version=17,
    do_constant_folding=True,
)
print('ONNX exported')

print('Done - AdaRound optimization complete')
