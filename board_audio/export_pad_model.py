"""Export ONNX with explicit Pad nodes, Conv padding=0 - may bypass QNN bug."""
import sys, torch, numpy as np, types, os, copy

for mn in ('torchaudio', 'librosa'):
    m = types.ModuleType(mn)
    setattr(m, 'load', lambda *a,**kw: (None, None))
    sys.modules[mn] = m

sys.path.insert(0, '/project/aimet_work')
from melo.models import SynthesizerTrn
from melo import utils as melo_utils
from torch.nn.utils import remove_weight_norm
import torch.nn.functional as F

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

dec = model.dec
dec.eval()

# Remove weight_norm
if hasattr(dec, 'remove_weight_norm'):
    dec.remove_weight_norm()
for _, m in dec.named_modules():
    try: remove_weight_norm(m)
    except ValueError: pass

# Create wrapper that adds explicit padding before Conv1d/ConvTranspose1d
class PadWrapper(torch.nn.Module):
    """Wrapper that does manual padding + Conv with padding=0."""
    def __init__(self, dec):
        super().__init__()
        # Clone the decoder but replace all Conv1d with padding=0 versions
        # We need to hook into the forward pass
        self.dec = dec

    def forward(self, z):
        return self.dec(z, g=None)

# Actually we need to modify the decoder's internal convs.
# Let's patch all Conv1d/ConvTranspose1d to use padding='same' (manual):
# Store original padding
conv_pads = {}
for name, module in dec.named_modules():
    if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        conv_pads[name] = module.padding
        # Set Conv to padding=0 - padding will be done manually
        module.padding = (0,)

# Now wrap forward to add manual padding before each conv
# This is complex. Let's use a simpler approach:
# Register a forward hook that pads the input

hooked = {}
def make_pad_hook(pad_size):
    def pad_hook(module, input):
        if isinstance(pad_size, (list, tuple)):
            p = pad_size[0]
        else:
            p = pad_size
        if p > 0 and len(input[0].shape) == 3:
            return (F.pad(input[0], (p, p), mode='reflect'),)
        return input
    return pad_hook

for name, module in dec.named_modules():
    if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        pad = conv_pads.get(name, 0)
        if isinstance(pad, (list, tuple)):
            p = pad[0]
        else:
            p = pad
        if p > 0:
            module.register_forward_pre_hook(make_pad_hook(p))
            hooked[name] = p

print(f'Hooked {len(hooked)} convs with manual padding')

class Wrapper(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
    def forward(self, z):
        return self.dec(z, g=None)

wrapper = Wrapper(dec)
wrapper.eval()

# Export
dummy = torch.randn(1, 192, 128)
out_path = '/project/aimet_work/generator_cle_pad.onnx'
torch.onnx.export(
    wrapper, (dummy,), out_path,
    input_names=['z'], output_names=['y'],
    opset_version=17,
    do_constant_folding=True,
)
print(f'Exported: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f}MB)')

# Verify
import onnxruntime as ort
s_ref = ort.InferenceSession('/project/aimet_work/generator_cle.onnx', providers=['CPUExecutionProvider'])
s_pad = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
z = np.random.randn(1, 192, 128).astype(np.float32)*0.5
y_ref = s_ref.run(None, {'z': z})[0].squeeze()
y_pad = s_pad.run(None, {'z': z})[0].squeeze()
diff = np.abs(y_ref - y_pad).max()
corr = np.corrcoef(y_ref, y_pad)[0,1]
print(f'Verify: diff={diff:.10f}, corr={corr:.10f}')

# Check Pad nodes in new model
import onnx
m2 = onnx.load(out_path)
pad_count = sum(1 for n in m2.graph.node if n.op_type == 'Pad')
print(f'Pad nodes in exported model: {pad_count}')
