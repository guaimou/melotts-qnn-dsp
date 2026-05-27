"""Export HiFiGAN decoder as 2D conv ONNX for QNN 2.29 compatibility."""
import sys, torch, numpy as np, types, os

for mn in ('torchaudio', 'librosa'):
    m = types.ModuleType(mn)
    setattr(m, 'load', lambda *a,**kw: (None, None))
    sys.modules[mn] = m

sys.path.insert(0, '/project/aimet_work')
from melo.models import SynthesizerTrn
from melo import utils as melo_utils
from torch.nn.utils import remove_weight_norm

# Load model
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
print("Model loaded")

dec = model.dec
dec.eval()

# Remove weight_norm
if hasattr(dec, 'remove_weight_norm'):
    dec.remove_weight_norm()
for _, m in dec.named_modules():
    try: remove_weight_norm(m)
    except ValueError: pass
print("weight_norm removed")

# Test: does decoder accept 4D input?
dummy_4d = torch.randn(1, 192, 1, 128)
try:
    with torch.no_grad():
        y = dec(dummy_4d, g=None)
    print(f"4D input works! Output shape: {y.shape}")
    four_d_works = True
except Exception as e:
    print(f"4D fails: {e}")
    four_d_works = False

if four_d_works:
    # Export with 4D input
    class Wrapper(torch.nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec
        def forward(self, z):
            # z: [N, 192, T] -> reshape to [N, 192, 1, T]
            z = z.unsqueeze(2)
            y = self.dec(z, g=None)
            # y: [N, 1, 1, T_audio] -> [N, 1, T_audio]
            return y.squeeze(2)

    wrapper = Wrapper(dec)
    wrapper.eval()

    out_path = '/project/aimet_work/generator_cle_4d.onnx'
    torch.onnx.export(
        wrapper,
        (dummy_4d.squeeze(2),),  # 3D input for the wrapper
        out_path,
        input_names=['z'],
        output_names=['y'],
        dynamic_axes={'z': {0: 'N', 2: 'T'}, 'y': {0: 'N', 2: 'T_audio'}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported: {out_path} ({os.path.getsize(out_path)/1024/1024:.1f}MB)")

    # Verify
    import onnxruntime as ort
    z_test = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

    # Original 1D model
    s1 = ort.InferenceSession('/project/aimet_work/generator_cle.onnx', providers=['CPUExecutionProvider'])
    y1 = s1.run(None, {'z': z_test})[0].squeeze()

    # New 4D model (with Squeeze/Unsqueeze)
    s2 = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    y2 = s2.run(None, {'z': z_test})[0].squeeze()

    diff = np.abs(y1 - y2).max()
    corr = np.corrcoef(y1, y2)[0, 1]
    print(f"Verify: diff_max={diff:.10f}, corr={corr:.10f}")
else:
    print("4D input not supported by decoder directly")
    print("Need to modify decoder internal convs from Conv1d to Conv2d")
