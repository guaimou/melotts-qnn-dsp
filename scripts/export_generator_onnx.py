"""Export MeloTTS HiFiGAN Generator and DurationPredictor directly from PyTorch.

This produces clean ONNX graphs with proper shape annotations, matching the
CosyVoice approach (which successfully converted to DLC).
"""

import sys
from pathlib import Path

# Setup paths — must match the training environment
import os
REPO = Path(r"D:\ai_model\melo_tts_zh_multi\melotts_repo")
MELO = REPO / "melo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MELO))
os.chdir(str(REPO))  # melo/text uses relative imports

import torch
import torch.nn as nn
from melo import utils as melo_utils

# Fix Windows encoding for torch.onnx emoji output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUT_DIR = Path(__file__).resolve().parent


def load_checkpoint(model_key):
    """Load SynthesizerTrn from checkpoint."""
    preset = {
        "thchs30_a2": REPO.parent / "data" / "aishell3" / "logs" / "gpt-medium-thchs30-a2",
        "thchs30_a13": REPO.parent / "data" / "aishell3" / "logs" / "gpt-medium-thchs30-a13",
    }[model_key]

    config_path = str(preset / "config.json")
    ckpt_path = str(preset / ("G_54600.pth" if "a2" in model_key else "G_34400.pth"))

    hps = melo_utils.get_hparams_from_file(config_path)
    hps.data.disable_bert = True

    from melo.models import SynthesizerTrn
    model = SynthesizerTrn(
        len(hps.symbols), hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    model.eval()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model, hps


def export_generator(model, hps, out_path):
    """Export the HiFiGAN Generator (model.dec) to ONNX."""
    print("Exporting Generator (model.dec) ...")

    # Trace the generator
    inter_c = hps.model.inter_channels  # 192
    gin_c = hps.model.gin_channels      # 256
    T = 128

    class GeneratorWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, z):
            # Omit speaker conditioning (g=None) to avoid [N,C,1] + [N,C,T] broadcast
            # which breaks SNPE/QNN shape inference. Single-speaker models don't need it.
            return self.dec(z, g=None)

    wrapper = GeneratorWrapper(model.dec)
    wrapper.eval()

    z = torch.randn(1, inter_c, T)

    torch.onnx.export(
        wrapper, (z,), str(out_path),
        input_names=["z"],
        output_names=["y"],
        dynamic_axes={"z": {0: "N", 2: "T"}, "y": {0: "N", 2: "T_audio"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def export_dp(model, hps, out_path):
    """Export the DurationPredictor (model.dp) to ONNX."""
    print("Exporting DurationPredictor (model.dp) ...")

    inter_c = hps.model.inter_channels  # 192
    gin_c = hps.model.gin_channels      # 256
    L = 64

    class DPWrapper(nn.Module):
        def __init__(self, dp):
            super().__init__()
            self.dp = dp

        def forward(self, x, x_mask):
            # Omit speaker conditioning to avoid broadcast issue
            return self.dp(x, x_mask, g=None)

    wrapper = DPWrapper(model.dp)
    wrapper.eval()

    x = torch.randn(1, inter_c, L)
    x_mask = torch.ones(1, 1, L)

    torch.onnx.export(
        wrapper, (x, x_mask), str(out_path),
        input_names=["x", "x_mask"],
        output_names=["log_duration"],
        dynamic_axes={"x": {0: "N", 2: "L"}, "x_mask": {0: "N", 2: "L"},
                      "log_duration": {0: "N", 2: "L"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def verify_onnx(out_path, feeds, label):
    """Verify ONNX model with ORT."""
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    for name, shape in feeds.items():
        feeds[name] = np.random.randn(*shape).astype(np.float32)
    outputs = sess.run(None, feeds)
    for out_vi, val in zip(sess.get_outputs(), outputs):
        print(f"  {label} output {out_vi.name}: shape={val.shape}  mean={val.mean():.4f}")
    print(f"  {label} ORT OK")


def main():
    for voice in ["thchs30_a2"]:  # Use A2; architecture is identical for A13
        print(f"\n{'='*50}")
        print(f"Loading {voice} ...")
        model, hps = load_checkpoint(voice)
        print("Model loaded.")

        gen_path = OUT_DIR / "generator_clean.onnx"
        export_generator(model, hps, gen_path)
        verify_onnx(gen_path, {"z": (1, 192, 128)}, "generator")

        dp_path = OUT_DIR / "dp_clean.onnx"
        export_dp(model, hps, dp_path)
        verify_onnx(dp_path, {"x": (1, 192, 64), "x_mask": (1, 1, 64)}, "dp")


if __name__ == "__main__":
    main()
