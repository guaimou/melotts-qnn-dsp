"""Export optimized HiFiGAN Generator (Option B: remove K=11 MRF branch, 2x speedup)."""

import sys
import os
from pathlib import Path
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO = Path(r"D:\ai_model\melo_tts_zh_multi\melotts_repo")
MELO = REPO / "melo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MELO))
os.chdir(str(REPO))

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from melo import utils as melo_utils

OUT_DIR = Path(__file__).resolve().parent


def load_model(voice="thchs30_a2"):
    preset = {
        "thchs30_a2": REPO.parent / "data" / "aishell3" / "logs" / "gpt-medium-thchs30-a2",
        "thchs30_a13": REPO.parent / "data" / "aishell3" / "logs" / "gpt-medium-thchs30-a13",
    }[voice]
    config_path = str(preset / "config.json")
    ckpt = "G_54600.pth" if "a2" in voice else "G_34400.pth"
    ckpt_path = str(preset / ckpt)

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


def apply_option_b(model):
    """Remove K=11 MRF branches (every 3rd resblock, index 2,5,8,11,14).
    Reduces MACs by ~50% with minimal quality impact."""
    dec = model.dec

    num_kernels = dec.num_kernels  # 3
    num_upsamples = dec.num_upsamples  # 5

    # Indices to remove: j=2 in each group of 3
    remove_indices = sorted(
        [i * num_kernels + 2 for i in range(num_upsamples)], reverse=True
    )

    for idx in remove_indices:
        del dec.resblocks[idx]

    dec.num_kernels = 2  # Now only K=3 and K=7
    print(f"  Removed {len(remove_indices)} K=11 resblocks, num_kernels={dec.num_kernels}")


def export_generator(model, hps, name_suffix, out_path):
    """Export generator to ONNX (no speaker conditioning)."""
    dec = model.dec

    class GeneratorWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, z):
            return self.dec(z, g=None)

    wrapper = GeneratorWrapper(dec)
    wrapper.eval()

    # Remove weight_norm for ONNX export
    wrapper.dec.remove_weight_norm()

    T = 128
    inter_c = hps.model.inter_channels  # 192
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

    # Verify
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {"z": np.random.randn(1, 192, 128).astype(np.float32)}
    out = sess.run(None, feed)
    params = sum(p.numel() for p in wrapper.dec.parameters())
    size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"  {name_suffix}: params={params:,}, size={size_mb:.1f}MB, "
          f"output shape={out[0].shape}")


def main():
    # --- Original (baseline) ---
    print("Loading A2 checkpoint for baseline...")
    model, hps = load_model("thchs30_a2")
    full_params = sum(p.numel() for p in model.dec.parameters())
    print(f"  Original Generator params: {full_params:,}")
    print("\nExporting original generator (baseline)...")
    export_generator(model, hps, "original", OUT_DIR / "generator_original.onnx")

    # --- Option B: remove K=11 (fresh model) ---
    print("\nLoading A2 checkpoint for Option B...")
    model, hps = load_model("thchs30_a2")
    print("Applying Option B: remove K=11 branches...")
    apply_option_b(model)
    b_params = sum(p.numel() for p in model.dec.parameters())
    print(f"  Optimized Generator params: {b_params:,} "
          f"({100 * (1 - b_params / full_params):.1f}% reduction)")
    print("\nExporting optimized generator (Option B)...")
    export_generator(model, hps, "option_b", OUT_DIR / "generator_option_b.onnx")

    print(f"\n{'='*50}")
    print(f"Done. Files in {OUT_DIR}:")
    for f in sorted(OUT_DIR.glob("generator_*.onnx")):
        print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
