"""Local test: Compare PyTorch Option B (decoder g=g) vs ONNX Option B (decoder g=None).
Determines if missing speaker conditioning in ONNX/QNN is the root cause of bad audio.
"""
import sys, os, io, time, wave
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import onnxruntime as ort

REPO = r"D:\ai_model\melo_tts_zh_multi\melotts_repo"
sys.path.insert(0, str(REPO))

from melo import utils as melo_utils
from melo import commons
from melo.models import SynthesizerTrn
from melo.split_utils import split_sentence
import re

OUT = r"D:\ai_model\melotts_qnn_dsp\output\local_test"
os.makedirs(OUT, exist_ok=True)

VOICE = "thchs30_a2"
CKPT = r"D:\ai_model\melo_tts_zh_multi\data\aishell3\logs\gpt-medium-thchs30-a2\G_54600.pth"
CONFIG = r"D:\ai_model\melo_tts_zh_multi\data\aishell3\logs\gpt-medium-thchs30-a2\config.json"
ONNX_PATH = r"D:\ai_model\melotts_qnn_dsp\models\generator_option_b.onnx"
SPEAKER_ID = 0
HOP = 512
SR = 44100

TEXT = "你好，欢迎使用本地语音合成系统进行测试。"


def load_model():
    hps = melo_utils.get_hparams_from_file(CONFIG)
    if hasattr(hps.data, "disable_bert"):
        hps.data.disable_bert = True
    else:
        setattr(hps.data, "disable_bert", True)

    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=hps.num_tones,
        num_languages=hps.num_languages,
        **hps.model,
    )
    model.eval()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    return model, hps


def apply_option_b(model):
    """Remove K=11 MRF branches (every 3rd resblock, index 2,5,8,11,14)."""
    dec = model.dec
    num_kernels = dec.num_kernels
    num_upsamples = dec.num_upsamples
    remove_indices = sorted(
        [i * num_kernels + 2 for i in range(num_upsamples)], reverse=True
    )
    for idx in remove_indices:
        del dec.resblocks[idx]
    dec.num_kernels = 2
    print(f"[Option B] Removed {len(remove_indices)} K=11 resblocks")


def encode_text(model, hps, text, speaker_id, speed=1.0):
    """Run MeloTTS encoder: text -> z_latent. Replicates model.infer() up to flow."""
    device = "cpu"
    symbol_to_id = {s: i for i, s in enumerate(hps.symbols)}
    language = "ZH_MIX_EN"

    texts = split_sentence(text, language_str=language)
    all_z = []

    for t in texts:
        if language in ["EN", "ZH_MIX_EN"]:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

        bert, ja_bert, phones, tones, lang_ids = melo_utils.get_text_for_tts_infer(
            t, language, hps, device, symbol_to_id
        )

        with torch.no_grad():
            x = phones.to(device).unsqueeze(0)
            tones_t = tones.to(device).unsqueeze(0)
            lang_ids_t = lang_ids.to(device).unsqueeze(0)
            bert_t = bert.to(device).unsqueeze(0)
            ja_bert_t = ja_bert.to(device).unsqueeze(0)
            x_lengths = torch.LongTensor([phones.size(0)]).to(device)
            sid = torch.LongTensor([speaker_id]).to(device)

            if model.n_speakers > 0:
                g = model.emb_g(sid).unsqueeze(-1)
            else:
                raise RuntimeError("n_speakers=0 not supported")

            g_p = None if model.use_vc else g

            x_enc, m_p, logs_p, x_mask = model.enc_p(
                x, x_lengths, tones_t, lang_ids_t, bert_t, ja_bert_t, g=g_p
            )

            logw = (model.sdp(x_enc, x_mask, g=g, reverse=True, noise_scale=0.8) * 0.2 +
                    model.dp(x_enc, x_mask, g=g) * 0.8)
            w = torch.exp(logw) * x_mask * (1.0 / speed)
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(w_ceil, attn_mask)

            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 0.6
            z = model.flow(z_p, y_mask, g=g, reverse=True)
            z_latent = (z * y_mask)

            all_z.append(z_latent)

    z_cat = torch.cat(all_z, dim=2)
    return z_cat, z_cat.size(2)


def save_wav(audio, path):
    """Save float32 audio as 44100Hz 16-bit WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(samples.tobytes())


# =====================================================
print("=" * 60)
print("Local MeloTTS Option B Audio Quality Test")
print("=" * 60)

# 1. Load model + apply Option B
print("\n[1] Loading model...")
model, hps = load_model()
print(f"  inter_channels={hps.model.inter_channels}, hop={hps.data.hop_length}")
print(f"  n_speakers={hps.data.n_speakers}, use_vc={model.use_vc}")

print("\n[2] Applying Option B...")
apply_option_b(model)

# 2. Encode text -> z_latent
print(f"\n[3] Encoding: '{TEXT}'")
t0 = time.perf_counter()
z_latent, T_enc = encode_text(model, hps, TEXT, SPEAKER_ID)
print(f"  z_latent shape: {z_latent.shape}, T_enc={T_enc}, time={time.perf_counter()-t0:.2f}s")

# 3. PyTorch Option B: decoder with g=g (original conditioning)
print("\n[4] PyTorch Option B decode (decoder g=g, FULL speaker conditioning)...")
with torch.no_grad():
    g = model.emb_g(torch.LongTensor([SPEAKER_ID])).unsqueeze(-1)
    audio_pt = model.dec(z_latent, g=g).squeeze().cpu().numpy()
pt_path = os.path.join(OUT, "pytorch_option_b.wav")
save_wav(audio_pt, pt_path)
print(f"  Saved: {pt_path} ({os.path.getsize(pt_path)/1024:.1f} KB)")

# 4. ONNX Option B: decoder with g=None (what QNN uses)
print("\n[5] ONNX Option B decode (decoder g=None, NO speaker conditioning)...")
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
z_np = z_latent.cpu().numpy().astype(np.float32)

t0 = time.perf_counter()
out = sess.run(["y"], {"z": z_np})
audio_onnx = out[0].squeeze()[:T_enc * HOP]
onnx_path = os.path.join(OUT, "onnx_option_b.wav")
save_wav(audio_onnx, onnx_path)
print(f"  Saved: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

# 5. Compare
print(f"\n{'=' * 60}")
print("RESULTS")
print("=" * 60)
print(f"  PyTorch (g=g):  {pt_path}")
print(f"  ONNX   (g=None): {onnx_path}")
print(f"\n  Diff: max={np.abs(audio_pt[:len(audio_onnx)] - audio_onnx).max():.6f}")
print(f"  Listen to both files — compare quality.")
print(f"\n  If ONNX sounds bad and PyTorch sounds good: bug is g=None in decoder.")
print(f"  If BOTH sound bad: bug is in encoder (z_latent) or Option B itself.")
print(f"  If BOTH sound good: problem is QNN INT8 calibration, not ONNX.")
print("Done")
