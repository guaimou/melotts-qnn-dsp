# MeloTTS → QNN DSP Acceleration Project

Convert MeloTTS VITS HiFiGAN Generator to QNN DSP format, replicating the official FiboTTS approach on SC171v3 (QCS6490, Hexagon v68).

## Goal

```
Text → [CPU: ONNX Encoder+DP] → z_latent → [DSP: QNN INT8 Generator] → Audio
```

## Status: Per-Channel INT8 Deployed, AdaRound In Progress (2026-05-22)

| Component | Status | Details |
|-----------|--------|---------|
| ONNX export (PyTorch) | Done | Clean export, no speaker conditioning |
| ONNX→QNN (2.26.2 + INT8) | Done | `qnn-onnx-converter` with calibration |
| .so build (aarch64) | Done | Native g++ 9.4 on board |
| QNN DSP INT8 inference | **Done** | **RTF 0.136 (202ms)** |
| Per-channel quantization | **Deployed** | 3 voices, per-channel SQNR calib |
| AdaRound+Bias Correction | **In Progress** | Scripts ready, Docker set up, NOT run |
| Full pipeline projection | Done | **1.6-5.8x faster** than ONNX CPU |
| Audio quality samples | Done | In `output/tts_samples/`, `board_audio/` |

### Quantization variants timeline

| Version | File | Method | Quality | Deployed |
|---------|------|--------|---------|----------|
| v1 sym | gen_default_sym.bin | Symmetric INT8 | Baseline | No |
| v2 int8 | gen_default_int8.bin | Asymmetric INT8, 20 random calib | Better | Was v1 |
| v3 p99 | gen_default_p99.bin | 100 calib, p99 clipping | Good | No |
| v4 100 | gen_default_100.bin | 100 real z_latent calib | Good | No |
| v5 sqnr | gen_default_sqnr.bin | SQNR-based quantization | Better | No |
| v6 entropy | gen_default_entropy.bin | Entropy calibration | Good | No |
| **v7 pch** | **gen_default_pch.bin** | **Per-channel SQNR** | **Best so far** | **Yes** |
| v8 aimet | gen_default_aimet.bin | QNN "aimet" encoding preset | Good | No |
| v9 prune | — | Weight pruning (Option B) | **Failed** — severe quality loss | No |
| **v10 adaround** | **gen_adaround_int8.bin** | **AdaRound opt + INT8** | **Good — DSP 274ms** | **Yes (testing)** |
| **v11 cle** | **gen_cle_sqnr.bin** | **CLE + SQNR** | **Good — Quiet RMS 156** | **Testing** |
| **v12 dual** | **gen_cle_sqnr.bin** | **CLE + SQNR + Dual-pass** | **Quiet RMS -24.5%** | **Testing** |
| **v13 noise_gate** | — | **CLE + SQNR + Noise Gate** | **Best — Quiet RMS -77.5%** | **Testing** |
| **v14 w8a16** | **gen_cle_w8a16.bin** | **CLE + W8A16** | **Best — Quiet RMS -89.8% vs INT8** | **Deployed (default)** |
| **Next** | **a2/a13 w8a16** | **Per-voice W8A16** | **TBD** | **TODO** |

## Performance (board, Option B generator)

| Backend | Quant | RTF | ms/chunk | Speed vs ONNX |
|---------|-------|-----|----------|---------------|
| **QNN DSP** | **INT8** | **0.136** | **202ms** | **3-6x** |
| v68 | QNN DSP | FP32 | — | N/A | **Failed** — "SocModel doesn't support FP16" |
| v68 | QNN DSP | INT16 | — | N/A | **Failed** — v68 INT8 only |
| **v68** | **QNN DSP** | **W8A16** | **0.417** | **619ms** | **Best quality, -89.8% noise** |
| **v68** | **QNN DSP** | **INT8** | **0.136** | **274ms** | **3-6x faster than CPU FP32** |
| v68 | SNPE DSP | INT16 | 1.92 | 2859ms | 0.5x (slower!) |
| CPU | ONNX | FP32 | 0.55 | 820ms | 1.0x baseline |
| v68 | Official FiboTTS | INT16 | 0.38 | 571ms | — |

### v68 DSP capability summary

Hexagon v68 (QCS6490) DSP only supports INT8 ops in QNN backend. FP32 and INT16 both fail at op validation. All high-precision approaches must work within INT8 constraint.

### Full pipeline projection (ONNX encoder + QNN DSP vocoder)

| Voice | ONNX CPU | QNN DSP est | Speedup |
|-------|---------|-------------|---------|
| default (198MB) | 1.5-3.0s | 0.9-1.8s | 1.6-1.9x |
| A2 (623MB) | 2.4-4.2s | 0.5-1.1s | 3.8-5.8x |
| A13 (623MB) | 3.4-4.5s | 0.9-1.6s | 2.1-3.7x |

## Key technical decisions

### Why SDK 2.26.2
SDK 2.29 introduced a shape inference regression (`expand_1d_spatial_nn_nodes` crash). 2.26.2 is the last version that handles VITS HiFiGAN 1D Conv correctly. 2.32.6's `qairt-converter` also works but produces .bin (QAR) not proper .cpp.

### Why INT8 not INT16
Hexagon v68 (QCS6490) supports INT8 ops but not INT16. The official FiboTTS .fmodel is labeled `qnn_2.26_dsp` but likely uses INT8 internally.

### Why QNN DSP loads .so directly
fiboaisdk `InferAPI` with framework="QNN" accepts `.so` files built by `qnn-model-lib-generator`. No .fmodel packaging needed. Context binary is not required — the .so contains the embedded model.

## SDK version compatibility

| Feature | SDK 2.26.2 | SDK 2.29 | SDK 2.32.6 |
|---------|-----------|----------|------------|
| qnn-onnx-converter (1D Conv) | Works | Crash | Crash |
| qairt-converter | Works | N/A | Works |
| INT8 quantization | Works | N/A | Via qairt-quantizer |
| SoC models (DSP ctx) | SA8295, SA8255 | SA8295, SA8255 | SA8295, SA8255 |
| qnn-model-lib-generator | Works | Works | Broken (.cpp=ZIP) |

## Full conversion workflow

```bash
# 1. Export ONNX from PyTorch
venv_local/Scripts/python.exe scripts/export_optimized_generator.py

# 2. Convert ONNX → QNN INT8 .cpp + .bin (in Docker)
docker exec my_work bash /project/cosyvoice_snpe_snpe/conv_quant.sh

# 3. Build .so on board
adb -s 28de40d2 shell "cd /home/fibo/melotts_qnn && \
  QNN_AARCH64_UBUNTU_GCC_94='' python3 qnn_sdk_262/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c gen262_int8.cpp -b gen262_int8.bin -t aarch64-ubuntu-gcc9.4 -l melotts_int8 -o ./lib_int8"

# 4. Test via fiboaisdk
python3 -c "
from fiboaisdk.api_aisdk_py import api_infer_py
params = api_infer_py.InferParams('lib_int8/.../libmelotts_int8.so', 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
api.Init(params)
api.Execute_float({'z': your_z_latent.tolist()})
"
```

## Project structure

```
melotts_qnn_dsp/
├── CLAUDE.md / README.md
├── models/                     # ONNX source models
├── scripts/                    # Conversion + test scripts
├── board_audio/                # TTS samples + AIMET scripts + board fix scripts
│   ├── aimet_adaround.py       # AdaRound v1 (Adaround.optimize API)
│   ├── aimet_adaround_v2.py    # AdaRound v2 (Adaround.apply_adaround — correct)
│   ├── gen_calib.py            # Generate 100 calibration z_latent .raw
│   ├── gen_calib_voices.py     # Per-voice calibration generation
│   └── final_*.wav / v*_*.wav  # Audio quality samples from each version
├── output/
│   ├── tts_samples/            # Synthesized audio samples
│   ├── generator_*.dlc         # DLC format (SNPE)
│   ├── gen262_*.cpp/.bin       # QNN format (SDK 2.26.2)
│   └── gen262_*int8*           # INT8 quantized QNN
├── gen_default_*.bin/.cpp      # Quantization variants in root (v1-v8)
├── qairt_2.32.6/               # SDK 2.32.6 (Docker copy)
├── qnn_tools/ / qnn_python/    # SDK 2.29 (Docker copy)
└── fiboaisdk-2.0.20-py3-none-any.whl  # Board SDK
```

## Board environment

- Device: SC171v3, QCS6490, Hexagon v68 DSP
- OS: Ubuntu 20.04 aarch64, Python 3.8.10
- fiboaisdk 2.0.20 at `/usr/local/lib/python3.8/dist-packages/fiboaisdk/`
- QNN SDK 2.26.2 at `/home/fibo/qnn_sdk_262/`
- ONNX models + .so at `/home/fibo/melotts_qnn/`
- Access: `adb -s 28de40d2 root`

## QNN TTS deployment (2026-05-21, updated 2026-05-22)

Replaced board `/home/fibo/AI model/tts_models/tts.py` — local DSP backend switched from FiboTTS `.fmodel` (AudioAPI) to QNN pipeline:

```
Text → [PyTorch Encoder CPU] → z_latent → [QNN DSP INT8 Generator .so] → Audio (44100Hz WAV)
```

| Component | Old | New |
|-----------|-----|-----|
| Encoder | Built into `.fmodel` | melo PyTorch checkpoint (CPU) |
| Vocoder | `AudioAPI.SpeechSynthesisSync` | `InferAPI.Execute_float` + per-channel .so |
| Voices | 1 fixed | default / a2 / a13 (3 voices, per-voice .so) |

### Current deployed .so (per-channel, 2026-05-22)

| Voice | .so path | Calibration |
|-------|----------|-------------|
| default | `lib_pch/aarch64-ubuntu-gcc9.4/libmelotts_pch.so` | Per-channel SQNR |
| a2 | `lib_a2_pch/aarch64-ubuntu-gcc9.4/libmelotts_a2_pch.so` | Per-channel SQNR |
| a13 | `lib_a13_pch/aarch64-ubuntu-gcc9.4/libmelotts_a13_pch.so` | Per-channel SQNR |

Backup: `tts(5.20).py`. Key files on board:
- TTS module: `/home/fibo/AI model/tts_models/tts.py`
- QNN .so (default): `/home/fibo/melotts_qnn/lib_pch/aarch64-ubuntu-gcc9.4/libmelotts_pch.so`
- QNN .so (old INT8): `/home/fibo/melotts_qnn/lib_int8/aarch64-ubuntu-gcc9.4/libmelotts_int8.so`
- Melo models: `/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/models/`
- Melo runtime: `/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/runtime/`

## AIMET AdaRound + Bias Correction (2026-05-22)

Goal: eliminate residual background noise in INT8 output. Current per-channel quant helps but AdaRound (weight rounding optimization) + Bias Correction should further reduce quantization error.

### Progress

| Step | Status | Detail |
|------|--------|--------|
| aimet-torch 2.31.0 in Docker | ✅ Done | `pip install aimet-torch==2.31.0` in my_work |
| Calibration data | ✅ Done | 30 z_latent .raw in `calib_real/`, 100+ in `calib_100/` (local) |
| AdaRound script v1 | ✅ Written | `aimet_adaround.py` — uses `Adaround.optimize` (wrong API) |
| AdaRound script v2 | ✅ Written | `aimet_adaround_v2.py` — uses `Adaround.apply_adaround` (correct) |
| AdaRound script v3 | ✅ Written | `aimet_adaround_v3.py` — 30 samples, 2000 iters, full validation |
| Model + melo in Docker | ✅ Done | G_default.pth (198MB), config.json, melo/ copied 2026-05-22 |
| Copy files to Docker | ✅ Done | calib_real/ 30 .raw + v3 script all in container |
| Run AdaRound in Docker | ✅ Done | v3: 5.8s, 54.9MB ONNX, corr=1.0 vs original FP32 |
| Export AIMET encodings | ❌ TODO | AdaRound optimized weights already in ONNX — direct QNN convert |
| Bias Correction | ❌ TODO | Apply after AdaRound (AIMET `correct_bias`) |
| Convert → QNN .so | 🔄 Next | qnn-onnx-converter with per-channel SQNR on AdaRound ONNX |

### Docker environment (2026-05-22 verified)

```
Container: my_work (fiboaistack_229_env:latest)
aimet-torch: 2.31.0 (pip installed), torchaudio: 2.11.0 (pip installed)
Python: system python3 (3.10.12)
Project: /project/aimet_work/
  ├── G_default.pth (198MB), config.json
  ├── aimet_adaround.py, aimet_adaround_v2.py, aimet_adaround_v3.py
  ├── melo/ (runtime .py from board)
  └── generator_adaround.onnx (54.9MB, output)
Calib data: /project/calib_real/ (30 z_*.raw), /project/cosyvoice_snpe_snpe/calib_real/ (30)
ONNX models: /project/cosyvoice_snpe_snpe/generator_default.onnx (correct baseline, corr=1.0)
Mounts: NONE — all files must use docker cp
```

### Source file locations

| File | Location | Size |
|------|----------|------|
| G_default.pth | Board: `/home/fibo/AI model/.../models/default_zh/G_default.pth` | 198MB |
| config.json | Board: same dir as G_default.pth | 2KB |
| melo/ runtime | Board: `/home/fibo/AI model/.../runtime/melo/` | ~30 .py files |
| calib_real/ | Local: `D:\ai_model\melotts_qnn_dsp\calib_real\` | 30 z_*.raw |
| calib_100/ | Local: `D:\ai_model\melotts_qnn_dsp\calib_100\` | 100 z_*.raw |
| aimet_adaround_v3.py | Local: `board_audio/aimet_adaround_v3.py` | Best script |

### AdaRound results (2026-05-22)

```
AdaRound v3: 5.8 seconds, 54.9MB ONNX, corr=1.0 vs generator_default.onnx
QNN INT8 conversion: 78 seconds, gen_adaround_int8.bin (14MB) + .cpp (2.6MB)
.so build: 27 seconds on board, libmelotts_adaround.so (15MB, ELF aarch64)
DSP inference: 274ms/chunk (T=128) — matches PCH baseline (277ms)
```

### Deployed on board (2026-05-22)

| Voice | .so path | QNN params | Status |
|-------|----------|------------|--------|
| **default_ada** | `lib_adaround/aarch64-ubuntu-gcc9.4/libmelotts_adaround.so` | in_scale=0.06378, in_off=-133, out_scale=0.001675, out_off=-115 | **Deployed, working** |
| default (pch) | `lib_pch/.../libmelotts_pch.so` | in_scale=0.06356, in_off=-133, out_scale=0.001691, out_off=-117 | Deployed |
| a2 (pch) | `lib_a2_pch/.../libmelotts_a2_pch.so` | — | Deployed |
| a13 (pch) | `lib_a13_pch/.../libmelotts_a13_pch.so` | — | Deployed |

Key findings:
- `generator_option_b.onnx` is a DIFFERENT model variant — use `generator_default.onnx` as baseline
- AdaRound FP32 output is identical to original (corr=1.0) — benefits appear during INT8 quantization
- QNN quantization params are nearly identical to default (0.3-0.9% diff) — same encoder, same calibration
- DSP API init is slow (~81s first load) but one-time cost; subsequent calls use cached API
- DSP inference speed matches PCH baseline (274ms vs 277ms per chunk)
- Required: mock `torchaudio` before melo import (system Python lacks torchaudio)
- Required: `remove_weight_norm()` before AdaRound (AIMET deepcopy fails otherwise)
- Required: copy `cn2an`, `jieba`, `pypinyin` to conda env (or use system Python + torchaudio mock)

```bash
# Step 1: Copy model from board to Windows temp
adb -s 28de40d2 pull "/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/models/default_zh/G_default.pth" .
adb -s 28de40d2 pull "/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/models/default_zh/config.json" .

# Step 2: Copy melo runtime from board
adb -s 28de40d2 pull "/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/runtime/melo/" ./melo/

# Step 3: docker cp everything into container
docker cp G_default.pth my_work:/project/aimet_work/
docker cp config.json my_work:/project/aimet_work/
docker cp melo/ my_work:/project/aimet_work/melo/
docker cp calib_real/ my_work:/project/calib_real/
docker cp board_audio/aimet_adaround_v3.py my_work:/project/aimet_work/

# Step 4: Run AdaRound v3 in Docker
docker exec my_work python3 /project/aimet_work/aimet_adaround_v3.py

# Output: /project/aimet_work/generator_adaround.onnx
```

### Known issues to resolve

1. ~~**v1/v2 scripts superseded**~~: Use v3 — 30 samples, 2000 iters, validates against `generator_default.onnx`
2. ~~**Baseline mismatch**~~: `generator_option_b.onnx` != G_default.pth — use `generator_default.onnx`
3. ~~**weight_norm deepcopy**~~: Remove before AdaRound with `dec.remove_weight_norm()` + recursive cleanup
4. ~~**torchaudio/librosa missing**~~: Mock via `sys.modules` before melo import
5. ~~**QNN conversion**~~: Done — 14MB INT8 .bin, DSP inference 274ms/chunk
6. ~~**Audio quality**~~: Multiple variants tested; CLE+SQNR shows lowest noise floor (see comparison below)
7. **DSP API init slow**: 81s first load for each new .so; root cause TBD (DSP model compilation?)
8. **Only default voice**: Need per-voice CLE/AdaRound (a2, a13) like per-channel pch
9. ~~**Bias Correction**~~: NOT available in aimet-torch 2.31.0 — CLE used instead

### Audio quality comparison (2026-05-23)

| Metric | CLE+SQNR | AdaRound SQNR | PCH baseline |
|--------|----------|---------------|-------------|
| Quiet RMS (noise floor) | **156** | 196 | 160 |
| Dynamic range | 30.3 dB | 27.4 dB | 29.3 dB |
| vs PCH improvement | **-2.5%** | +22.5% (worse) | — baseline |

CLE+SQNR produces the lowest noise floor. Run-to-run variation exists due to encoder non-determinism (jieba tokenization). SQNR calibration is critical — min-max INT8 is significantly worse.

### Dual-pass residual INT8 (2026-05-23)

Technique: run DSP INT8 inference twice with 0.5 LSB offset, average outputs.

```
z_latent → INT8 encode (offset)     → DSP → audio_A
z_latent → INT8 encode (offset+0.5) → DSP → audio_B
final = (audio_A + audio_B) / 2
```

The 0.5 LSB shift creates different quantization error patterns. Averaging reduces noise floor.

| Metric | Single-pass | Dual-pass | Improvement |
|--------|-------------|-----------|-------------|
| Quiet RMS (normalized) | 0.0075 | **0.0057** | **-24.5%** |
| DSP time per chunk | ~300ms | ~671ms | 2.2x slower |
| Effective RTF | ~0.11 | ~0.24 | Still << 1 |
| Pass A/B correlation | — | 0.74 | Partial decorrelation |

Dual-pass effectively trades 2x DSP time for ~25% noise reduction. RTF 0.24 is still well within real-time. No model changes needed — pure inference technique.

### Expected improvement

Dual-pass effectively trades 2x DSP time for ~25% noise reduction. RTF 0.24 is still well within real-time. No model changes needed — pure inference technique.

### 方案3: 输出端噪声整形 (2026-05-23)

Post-processing: soft noise gate or spectral subtraction applied to DSP INT8 output. No model changes, negligible CPU cost (~25ms for 3s audio).

| Metric | Base INT8 | Noise Gate | Spectral Sub |
|--------|-----------|------------|-------------|
| Quiet RMS (int16) | 200 | **45** | **32** |
| Total RMS | 5143 | 5139 | 5438 |
| Improvement | — | **-77.5%** | **-84%** |
| Processing time | — | 27ms | 23ms |
| Artifacts risk | — | Low (only quiet segments) | Medium (oversubtraction) |

Soft noise gate recommended for production: dramatic noise reduction with minimal artifacts. Total RMS preserved — only quiet segments attenuated.

### Overall best configs (2026-05-23)

| Priority | Name | Techniques | Quiet RMS | RTF |
|----------|------|-----------|-----------|-----|
| 1 (best audio) | CLE + Dual-pass + Gate | CLE + SQNR + dual-offset + noise gate | **~20** (est) | ~0.24 |
| 2 (balanced) | CLE + Gate | CLE + SQNR + noise gate | **60 (raw: 118→60, -49%)** | ~0.11 |
| 3 (fastest) | CLE | CLE + SQNR | 200 | ~0.11 |

### Noise gate integration (2026-05-23)

Integrated into TTS module (`tts.py`) via `_apply_noise_gate()` in `_qnn_synthesize_to_wav`. Applied after DSP output, before normalization.

```python
QNN_NOISE_GATE_ENABLED = True   # 开关
QNN_NOISE_GATE_STRENGTH = 0.5   # 0.0=旁路, 0.5=推荐, 1.0=最强
```

Algorithm: soft noise gate based on short-time SNR. 20ms windows, noise floor estimated from quietest 10%. Sigmoid knee for smooth transition. CPU cost ~27ms for 3s audio. Loudness and speech segments fully preserved.

## Dev environment (Windows)

- **Node.js**: Portable install at `%LOCALAPPDATA%\nodejs` (v24.15.0), added to user PATH
- **caveman**: Claude Code plugin + hooks (SessionStart, UserPromptSubmit) + statusline badge + `caveman-shrink` MCP. Installed via `node bin/install.js --only claude --force` from local clone at `D:\ai_model\melotts_qnn_dsp\caveman\`
- **adb**: `adb -s 28de40d2` (must `adb root` before writing to board)
