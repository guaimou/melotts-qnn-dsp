# MeloTTS QNN DSP Acceleration

MeloTTS HiFiGAN Generator running on Qualcomm Hexagon v68 DSP via QNN framework. Achieves **RTF 0.136** (202ms/chunk) — 3-6x faster than ONNX CPU on SC171v3 (QCS6490).

## Performance

| Backend | Quantization | RTF | ms/chunk | vs ONNX CPU |
|---------|-------------|-----|----------|-------------|
| **QNN DSP** | **INT8 (per-channel)** | **0.136** | **202ms** | **3-6x** |
| QNN DSP | W8A16 | 0.417 | 619ms | Best quality |
| SNPE DSP | INT16 | 1.92 | 2859ms | 0.5x (slower) |
| ONNX CPU | FP32 | 0.55 | 820ms | baseline |
| Official FiboTTS | INT16 | 0.38 | 571ms | — |

### Full pipeline (encoder CPU + vocoder DSP)

| Voice | ONNX CPU | QNN DSP | Speedup |
|-------|---------|---------|---------|
| default (198MB) | 1.5-3.0s | 0.9-1.8s | 1.6-1.9x |
| A2 (623MB) | 2.4-4.2s | 0.5-1.1s | 3.8-5.8x |
| A13 (623MB) | 3.4-4.5s | 0.9-1.6s | 2.1-3.7x |

## Quantization variants

14 quantization approaches tested. Key milestones:

| Variant | Method | Quiet RMS (noise floor) | Status |
|---------|--------|------------------------|--------|
| v7 pch | Per-channel SQNR | 160 | Deployed |
| v10 adaround | AIMET AdaRound + INT8 | 196 | Testing |
| v11 cle | CLE + SQNR | 156 | Testing |
| v13 noise_gate | CLE + SQNR + Noise Gate | 45 (-77.5%) | Testing |
| v14 w8a16 | CLE + W8A16 | best (-89.8% vs INT8) | Deployed (default) |

### Noise reduction techniques

| Technique | Quiet RMS | Processing cost | Notes |
|-----------|-----------|----------------|-------|
| Dual-pass residual INT8 | -24.5% | 2x DSP time | 0.5 LSB offset, average outputs |
| Soft noise gate | -77.5% | ~27ms/3s audio | Recommended for production |
| Spectral subtraction | -84% | ~23ms/3s audio | Risk of oversubtraction |

## Architecture

```
Text → [CPU: MeloTTS Encoder] → z_latent → [DSP: QNN INT8 Generator .so] → Audio (44100Hz)
```

- **SDK**: QNN 2.26.2 (2.29+ has 1D Conv shape inference regression)
- **Quantization**: INT8 (v68 DSP doesn't support FP32/INT16)
- **Model format**: `.so` built by `qnn-model-lib-generator`, loaded by `fiboaisdk` InferAPI
- **No .fmodel packaging** — fiboaisdk QNN framework loads `.so` directly

## Project structure

```
melotts_qnn_dsp/
├── scripts/                    # Conversion, export, test, benchmark scripts
│   ├── export_optimized_generator.py    # PyTorch → ONNX export
│   ├── conv262_qnn.sh                   # ONNX → QNN INT8 conversion (Docker)
│   ├── board_build_qnn.sh               # Build .so on board
│   └── board_test_qnn*.py               # Board inference tests
├── board_audio/                # Board-side scripts + AIMET quantization
│   ├── aimet_adaround*.py     # AIMET AdaRound (v1/v2/v3)
│   ├── aimet_cle.py           # Cross-Layer Equalization
│   ├── gen_calib*.py          # Calibration data generation
│   ├── run_melo_tts.py        # Full TTS pipeline on board
│   └── test_*.py              # DSP inference tests
├── melo_runtime/               # MeloTTS runtime (from board)
├── checkpoints/                # MeloTTS model source code
├── config.json                 # Model config (44100Hz, 256 speakers)
├── tts_board_current.py        # Current board TTS module
└── requirements.txt
```

## Quick start

### 1. Export ONNX from PyTorch

```bash
python scripts/export_optimized_generator.py
```

### 2. Convert ONNX → QNN INT8 (in Docker)

```bash
docker exec my_work bash /project/cosyvoice_snpe_snpe/conv_quant.sh
```

### 3. Build .so on board

```bash
adb shell "cd /home/fibo/melotts_qnn && bash build_test_qnn.sh"
```

### 4. Test on board

```bash
adb shell "python3 /tmp/test_qnn_so.py"
```

## Board environment

- **Device**: SC171v3, QCS6490, Hexagon v68 DSP
- **OS**: Ubuntu 20.04 aarch64, Python 3.8.10
- **SDK**: fiboaisdk 2.0.20, QNN SDK 2.26.2
- **Access**: `adb -s 28de40d2 root`

## SDK compatibility

| Feature | SDK 2.26.2 | SDK 2.29 | SDK 2.32.6 |
|---------|-----------|----------|------------|
| qnn-onnx-converter (1D Conv) | Works | Crash | Crash |
| qairt-converter | Works | N/A | Works |
| INT8 quantization | Works | N/A | Via qairt-quantizer |
| qnn-model-lib-generator | Works | Works | Broken (.cpp=ZIP) |

**Use SDK 2.26.2** for HiFiGAN 1D Conv models. Later versions have shape inference regression.

## Audio samples

`output/tts_samples/` — 30 WAV files across default/A2/A13 voices, multiple text types (daily, news, story, tech, intro).

## Key findings

- Hexagon v68 only supports INT8 ops in QNN backend. FP32 and INT16 both fail at op validation.
- Per-channel SQNR calibration is critical — min-max INT8 is significantly worse.
- DSP API init is ~81s first load (one-time cost), subsequent calls use cached context.
- AdaRound FP32 output is identical to original (corr=1.0) — benefits appear during INT8 quantization.
- CLE + SQNR produces the lowest noise floor among INT8 variants.

## Related projects

- [CosyVoice_edge](https://github.com/guaimou/CosyVoice_edge) — CosyVoice + SNPE edge TTS (same board)
- [MeloTTS](https://github.com/myshell-ai/MeloTTS) — upstream MeloTTS project
