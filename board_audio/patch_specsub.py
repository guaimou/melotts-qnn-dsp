"""Replace noise gate with spectral subtraction in TTS module."""
tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# 1. Replace gate config with specsub config
old_cfg = "QNN_NOISE_GATE_STRENGTH = 0.5  # 噪音门强度: 0.0=关闭, 1.0=最强(推荐0.3-0.7)"
new_cfg = "QNN_SPECSUB_ENABLED = True   # 频谱减法降噪\nQNN_SPECSUB_ALPHA = 2.0      # 过减因子: 1.0-3.0(推荐2.0)\nQNN_SPECSUB_BETA = 0.01      # 频谱底: 0.001-0.1(推荐0.01)"
content = content.replace(old_cfg, new_cfg)

# 2. Replace gate call in _qnn_synthesize_to_wav
old_call = "    # Step 2b: 噪音门后处理\n    if QNN_NOISE_GATE_ENABLED:\n        audio = _apply_noise_gate(audio, sr=SAMPLE_RATE, strength=QNN_NOISE_GATE_STRENGTH)"
new_call = "    # Step 2b: 频谱减法降噪\n    if QNN_SPECSUB_ENABLED:\n        audio = _apply_spectral_subtraction(audio, sr=SAMPLE_RATE, alpha=QNN_SPECSUB_ALPHA, beta=QNN_SPECSUB_BETA)"
content = content.replace(old_call, new_call)

# 3. Replace _apply_noise_gate with _apply_spectral_subtraction
old_fn_marker = "def _apply_noise_gate(audio, sr=44100, strength=0.5):"
fn_end_marker = "def _qnn_synthesize_to_wav(text, output_path, voice=None, speed=1.0):"

# Find and replace the function
old_fn_start = content.index(old_fn_marker)
new_fn_start = content.index(fn_end_marker, old_fn_start)
old_fn = content[old_fn_start:new_fn_start]

new_fn = '''def _apply_spectral_subtraction(audio, sr=44100, alpha=2.0, beta=0.01):
    """频谱减法降噪: 估计噪声谱，从所有帧减去。

    alpha: 过减因子 (1.0=温和, 2.0=推荐, 3.0=激进)
    beta:  频谱底 (防止"音乐噪音", 0.01=推荐)
    """
    n_fft = 2048
    hop = 512
    win = np.hanning(n_fft)

    n_frames = (len(audio) - n_fft) // hop + 1
    if n_frames < 3:
        return audio

    # STFT
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        frame = audio[i*hop:i*hop+n_fft] * win
        stft[:, i] = np.fft.rfft(frame)

    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # 噪音谱估计: 取最安静20%帧的中位数
    frame_energies = np.sum(magnitude**2, axis=0)
    quiet_thresh = np.percentile(frame_energies, 20)
    quiet_mask = frame_energies < quiet_thresh
    noise_mag = np.median(magnitude[:, quiet_mask], axis=1) if np.sum(quiet_mask) > 0 else np.zeros(n_fft//2+1)

    # 频谱减法
    clean_mag = np.maximum(magnitude - alpha * noise_mag[:, np.newaxis], beta * magnitude)

    # ISTFT
    clean_stft = clean_mag * np.exp(1j * phase)
    clean_frames = np.fft.irfft(clean_stft, n=n_fft, axis=0)

    # Overlap-add
    out = np.zeros(len(audio))
    for i in range(n_frames):
        out[i*hop:i*hop+n_fft] += clean_frames[:, i] * win

    return out[:len(audio)]

'''
content = content.replace(old_fn, new_fn)

with open(tts_path, 'w') as f:
    f.write(content)

# Verify
checks = ['QNN_SPECSUB_ENABLED', '_apply_spectral_subtraction', 'QNN_SPECSUB_ALPHA']
for c in checks:
    if c in content:
        print(f"  OK: {c}")
    else:
        print(f"  MISSING: {c}")

# Count - ensure no duplicates
for kw in ['_apply_spectral_subtraction', '_apply_noise_gate']:
    count = content.count(kw)
    status = "OK" if (kw == '_apply_spectral_subtraction' and count == 2) or (kw == '_apply_noise_gate' and count == 0) else f"WARN({count})"
    print(f"  {kw}: {status}")
