"""Integrate noise gate into TTS module."""
import re

tts_path = '/home/fibo/AI model/tts_models/tts.py'
with open(tts_path, 'r') as f:
    content = f.read()

# 1. Add noise gate config flag after SAMPLE_RATE
old_cfg = "SAMPLE_RATE = 44100    # 模型采样率"
new_cfg = '''SAMPLE_RATE = 44100    # 模型采样率
QNN_NOISE_GATE_ENABLED = True  # 输出端噪音门: 衰减静音段量化噪音
QNN_NOISE_GATE_STRENGTH = 0.5  # 噪音门强度: 0.0=关闭, 1.0=最强(推荐0.3-0.7)'''
content = content.replace(old_cfg, new_cfg)

# 2. Add noise gate function before _qnn_synthesize_to_wav
noise_gate_fn = '''
def _apply_noise_gate(audio, sr=44100, strength=0.5):
    """软噪音门: 衰减静音段以减少量化噪音。仅处理低能量段，保留语音。

    strength: 0.0=旁路, 0.5=推荐, 1.0=最强压制
    """
    if strength <= 0.01:
        return audio

    # 短时RMS能量 (20ms窗口)
    win_len = int(sr * 0.02)
    hop_len = win_len // 4
    n_wins = (len(audio) - win_len) // hop_len + 1
    if n_wins < 2:
        return audio

    rms = np.array([np.sqrt(np.mean(audio[i*hop_len:i*hop_len+win_len]**2))
                     for i in range(n_wins)])

    # 噪音底估计: 最安静的10%窗口
    noise_rms = np.percentile(rms, 10)
    if noise_rms < 1e-10:
        return audio

    # 逐样本增益: 信噪比越高增益越接近1
    gain = np.ones(len(audio))
    for i in range(n_wins):
        snr = max(0.0, rms[i] / noise_rms - 1.0)  # 线性SNR
        # sigmoid软膝盖: gain = 1 - strength * exp(-SNR)
        g = 1.0 - strength * np.exp(-snr * 3.0)
        g = np.clip(g, 0.01, 1.0)  # 最低增益1%
        start = i * hop_len
        end = min(start + win_len, len(audio))
        # 取逐样本最小值 (保守: 避免增益突变)
        gain[start:end] = np.minimum(gain[start:end], g)

    return audio * gain

'''
content = content.replace(
    "def _qnn_synthesize_to_wav(text, output_path, voice=None, speed=1.0):",
    noise_gate_fn + "def _qnn_synthesize_to_wav(text, output_path, voice=None, speed=1.0):"
)

# 3. Insert noise gate call after DSP generation, before normalization
old_dsp = "    # Step 3: Remove DC offset and normalize, then write WAV (44100Hz, S16_LE)\n    audio = audio - np.mean(audio)"
new_dsp = "    # Step 2b: 噪音门后处理\n    if QNN_NOISE_GATE_ENABLED:\n        audio = _apply_noise_gate(audio, sr=SAMPLE_RATE, strength=QNN_NOISE_GATE_STRENGTH)\n\n    # Step 3: Remove DC offset and normalize, then write WAV (44100Hz, S16_LE)\n    audio = audio - np.mean(audio)"
content = content.replace(old_dsp, new_dsp)

with open(tts_path, 'w') as f:
    f.write(content)

# Verify
if '_apply_noise_gate' in content and 'QNN_NOISE_GATE_ENABLED' in content:
    print("Noise gate integrated successfully!")
    for line in content.split('\n'):
        if 'NOISE_GATE' in line:
            print(f"  {line.strip()}")
else:
    print("ERROR: patch failed")
    # Find the relevant lines
    for i, line in enumerate(content.split('\n')):
        if 'SAMPLE_RATE = 44100' in line:
            print(f"  Line {i}: {line}")
        if 'Step 3: Remove DC offset' in line or 'Step 2b' in line:
            print(f"  Line {i}: {line}")
