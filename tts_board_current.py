#!/usr/bin/env python3
"""
统一 TTS 模块 — 云端(腾讯WebSocket) / 本地QNN DSP(ONNX编码器 + QNN声码器) 双后端
默认本地QNN DSP，可切换云端。

对外接口:
  speak_text(text)             — 合成+播放
  synthesize_to_file(text, path) — 合成到文件
  synthesize_to_wav(text, wav_path) — 合成到指定WAV路径（供流式管道用）
  play_wav_file(wav_path)      — 播放已有WAV
  get_stream_player()          — 获取 StreamAudioPlayer
  _get_tts()                   — 获取TTS实例（兼容 link_test.py）
  Config                       — 配置类
  get_last_tts_metrics()       — 最近一次指标
  set_cloud_mode() / set_local_mode() — 切换后端
  set_voice() / get_voice()    — 音色切换
"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import struct
import subprocess
import threading
import time
import uuid
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import wave

# ==================== TTS 模式 ====================

_tts_mode = "local"  # "cloud" | "local" (local=QNN DSP)
_mode_lock = threading.Lock()

# 当前音色（云端: yuehua/zhimu/xingchen/..., 本地QNN: default/a2/a13）
_current_voice = "default"
_voice_lock = threading.Lock()


def set_cloud_mode():
    """切换到云端 TTS（腾讯WebSocket）"""
    global _tts_mode
    with _mode_lock:
        _tts_mode = "cloud"
    print("[TTS] 已切换到云端TTS（腾讯WebSocket）")


def set_local_mode():
    """切换到本地 TTS（QNN DSP 板端加速）"""
    global _tts_mode
    with _mode_lock:
        _tts_mode = "local"
    print("[TTS] 已切换到本地TTS（QNN DSP 板端加速）")


def get_mode():
    return _tts_mode


def set_voice(voice_name):
    """设置TTS音色（影响后续所有合成）

    本地QNN: 支持 default / a2 / a13 三种音色
    云端可用音色: yuehua, zhimu, xingchen, yunjian, yunwan, xiaowu 等
    """
    global _current_voice
    with _voice_lock:
        _current_voice = voice_name
    mode_label = "本地QNN DSP" if _tts_mode == "local" else "云端"
    print(f"[TTS] 音色已切换为: {voice_name} ({mode_label}模式)")


def get_voice():
    """获取当前TTS音色"""
    with _voice_lock:
        return _current_voice


def preload():
    """预加载TTS模型（启动时调用，避免首次合成时延迟）"""
    if _tts_mode == "local":
        api = _qnn_get_api()
        if api is not None:
            print("[TTS] QNN DSP模型预加载完成")
        else:
            print("[TTS] QNN DSP模型预加载失败")
    else:
        print("[TTS] 云端TTS模式，无需预加载模型")


# ==================== 声卡检测（懒加载，共用） ====================

SOUND_CARD_NAME = "lahainayupikiot"
_card_index = None
_card_lock = threading.Lock()


def _get_card_index():
    """懒获取声卡索引（仅首次调用时检测）"""
    global _card_index
    if _card_index is None:
        with _card_lock:
            if _card_index is None:
                _card_index = _detect_sound_card(SOUND_CARD_NAME)
                if _card_index:
                    _setup_mixer(_card_index)
    return _card_index


def _detect_sound_card(card_name):
    try:
        ret = subprocess.run(
            ["cat", "/proc/asound/cards"],
            capture_output=True, text=True,
        )
        for line in ret.stdout.splitlines():
            if card_name in line:
                parts = line.split()
                if parts and parts[0].isdigit():
                    return parts[0]
    except Exception:
        pass
    return None


def _setup_mixer(card_index):
    if not card_index:
        return
    mixer_cmds = [
        ["amixer", "-c", str(card_index), "cset", "numid=243,iface=MIXER,name='RX HPH Mode'", "CLS_AB"],
        ["amixer", "-c", str(card_index), "cset", "numid=90,iface=MIXER,name='RX_MACRO RX0 MUX'", "AIF1_PB"],
        ["amixer", "-c", str(card_index), "cset", "numid=91,iface=MIXER,name='RX_MACRO RX1 MUX'", "AIF1_PB"],
        ["amixer", "-c", str(card_index), "cset", "numid=6639,iface=MIXER,name='RX_CDC_DMA_RX_0 Channels'", "Two"],
        ["amixer", "-c", str(card_index), "cset", "numid=112,iface=MIXER,name='RX INT0_1 MIX1 INP0'", "RX0"],
        ["amixer", "-c", str(card_index), "cset", "numid=115,iface=MIXER,name='RX INT1_1 MIX1 INP0'", "RX1"],
        ["amixer", "-c", str(card_index), "cset", "numid=107,iface=MIXER,name='RX INT0 DEM MUX'", "CLSH_DSM_OUT"],
        ["amixer", "-c", str(card_index), "cset", "numid=108,iface=MIXER,name='RX INT1 DEM MUX'", "CLSH_DSM_OUT"],
        ["amixer", "-c", str(card_index), "cset", "numid=137,iface=MIXER,name='RX_COMP1 Switch'", "1"],
        ["amixer", "-c", str(card_index), "cset", "numid=138,iface=MIXER,name='RX_COMP2 Switch'", "1"],
        ["amixer", "-c", str(card_index), "cset", "numid=244,iface=MIXER,name='HPHL_COMP Switch'", "1"],
        ["amixer", "-c", str(card_index), "cset", "numid=245,iface=MIXER,name='HPHR_RDAC Switch'", "1"],
        ["amixer", "-c", str(card_index), "cset", "numid=269,iface=MIXER,name='HPHL_RDAC Switch'", "1"],
        ["amixer", "-c", str(card_index), "cset", "numid=270,iface=MIXER,name='HPHR_RDAC Switch'", "1"],
        ["amixer", "-c", str(card_index), "cset", "numid=520,iface=MIXER,name='RX_CDC_DMA_RX_0 Audio Mixer MultiMedia1'", "1"],
    ]
    for cmd in mixer_cmds:
        try:
            subprocess.run(cmd, capture_output=True)
        except Exception:
            pass


# ==================== aplay 播放（共用） ====================

def _play_wav_aplay(wav_path, card_index=None):
    """用 aplay 直接播放 WAV，返回 True/False"""
    ci = card_index or _get_card_index()
    if not ci or not os.path.exists(wav_path):
        return False
    try:
        with open(wav_path, "rb") as f:
            f.read(12)
            sr, ch = 16000, 1
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                csize = struct.unpack("<I", f.read(4))[0]
                if chunk_id == b"fmt ":
                    fmt = struct.unpack("<HHIIHH", f.read(16))
                    ch, sr = fmt[1], fmt[2]
                    f.read(csize - 16)
                elif chunk_id == b"data":
                    break
                else:
                    f.read(csize)
        cmd = ["aplay", "-D", f"plughw:{ci},0",
               "-f", "S16_LE", "-c", str(ch), "-r", str(sr), wav_path]
        proc = subprocess.run(cmd, capture_output=True)
        return proc.returncode == 0
    except Exception as e:
        print(f"[TTS] aplay播放失败: {e}")
        return False


# ==================== 全局指标 ====================

LAST_TTS_METRICS = {
    "first_audio_latency_s": None,
    "synthesis_latency_s": None,
    "playback_ok": False,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Config（link_test.py 兼容） ====================

class Config:
    AUDIO_CONFIG = {
        "sample_rate": 16000,
        "channels": 1,
        "format": 1,  # PCM 16bit
    }
    TTS_CONFIG = {
        "output_dir": str(OUTPUT_DIR),
        "language": "zh",
        "speed": 0.85,
    }


# ==================== StreamAudioPlayer（link_test.py 兼容） ====================

class StreamAudioPlayer:
    """共享音频管道：多个WAV文件顺序播放，避免声卡竞争

    核心优化：持久化 aplay 进程 + stdin 管道模式
    - 不再每段音频启动/退出 aplay 进程（避免 ALSA 设备反复开关的段间间隙）
    - 持久化线程：队列空时 wait 而非退出
    - PCM 数据直接写入 aplay stdin，段间无缝衔接
    """
    def __init__(self):
        self._queue = []          # [(wav_path, start_tts), ...]
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._play_thread = None
        self._stop_event = threading.Event()
        self._first_aplay_start = None
        self._is_playing = False
        self._aplay_proc = None
        self._sample_rate = 16000
        self._channels = 1
        self._bytes_per_sec = self._sample_rate * self._channels * 2
        self._pcm_bytes_written = 0
        self._write_start_time = None

    def play_wav_data(self, wav_path, start_tts=None):
        """将WAV文件加入播放队列（非阻塞，立即返回）"""
        if not os.path.exists(wav_path):
            print(f"[StreamPlayer] 文件不存在: {wav_path}")
            return False
        with self._not_empty:
            self._queue.append((wav_path, start_tts))
            self._not_empty.notify()
            if self._play_thread is None or not self._play_thread.is_alive():
                self._stop_event.clear()
                self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
                self._play_thread.start()
        return True

    def flush_and_wait(self, timeout=30.0):
        """等待队列中所有音频播放完毕（含ALSA驱动缓冲+扬声器惯性余量）"""
        _ALSA_MARGIN_S = 0.3
        start = time.perf_counter()
        while (time.perf_counter() - start) < timeout:
            with self._not_empty:
                if not self._queue and not self._is_playing:
                    remaining = self._estimated_playback_remaining()
                    if remaining <= 0:
                        time.sleep(_ALSA_MARGIN_S)
                        return True
            time.sleep(0.05)
        print("[StreamPlayer] flush_and_wait 超时")
        return False

    def reset_timing(self):
        """重置计时标记"""
        self._first_aplay_start = None
        self._pcm_bytes_written = 0
        self._write_start_time = None

    def stop(self):
        """停止播放并清空队列"""
        self._stop_event.set()
        with self._not_empty:
            self._queue.clear()
            self._not_empty.notify()
        self._kill_aplay()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=5.0)

    # ==================== 持久 aplay 管道管理 ====================

    def _ensure_aplay(self):
        """确保持久 aplay 进程存活，死掉则重启"""
        if self._aplay_proc is not None and self._aplay_proc.poll() is None:
            return True
        card_index = _get_card_index()
        if not card_index:
            return False
        cmd = ["aplay", "-D", f"plughw:{card_index},0",
               "-f", "S16_LE", "-c", str(self._channels), "-r", str(self._sample_rate), "-"]
        try:
            self._aplay_proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._pcm_bytes_written = 0
            self._write_start_time = None
            return True
        except Exception as e:
            print(f"[StreamPlayer] aplay 管道启动失败: {e}")
            return False

    def _kill_aplay(self):
        """杀掉持久 aplay 进程"""
        if self._aplay_proc is None:
            return
        try:
            self._aplay_proc.stdin.close()
        except Exception:
            pass
        try:
            self._aplay_proc.terminate()
            self._aplay_proc.wait(timeout=2)
        except Exception:
            try:
                self._aplay_proc.kill()
            except Exception:
                pass
        self._aplay_proc = None
        self._pcm_bytes_written = 0
        self._write_start_time = None

    def _estimated_playback_remaining(self):
        """估算 aplay 缓冲中剩余的播放时间（秒）"""
        if self._pcm_bytes_written == 0 or self._write_start_time is None:
            return 0
        total_duration = self._pcm_bytes_written / self._bytes_per_sec
        elapsed = time.perf_counter() - self._write_start_time
        remaining = total_duration - elapsed
        return max(0, remaining)

    # ==================== 播放核心 ====================

    def _play_loop(self):
        """后台线程：依次播放队列中的WAV文件（持久化，队列空时等待而非退出）"""
        _IDLE_TIMEOUT = 5.0
        idle_since = None

        while not self._stop_event.is_set():
            with self._not_empty:
                while not self._queue and not self._stop_event.is_set():
                    self._is_playing = False
                    self._not_empty.notify_all()
                    self._not_empty.wait(timeout=1.0)
                    if not self._queue and self._aplay_proc is not None:
                        if idle_since is None:
                            idle_since = time.perf_counter()
                        elif time.perf_counter() - idle_since >= _IDLE_TIMEOUT:
                            self._kill_aplay()
                            idle_since = None
                if self._stop_event.is_set() and not self._queue:
                    break
                if not self._queue:
                    continue
                wav_path, start_tts = self._queue.pop(0)
                self._is_playing = True
                idle_since = None
            self._play_one(wav_path)
        self._kill_aplay()

    def _play_one(self, wav_path):
        """播放单个WAV文件 — 提取PCM数据写入持久 aplay 管道"""
        if not os.path.exists(wav_path):
            return
        try:
            pcm_data = self._extract_wav_pcm(wav_path)
            if not pcm_data:
                return
            if not self._ensure_aplay():
                return
            if self._first_aplay_start is None:
                self._first_aplay_start = time.perf_counter()
            if self._write_start_time is None:
                self._write_start_time = time.perf_counter()
            try:
                self._aplay_proc.stdin.write(pcm_data)
                self._aplay_proc.stdin.flush()
                self._pcm_bytes_written += len(pcm_data)
            except BrokenPipeError:
                self._kill_aplay()
                if not self._ensure_aplay():
                    return
                self._write_start_time = time.perf_counter()
                self._aplay_proc.stdin.write(pcm_data)
                self._aplay_proc.stdin.flush()
                self._pcm_bytes_written += len(pcm_data)
        except Exception as e:
            print(f"[StreamPlayer] 播放失败: {e}")
        finally:
            try:
                if os.path.basename(wav_path).startswith("stream_tts_") and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

    @staticmethod
    def _extract_wav_pcm(wav_path):
        """从WAV文件提取原始PCM数据（跳过WAV头部）"""
        try:
            with open(wav_path, "rb") as f:
                data = f.read()
            idx = data.find(b"data")
            if idx < 0:
                return None
            pcm_start = idx + 8
            return data[pcm_start:]
        except Exception as e:
            print(f"[StreamPlayer] 读取WAV失败: {e}")
            return None


_stream_player = None


def get_stream_player():
    """获取 StreamAudioPlayer（共享音频管道）"""
    global _stream_player
    if _stream_player is not None:
        return _stream_player
    _stream_player = StreamAudioPlayer()
    print("[TTS] StreamAudioPlayer 已初始化（aplay管道）")
    return _stream_player


# ==================== 本地 QNN DSP TTS ====================

# MeloTTS PyTorch 模型路径（编码器：text → z_latent）
MELO_RUNTIME_DIR = str(Path(__file__).resolve().parent / "before" / "melo_minimal" / "melo_minimal_bundle" / "runtime")
MELO_MODELS_DIR = str(Path(__file__).resolve().parent / "before" / "melo_minimal" / "melo_minimal_bundle" / "models")

# QNN DSP 声码器 .so
QNN_SO_PATH = "/home/fibo/melotts_qnn/lib_int8/aarch64-ubuntu-gcc9.4/libmelotts_int8.so"

# 编码器输出 / 声码器输入参数
INTER_CHANNELS = 192  # VITS inter_channels
QNN_CHUNK_T = 128     # QNN INT8 校准的时间维度
HOP_LENGTH = 512       # VITS hop_length
SAMPLE_RATE = 44100    # 模型采样率

# 音色 → (checkpoint, speaker_id)
QNN_VOICE_MAP = {
    "default": (os.path.join(MELO_MODELS_DIR, "default_zh", "G_default.pth"), 1),
    "a2":      (os.path.join(MELO_MODELS_DIR, "a2", "G_54600.pth"), 0),
    "a13":     (os.path.join(MELO_MODELS_DIR, "a13", "G_34400.pth"), 0),
}

_qnn_api = None          # InferAPI 实例
_qnn_encoder = None      # (model, hps, symbol_to_id, device) 编码器实例
_qnn_current_voice = None  # 当前已加载的编码器音色
_qnn_lock = threading.Lock()


def _qnn_load_encoder(voice="default"):
    """加载 MeloTTS PyTorch 编码器（text → z_latent）"""
    global _qnn_encoder, _qnn_current_voice

    if _qnn_encoder is not None and _qnn_current_voice == voice:
        return _qnn_encoder

    if MELO_RUNTIME_DIR not in sys.path:
        sys.path.insert(0, MELO_RUNTIME_DIR)

    from deploy import _install_local_download_utils
    _install_local_download_utils()

    from melo import utils as melo_utils
    from melo.models import SynthesizerTrn

    ckpt_path, speaker_id = QNN_VOICE_MAP[voice]
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")

    hps = melo_utils.get_hparams_from_file(config_path)
    if hasattr(hps.data, "disable_bert"):
        hps.data.disable_bert = True
    else:
        setattr(hps.data, "disable_bert", True)

    device = "cpu"
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=hps.num_tones,
        num_languages=hps.num_languages,
        **hps.model,
    ).to(device)
    model.eval()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)

    symbol_to_id = {s: i for i, s in enumerate(hps.symbols)}

    _qnn_encoder = (model, hps, symbol_to_id, device, speaker_id)
    _qnn_current_voice = voice

    print(f"[QNN-TTS] 编码器加载完成: voice={voice}, speaker_id={speaker_id}")
    return _qnn_encoder


def _qnn_encode(text, voice="default", speed=1.0, sdp_ratio=0.2,
                noise_scale=0.6, noise_scale_w=0.8):
    """文本 → z_latent（PyTorch CPU 编码器）

    返回: (z_latent_np, y_lengths)
      z_latent_np: numpy array shape [1, 192, T_enc]
      y_lengths: 有效长度
    """
    model, hps, symbol_to_id, device, speaker_id = _qnn_load_encoder(voice)

    language = "ZH_MIX_EN"

    from melo import utils as melo_utils
    from melo.split_utils import split_sentence

    texts = split_sentence(text, language_str=language)

    all_z = []
    total_len = 0

    for t in texts:
        if language in ["EN", "ZH_MIX_EN"]:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

        bert, ja_bert, phones, tones, lang_ids = melo_utils.get_text_for_tts_infer(
            t, language, hps, device, symbol_to_id
        )

        with torch.no_grad():
            x = phones.to(device).unsqueeze(0)
            tones = tones.to(device).unsqueeze(0)
            lang_ids = lang_ids.to(device).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)
            ja_bert = ja_bert.to(device).unsqueeze(0)
            x_lengths = torch.LongTensor([phones.size(0)]).to(device)

            sid = torch.LongTensor([speaker_id]).to(device)

            # --- 编码器前向（复制 model.infer 但去掉 self.dec） ---
            if model.n_speakers > 0:
                g = model.emb_g(sid).unsqueeze(-1)
            else:
                raise RuntimeError("n_speakers=0 not supported")

            g_p = None if model.use_vc else g

            x_enc, m_p, logs_p, x_mask = model.enc_p(
                x, x_lengths, tones, lang_ids, bert, ja_bert, g=g_p
            )

            logw = (model.sdp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio +
                    model.dp(x_enc, x_mask, g=g) * (1 - sdp_ratio))
            w = torch.exp(logw) * x_mask * (1.0 / speed)
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(melo_utils_get_sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = melo_utils_get_generate_path(w_ceil, attn_mask)

            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            z = model.flow(z_p, y_mask, g=g, reverse=True)

            max_len = None  # no truncation
            z_latent = (z * y_mask)[:, :, :max_len]  # [1, 192, T_enc]

            all_z.append(z_latent)
            total_len += z_latent.size(2)

    if len(all_z) == 0:
        raise RuntimeError("编码器未产生输出")

    # 拼接所有句子
    z_cat = torch.cat(all_z, dim=2)  # [1, 192, total_T]
    z_np = z_cat.cpu().numpy().astype(np.float32)

    return z_np, z_cat.size(2)


# 导入 melo commons 函数用于编码器
def _qnn_import_commons():
    if MELO_RUNTIME_DIR not in sys.path:
        sys.path.insert(0, MELO_RUNTIME_DIR)
    from melo import commons as _com
    return _com


def melo_utils_get_sequence_mask(lengths, maxlen):
    com = _qnn_import_commons()
    return com.sequence_mask(lengths, maxlen)


def melo_utils_get_generate_path(w_ceil, attn_mask):
    com = _qnn_import_commons()
    return com.generate_path(w_ceil, attn_mask)


def _qnn_get_api():
    """获取或初始化 QNN DSP 声码器（持久化单例）"""
    global _qnn_api
    if _qnn_api is not None:
        return _qnn_api
    with _qnn_lock:
        if _qnn_api is not None:
            return _qnn_api
        try:
            from fiboaisdk.api_aisdk_py import api_infer_py
        except ImportError:
            print("[QNN-TTS] fiboaisdk 不可用，无法使用 QNN DSP")
            return None
        try:
            params = api_infer_py.InferParams(
                QNN_SO_PATH, "QUALCOMM", "QNN", "DSP", "ERROR", 5
            )
            api = api_infer_py.InferAPI()
            ret = api.Init(params)
            if ret != 0:
                print(f"[QNN-TTS] QNN Init 失败: {ret}")
                return None
            _qnn_api = api
            print("[QNN-TTS] QNN DSP INT8 声码器初始化成功")
            return _qnn_api
        except Exception as e:
            print(f"[QNN-TTS] QNN 初始化异常: {e}")
            return None


def _qnn_generate(z_latent):
    """QNN DSP 声码器：z_latent → audio samples

    Args:
        z_latent: numpy array shape [1, 192, T_enc]

    Returns:
        audio: numpy array shape [T_enc * 512] float32
    """
    api = _qnn_get_api()
    if api is None:
        raise RuntimeError("QNN DSP 未初始化")

    _, C, T_enc = z_latent.shape

    if T_enc <= QNN_CHUNK_T:
        # 单次推理：pad 到 128
        pad_len = QNN_CHUNK_T - T_enc
        z_pad = np.pad(z_latent[0], ((0, 0), (0, pad_len)), mode='constant')  # [192, 128]
        z_pad = z_pad[np.newaxis, :, :]  # [1, 192, 128]

        feed = {"z": z_pad.flatten().tolist()}
        ret = api.Execute_float(feed)
        if ret != 0:
            raise RuntimeError(f"QNN Execute_float failed: {ret}")

        output = api.FetchOutputs_float(["y"])
        audio_full = np.array(output["y"], dtype=np.float32).reshape(1, 1, -1)  # [1, 1, 128*512]

        valid_samples = T_enc * HOP_LENGTH
        return audio_full[0, 0, :valid_samples]  # [T_enc * 512]

    else:
        # 分段推理 + overlap-add
        overlap_t = 16  # 12.5% overlap
        hop_t = QNN_CHUNK_T - overlap_t  # 112
        audio_parts = []
        fade_in = np.linspace(0, 1, overlap_t * HOP_LENGTH)
        fade_out = np.linspace(1, 0, overlap_t * HOP_LENGTH)

        prev_tail = None

        pos = 0
        while pos < T_enc:
            end = min(pos + QNN_CHUNK_T, T_enc)
            chunk_len = end - pos
            z_chunk = z_latent[:, :, pos:end]  # [1, 192, chunk_len]

            # pad 到 128 if needed
            if chunk_len < QNN_CHUNK_T:
                pad_len = QNN_CHUNK_T - chunk_len
                z_chunk = np.pad(z_chunk[0], ((0, 0), (0, pad_len)), mode='constant')
                z_chunk = z_chunk[np.newaxis, :, :]
                valid_audio = chunk_len * HOP_LENGTH
            else:
                valid_audio = QNN_CHUNK_T * HOP_LENGTH

            feed = {"z": z_chunk.flatten().tolist()}
            ret = api.Execute_float(feed)
            if ret != 0:
                raise RuntimeError(f"QNN Execute_float failed: {ret}")

            output = api.FetchOutputs_float(["y"])
            audio_chunk = np.array(output["y"], dtype=np.float32).reshape(1, 1, -1)[0, 0, :valid_audio]

            if prev_tail is not None and overlap_t > 0:
                # crossfade in overlap region
                cross_len = overlap_t * HOP_LENGTH
                if cross_len <= len(prev_tail) and cross_len <= len(audio_chunk):
                    prev_tail[-cross_len:] = prev_tail[-cross_len:] * fade_out[:cross_len] + audio_chunk[:cross_len] * fade_in[:cross_len]
                    audio_parts.append(prev_tail[:-cross_len])
                    prev_tail = audio_chunk
                else:
                    audio_parts.append(prev_tail)
                    prev_tail = audio_chunk
            else:
                if prev_tail is not None:
                    audio_parts.append(prev_tail)
                prev_tail = audio_chunk

            pos += hop_t if hop_t > 0 else QNN_CHUNK_T

        if prev_tail is not None:
            audio_parts.append(prev_tail)

        return np.concatenate(audio_parts)


def _qnn_synthesize_to_wav(text, output_path, voice=None, speed=1.0):
    """QNN TTS 合成到 WAV 文件（44100Hz S16_LE）

    流水线: text → [PyTorch Encoder CPU] → z_latent → [QNN DSP Generator] → audio → WAV
    """
    if voice is None:
        voice = get_voice()
    if voice not in QNN_VOICE_MAP:
        raise ValueError(f"未知音色 '{voice}'，可用: {list(QNN_VOICE_MAP.keys())}")

    # Step 1: 编码器（CPU PyTorch）
    z_latent, T_enc = _qnn_encode(text, voice=voice, speed=speed)

    # Step 2: 声码器（QNN DSP）
    audio = _qnn_generate(z_latent)  # float32

    # Step 3: 写入 WAV (44100Hz, S16_LE)
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767.0).astype(np.int16)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    return str(out)


# ==================== 云端 TTS（腾讯 WebSocket） ====================

APP_ID = os.getenv("TENCENT_APP_ID", "")
SECRET_ID = os.getenv("TENCENT_SECRET_ID", "")
SECRET_KEY = os.getenv("TENCENT_SECRET_KEY", "")


def _read_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("\"'")
            if k == "TENCENT_APP_ID":
                globals()["APP_ID"] = v
            elif k == "TENCENT_SECRET_ID":
                globals()["SECRET_ID"] = v
            elif k == "TENCENT_SECRET_KEY":
                globals()["SECRET_KEY"] = v


_read_env()

VOICE_MAP = {
    "yuehua": 501004,
    "zhimu": 101001, "xiaoou": 101002, "xingchen": 101003,
    "yunjian": 101007, "yunwan": 101009, "xiaowu": 101011,
    "yeyang": 101012, "zhijing": 101013, "xiaolian": 101014,
    "weilin": 101015, "feixiao": 101016, "jingchao": 101021,
    "mengmeng": 101023, "xiaodi": 101025, "ruolan": 101033,
    "siyu": 101034, "duoduo": 101035, "xiaoyun": 101037,
    "jingjing": 101038, "xiaoqi": 101039, "yijia": 101043,
    "xiaoman": 101049, "aer": 101055, "xiaoshu": 101057,
    "yingzhu": 101065, "taoqi": 101066, "feitu": 101068,
    "xiangting": 101069,
}

EMOTIONS = [
    "neutral", "sad", "happy", "angry", "fear", "news", "story",
    "radio", "poetry", "call", "sajiao", "disgusted", "amaze",
    "peaceful", "exciting", "aojiao", "jieshuo",
]


def build_params_and_signature(
    app_id, secret_id, secret_key,
    session_id, text, voice_type,
    codec, sample_rate, speed, volume,
    enable_subtitle, expired, timestamp,
    emotion_category="", emotion_intensity=100,
    segment_rate=0, fast_voice_type="",
):
    params = {
        "Action": "TextToStreamAudioWS",
        "AppId": app_id,
        "Codec": codec,
        "EnableSubtitle": str(enable_subtitle).lower(),
        "Expired": expired,
        "SampleRate": sample_rate,
        "SecretId": secret_id,
        "SessionId": session_id,
        "Speed": speed,
        "Text": text,
        "Timestamp": timestamp,
        "VoiceType": voice_type,
        "Volume": volume,
    }
    if fast_voice_type:
        params["VoiceType"] = 200000000
        params["FastVoiceType"] = fast_voice_type
    if emotion_category:
        params["EmotionCategory"] = emotion_category
        params["EmotionIntensity"] = emotion_intensity
    params["SegmentRate"] = segment_rate

    sorted_params = sorted(params.items(), key=lambda x: x[0])
    raw_query = "&".join(f"{k}={v}" for k, v in sorted_params)
    sign_str = f"GETtts.cloud.tencent.com/stream_ws?{raw_query}"

    hmac_sha1 = hmac.new(
        secret_key.encode("utf-8"),
        sign_str.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    signature = base64.b64encode(hmac_sha1).decode("utf-8")
    return params, signature, sorted_params


def build_ws_url(sorted_params, signature):
    param_dict = dict(sorted_params)
    param_dict["Text"] = quote(param_dict["Text"], safe="")
    sorted_encoded = sorted(param_dict.items(), key=lambda x: x[0])
    query = "&".join(f"{k}={v}" for k, v in sorted_encoded)
    sig_encoded = quote(signature, safe="")
    return f"wss://tts.cloud.tencent.com/stream_ws?{query}&Signature={sig_encoded}"


def pcm_to_wav(pcm_data, sample_rate):
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", data_size + 36, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b"data", data_size,
    )
    return header + pcm_data


async def _cloud_synthesize(
    text,
    app_id=None, secret_id=None, secret_key=None,
    voice_type=101001, fast_voice_type="",
    codec="pcm", sample_rate=16000, speed=0, volume=0,
    enable_subtitle=False,
    emotion_category="", emotion_intensity=100, segment_rate=0,
    output_path=None, verbose=True,
):
    """云端 TTS 核心合成（腾讯WebSocket），返回 (audio_bytes, subtitles, saved_path)"""
    app_id = app_id or int(APP_ID) if APP_ID else 0
    secret_id = secret_id or SECRET_ID
    secret_key = secret_key or SECRET_KEY

    if not all([app_id, secret_id, secret_key]):
        raise RuntimeError("云端TTS未配置: 请设置 TENCENT_APP_ID / SECRET_ID / SECRET_KEY")

    session_id = uuid.uuid4().hex
    timestamp = int(time.time())
    expired = timestamp + 86400

    params, signature, sorted_params = build_params_and_signature(
        app_id=int(app_id), secret_id=secret_id, secret_key=secret_key,
        session_id=session_id, text=text, voice_type=voice_type,
        codec=codec, sample_rate=sample_rate, speed=speed,
        volume=volume, enable_subtitle=enable_subtitle,
        expired=expired, timestamp=timestamp,
        emotion_category=emotion_category, emotion_intensity=emotion_intensity,
        segment_rate=segment_rate, fast_voice_type=fast_voice_type,
    )

    ws_url = build_ws_url(sorted_params, signature)

    if verbose:
        print(f"  [云端] 文本: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  [云端] 音色: {voice_type}  采样率: {sample_rate}Hz")

    audio_data = bytearray()
    subtitles = []
    handshake_ok = False

    try:
        import websockets
    except ImportError:
        raise RuntimeError("请安装 websockets: pip install websockets")

    async with websockets.connect(ws_url, ping_interval=None) as ws:
        async for message in ws:
            if isinstance(message, bytes):
                audio_data.extend(message)
            else:
                data = json.loads(message)
                code = data.get("code", -1)
                if code != 0:
                    raise RuntimeError(f"云端TTS错误 code={code}: {data.get('message', '')}")
                if not handshake_ok:
                    if verbose:
                        print(f"  [云端] 握手成功")
                    handshake_ok = True
                if enable_subtitle and data.get("result", {}).get("subtitles"):
                    subtitles.extend(data["result"]["subtitles"])
                if data.get("final") == 1:
                    if verbose:
                        print(f"  [云端] 合成完成, {len(audio_data)/1024:.1f} KB")
                    break

    saved_path = None
    if output_path and audio_data:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if codec == "pcm":
            wav_data = pcm_to_wav(bytes(audio_data), sample_rate)
            out = out.with_suffix(".wav")
        else:
            wav_data = bytes(audio_data)
            out = out.with_suffix(f".{codec}")
        with open(out, "wb") as f:
            f.write(wav_data)
        saved_path = str(out)

    return bytes(audio_data), subtitles, saved_path


# ==================== 统一对外接口 ====================

def _get_tts():
    """获取TTS实例（兼容 link_test.py 的 _get_tts() 接口）"""
    class _TTSAdapter:
        @property
        def api(self):
            return self

        def SpeechSynthesisSync(self, text, output_audio):
            """统一合成接口：text→WAV文件（output_audio.audio_path）"""
            wav_path = output_audio.audio_path
            try:
                if _tts_mode == "local":
                    _qnn_synthesize_to_wav(text, wav_path)
                else:
                    audio_bytes, _, _ = asyncio.run(_cloud_synthesize(
                        text=text,
                        output_path=wav_path,
                        codec="pcm",
                        sample_rate=output_audio.audio_sample_rate,
                        verbose=False,
                    ))
                    if not audio_bytes:
                        return -1
                return 0
            except Exception as e:
                print(f"[TTS] SpeechSynthesisSync 失败: {e}")
                return -1

    return _TTSAdapter()


def synthesize_to_wav(text, wav_path, voice=None, speed=None):
    """合成文本到指定WAV路径（供流式管道使用）
    自动选择云端/本地后端，返回 (wav_path, synth_elapsed_seconds)
    """
    if voice is None:
        voice = get_voice()
    t0 = time.perf_counter()
    try:
        if _tts_mode == "local":
            result = _qnn_synthesize_to_wav(text, wav_path, voice=voice, speed=speed or 1.0)
        else:
            cloud_speed = speed if speed is not None else 0
            voice_type = VOICE_MAP.get(voice, voice)
            if isinstance(voice_type, str):
                try:
                    voice_type = int(voice_type)
                except ValueError:
                    voice_type = 101001
            audio_bytes, _, _ = asyncio.run(_cloud_synthesize(
                text=text,
                voice_type=voice_type,
                speed=cloud_speed,
                codec="pcm",
                sample_rate=16000,
                output_path=wav_path,
                verbose=False,
            ))
            result = wav_path
        elapsed = time.perf_counter() - t0
        return result, elapsed
    except Exception as e:
        print(f"[TTS] synthesize_to_wav 失败: {e}")
        return None, time.perf_counter() - t0


def speak_text(text, voice=None, speed=0, **kwargs):
    """合成 + 播放。voice: 音色名或ID, speed: 云端-2~6"""
    if voice is None:
        voice = get_voice()
    global LAST_TTS_METRICS
    start = time.perf_counter()
    LAST_TTS_METRICS = {
        "first_audio_latency_s": None,
        "synthesis_latency_s": None,
        "playback_ok": False,
    }

    try:
        if _tts_mode == "local":
            out_path = OUTPUT_DIR / f"_tts_speak_{int(time.time())}.wav"
            _qnn_synthesize_to_wav(text, str(out_path), voice=voice)
        else:
            voice_type = VOICE_MAP.get(voice, voice)
            if isinstance(voice_type, str):
                try:
                    voice_type = int(voice_type)
                except ValueError:
                    print(f"[TTS] 未知音色 '{voice}'")
                    return False
            out_path = OUTPUT_DIR / f"_tts_speak_{int(time.time())}.wav"
            asyncio.run(_cloud_synthesize(
                text=text, voice_type=voice_type, speed=speed,
                output_path=str(out_path), verbose=True, **kwargs,
            ))

        synth_time = time.perf_counter() - start

        if _get_card_index():
            LAST_TTS_METRICS["first_audio_latency_s"] = time.perf_counter() - start
        ok = _play_wav_aplay(str(out_path))

        LAST_TTS_METRICS["synthesis_latency_s"] = synth_time
        LAST_TTS_METRICS["playback_ok"] = ok

        try:
            os.remove(str(out_path))
        except Exception:
            pass

        return ok
    except Exception as e:
        print(f"[TTS] speak_text 失败: {e}")
        return False


def synthesize_to_file(text, output_path, voice=None, speed=0, **kwargs):
    """合成到文件（不播放），返回文件路径"""
    if voice is None:
        voice = get_voice()
    try:
        if _tts_mode == "local":
            return _qnn_synthesize_to_wav(text, output_path, voice=voice, speed=1.0 if speed is None else speed)
        else:
            voice_type = VOICE_MAP.get(voice, voice)
            if isinstance(voice_type, str):
                try:
                    voice_type = int(voice_type)
                except ValueError:
                    raise ValueError(f"未知音色 '{voice}'")
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = OUTPUT_DIR / output_path
            _, _, saved = asyncio.run(_cloud_synthesize(
                text=text, voice_type=voice_type, speed=speed,
                output_path=str(output_path), verbose=True, **kwargs,
            ))
            return saved
    except Exception as e:
        print(f"[TTS] synthesize_to_file 失败: {e}")
        return None


def play_wav_file(wav_path):
    """播放已有 WAV 文件"""
    if not os.path.exists(wav_path):
        print(f"[TTS] 文件不存在: {wav_path}")
        return False
    return _play_wav_aplay(wav_path)


def get_speakers():
    """返回可用音色列表"""
    # 本地QNN音色 + 云端音色
    speakers = dict(VOICE_MAP)
    for v in QNN_VOICE_MAP:
        if v not in speakers:
            speakers[v] = v
    return speakers


def get_last_tts_metrics():
    """返回最近一次 TTS 的时延指标"""
    return dict(LAST_TTS_METRICS)


def release():
    """释放资源"""
    global _qnn_api, _qnn_encoder, _stream_player
    if _qnn_api is not None:
        try:
            _qnn_api.Release()
        except Exception:
            pass
        _qnn_api = None
    _qnn_encoder = None
    if _stream_player is not None:
        try:
            _stream_player.stop()
        except Exception:
            pass
        _stream_player = None


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="统一 TTS（云端腾讯 / 本地QNN DSP）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s "你好"                           # 默认本地QNN DSP合成+播放
  %(prog)s "你好" --mode cloud               # 云端合成+播放
  %(prog)s "你好" --voice a2 --speed 1.5     # 指定音色语速（本地QNN: default/a2/a13）
  %(prog)s "你好" --output output/hello.wav  # 仅合成文件
  %(prog)s --play-file output/test.wav       # 播放已有文件
  %(prog)s --list-voices                     # 列出可用音色
        """,
    )
    parser.add_argument("text", type=str, nargs="?", help="待合成文本")
    parser.add_argument("--text-file", "-f", type=str, help="从文件读取文本")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径（不播放）")
    parser.add_argument("--voice", "-v", type=str, default="default", help="音色名称或ID")
    parser.add_argument("--mode", "-m", type=str, default="local", choices=["cloud", "local"],
                        help="TTS模式: cloud(云端腾讯) / local(本地QNN DSP)")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="语速")
    parser.add_argument("--format", type=str, default="pcm", choices=["pcm", "mp3"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--volume", type=float, default=0)
    parser.add_argument("--emotion", type=str, default="", choices=EMOTIONS)
    parser.add_argument("--play-file", type=str, default=None)
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if args.mode == "cloud":
        set_cloud_mode()
    else:
        set_local_mode()

    if args.list_voices:
        print("本地QNN DSP音色:")
        for name, (ckpt, sid) in QNN_VOICE_MAP.items():
            ok = "✓" if os.path.exists(ckpt) else "✗"
            print(f"  {name:12s}  {ok}  sid={sid}")
        print("\n云端音色:")
        for name, vid in sorted(VOICE_MAP.items(), key=lambda x: x[1]):
            print(f"  {name:12s}  {vid:>10d}")
        return

    if args.play_file:
        ok = play_wav_file(args.play_file)
        print(f"播放: {'成功' if ok else '失败'}")
        return

    text = args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    if not text:
        print("错误: 请提供文本")
        sys.exit(1)

    mode_label = "云端腾讯" if _tts_mode == "cloud" else "本地QNN DSP"
    print(f"统一 TTS ({mode_label}) + 播放")
    print("=" * 40)

    if args.output:
        path = synthesize_to_file(
            text=text, output_path=args.output,
            voice=args.voice, speed=args.speed,
            codec=args.format, sample_rate=args.sample_rate,
        )
        if path:
            print(f"  保存至: {path}")
    else:
        speak_text(text, voice=args.voice, speed=args.speed,
                   codec=args.format, sample_rate=args.sample_rate,
                   emotion_category=args.emotion)
        m = get_last_tts_metrics()
        if m["synthesis_latency_s"] is not None:
            print(f"  合成耗时: {m['synthesis_latency_s']:.2f}s")
        print(f"  播放: {'成功' if m['playback_ok'] else '失败'}")


if __name__ == "__main__":
    main()
