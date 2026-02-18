import cv2
import numpy as np
import time
import pyautogui
from collections import deque
import threading
import math

pyautogui.FAILSAFE = False

MODE_GESTURE = 0
MODE_MOUSE = 1
MODE_ASCII = 2
MODE_COLOR = 3
MODE_SYNTH = 4
MODE_SAMPLER = 5
MODE_SYNTH_VISUAL = 7
MODE_ART_PIONEER = 8
current_mode = MODE_GESTURE

synth = None
sampler = None
synth_visual = None
synth_art_pioneer = None
prev_frame_gray = None

# 平滑马赛克位置历史
mosaic_position_history = deque(maxlen=50)

gesture_history = deque(maxlen=5)

# 全局缩放因子（用于适配不同分辨率）
scale_factor = 1.0

# 背景减除器
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)

# 手部位置历史（用于平滑）- 减少历史长度
hand_position_history = deque(maxlen=3)
hand_size_history = deque(maxlen=3)

# 上一次有效的轮廓
last_valid_contour = None
last_valid_time = 0

# 音乐合成器类
class AudioSynthesizer:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.thread = None
        
        # 主振荡器参数
        self.frequency = 440.0
        self.frequency2 = 550.0  # 第二振荡器
        self.detune = 0.0  # 失谐
        
        # LFO参数
        self.lfo_rate = 1.0
        self.lfo_depth = 0.3
        self.lfo_type = 0  # 0=sin, 1=square, 2=triangle
        
        # 包络参数
        self.attack = 0.05
        self.decay = 0.1
        self.sustain = 0.7
        self.release = 0.2
        self.envelope = 0.0
        self.envelope_target = 0.0
        
        # 滤波器参数
        self.filter_cutoff = 1000.0
        self.filter_resonance = 0.5
        
        # 效果参数
        self.reverb_mix = 0.3
        self.delay_time = 0.3
        
        # 相位
        self.phase = 0.0
        self.phase2 = 0.0
        self.lfo_phase = 0.0
        
        # 颜色影响
        self.color_hue = 0.0
        self.color_sat = 0.0
        
        # 音量
        self.volume = 0.4
        
        # 历史缓冲（用于延迟效果）
        self.delay_buffer = np.zeros(int(sample_rate * 0.5))
        self.delay_index = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
    
    def _audio_loop(self):
        try:
            import sounddevice as sd
            with sd.OutputStream(samplerate=self.sample_rate, 
                                channels=1, 
                                callback=self._audio_callback,
                                blocksize=self.buffer_size):
                while self.running:
                    time.sleep(0.005)
        except ImportError:
            print("sounddevice not installed, audio disabled")
            self.running = False
    
    def _audio_callback(self, outdata, frames, time_info, status):
        t = np.arange(frames) / self.sample_rate
        
        # LFO生成
        if self.lfo_type == 0:
            lfo = np.sin(2 * np.pi * self.lfo_rate * (self.lfo_phase + t))
        elif self.lfo_type == 1:
            lfo = np.sign(np.sin(2 * np.pi * self.lfo_rate * (self.lfo_phase + t)))
        else:
            lfo = 2 * np.abs(2 * (self.lfo_phase + t) * self.lfo_rate % 1 - 0.5) - 1
        
        lfo_mod = 1.0 + lfo * self.lfo_depth
        
        # 主振荡器
        freq1 = self.frequency * lfo_mod * (1.0 + self.color_hue * 0.1)
        phase_inc1 = 2 * np.pi * freq1 / self.sample_rate
        self.phase = self.phase + np.cumsum(phase_inc1)
        osc1 = np.sin(self.phase)
        
        # 第二振荡器（失谐）
        freq2 = self.frequency2 * (1.0 + self.detune * 0.1) * lfo_mod
        phase_inc2 = 2 * np.pi * freq2 / self.sample_rate
        self.phase2 = self.phase2 + np.cumsum(phase_inc2)
        osc2 = np.sin(self.phase2)
        
        # 混合振荡器
        wave = osc1 * 0.6 + osc2 * 0.4
        
        # 添加泛音
        wave += np.sin(self.phase * 2) * 0.2 * self.color_sat
        wave += np.sin(self.phase * 3) * 0.1 * self.color_sat
        
        # 简单低通滤波
        cutoff_mod = self.filter_cutoff * (1.0 + lfo * 0.3)
        filter_coef = np.exp(-2 * np.pi * cutoff_mod / self.sample_rate)
        
        # 包络处理
        envelope_diff = self.envelope_target - self.envelope
        if abs(envelope_diff) > 0.001:
            rate = 1.0 / self.attack if envelope_diff > 0 else 1.0 / self.release
            self.envelope += np.sign(envelope_diff) * min(abs(envelope_diff), rate / self.sample_rate * frames)
        
        wave = wave * self.envelope * self.volume
        
        # 延迟效果
        delay_samples = int(self.delay_time * self.sample_rate)
        for i in range(frames):
            delayed = self.delay_buffer[self.delay_index] * 0.3
            wave[i] = wave[i] + delayed
            self.delay_buffer[self.delay_index] = wave[i]
            self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
        
        # 更新相位
        self.phase = self.phase[-1] % (2 * np.pi * 1000)
        self.phase2 = self.phase2[-1] % (2 * np.pi * 1000)
        self.lfo_phase = (self.lfo_phase + self.lfo_rate * frames / self.sample_rate) % 1.0
        
        outdata[:, 0] = np.clip(wave, -1, 1).astype(np.float32)
    
    def update_from_gesture(self, hand_y, hand_area, hand_x, colors, finger_count=0):
        """从手势和颜色更新合成器参数，让声音符合画面"""
        # 手的Y位置控制频率 (高=高频，低=低频)
        self.frequency = 80 + (1.0 - hand_y) * 700  # 80-780 Hz
        self.frequency2 = self.frequency * 1.5  # 五度
        
        # 手的面积控制LFO深度和包络
        self.lfo_depth = min(0.6, hand_area * 0.0008)
        self.envelope_target = min(1.0, hand_area * 0.0015)
        
        # 手的X位置控制LFO速率和滤波器
        self.lfo_rate = 0.3 + hand_x * 4.0  # 0.3-4.3 Hz
        self.filter_cutoff = 200 + hand_x * 2000  # 200-2200 Hz
        
        # 手指数量控制失谐和LFO类型
        self.detune = finger_count * 2.0
        self.lfo_type = finger_count % 3
        
        # 颜色影响 - 让声音符合画面
        if colors is not None and len(colors) > 0:
            # 计算主色调的HSV
            main_color = colors[0]
            r, g, b = main_color[0] / 255.0, main_color[1] / 255.0, main_color[2] / 255.0
            
            # 计算色相 (Hue) 0-360度
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            delta = max_c - min_c
            
            if delta == 0:
                hue = 0
            elif max_c == r:
                hue = 60 * (((g - b) / delta) % 6)
            elif max_c == g:
                hue = 60 * (((b - r) / delta) + 2)
            else:
                hue = 60 * (((r - g) / delta) + 4)
            
            # 计算饱和度 (Saturation)
            saturation = delta / max_c if max_c > 0 else 0
            
            # 计算明度 (Value/Brightness)
            value = max_c
            
            # 色相影响：频率微调（不同色相产生不同音色感觉）
            # 红(0°) -> 低频偏移，蓝(240°) -> 高频偏移
            hue_factor = (hue / 360.0) * 0.25  # ±25%频率偏移（增加影响）
            self.frequency = self.frequency * (1.0 + hue_factor - 0.125)
            
            # 饱和度影响：泛音丰富度和LFO深度
            # 高饱和度 -> 更丰富的泛音，更深的LFO
            self.color_sat = saturation
            self.lfo_depth = min(0.7, self.lfo_depth + saturation * 0.2)  # 增加影响
            
            # 明度影响：音量和滤波器
            # 高明度 -> 更亮的声音（更高的滤波器截止）
            self.filter_cutoff = self.filter_cutoff * (0.6 + value * 0.8)  # 增加影响
            self.volume = 0.3 + value * 0.3  # 明度影响音量
            
            # 存储用于显示
            self.color_hue = hue / 360.0
            
            # 颜色多样性：如果有多种颜色，增加声音复杂度
            if len(colors) >= 3:
                # 计算颜色多样性
                color_variance = np.var(colors[:4], axis=0).mean() / 255.0
                self.detune += color_variance * 5  # 增加失谐
                self.reverb_mix = 0.2 + color_variance * 0.4  # 增加混响
        
        # 延迟时间
        self.delay_time = 0.1 + hand_x * 0.3

# 高级采样器类 - 多音色旋律生成
class TR808Sequencer:
    """TR-808风格音序器采样器 - 支持音频样本和MIDI输出"""
    def __init__(self, sample_rate=48000, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.thread = None
        
        # MIDI支持
        self.midi_enabled = False
        self.midi_port = None
        self._init_midi()
        
        # 音频样本存储
        self.sample_library = {}
        self._generate_sample_library()
        
        # 声音库选择（12种经典鼓机）
        self.sound_kit = '808'
        self.available_kits = ['808', '909', 'linndrum', 'sp1200', 'cr78', 'dmx', 'mmt8', 'tr707', 'sr16', 'dr660', 'r8', 'kpr77']
        
        # 音序器参数（带平滑过渡）
        self.tempo = 120
        self.tempo_target = 120
        self.tempo_smoothing = 0.005
        
        self.step_count = 16
        self.current_step = 0
        self.step_phase = 0.0
        
        # 风格混合系统
        self.style_weights = {
            'classic': 1.0, 'hiphop': 0.0, 'house': 0.0,
            'techno': 0.0, 'breakbeat': 0.0, 'dubstep': 0.0,
            'jungle': 0.0, 'ambient': 0.0, 'trap': 0.0,
            'drill': 0.0, 'garage': 0.0, 'footwork': 0.0,
            'afrobeat': 0.0, 'samba': 0.0, 'reggaeton': 0.0,
            'dnb': 0.0, 'trance': 0.0, 'hardstyle': 0.0,
            'moombahton': 0.0, 'kwaito': 0.0, 'grime': 0.0,
            'crunk': 0.0, 'bounce': 0.0, 'electro': 0.0,
            'ukgarage': 0.0, 'dub': 0.0, 'dancehall': 0.0,
            'minimal': 0.0, 'progressive': 0.0, 'chicago': 0.0,
            'detroit': 0.0, 'fidget': 0.0, 'glitch': 0.0,
        }
        self.style_target = 'classic'
        self.style_transition_speed = 0.05
        
        # 808音色定义（扩展）
        self.drum_kits = {
            'kick': {'name': '底鼓', 'short': 'KD', 'midi_note': 36},
            'snare': {'name': '军鼓', 'short': 'SD', 'midi_note': 38},
            'clap': {'name': '拍手', 'short': 'CP', 'midi_note': 39},
            'hihat_closed': {'name': '闭镲', 'short': 'CH', 'midi_note': 42},
            'hihat_open': {'name': '开镲', 'short': 'OH', 'midi_note': 46},
            'tom_high': {'name': '高通', 'short': 'HT', 'midi_note': 50},
            'tom_mid': {'name': '中通', 'short': 'MT', 'midi_note': 48},
            'tom_low': {'name': '低通', 'short': 'LT', 'midi_note': 45},
            'rimshot': {'name': '边击', 'short': 'RS', 'midi_note': 37},
            'cowbell': {'name': '牛铃', 'short': 'CB', 'midi_note': 56},
            'clave': {'name': '克拉维', 'short': 'CL', 'midi_note': 75},
            'maracas': {'name': '沙锤', 'short': 'MA', 'midi_note': 70},
            'conga_high': {'name': '高康加', 'short': 'CG', 'midi_note': 63},
            'conga_low': {'name': '低康加', 'short': 'CL', 'midi_note': 64},
            'cymbal': {'name': '镲片', 'short': 'CY', 'midi_note': 49},
            'crash': {'name': '碎音镲', 'short': 'CR', 'midi_note': 49},
            'ride': {'name': '叠音镲', 'short': 'RD', 'midi_note': 51},
            'ride_bell': {'name': '叠音铃', 'short': 'RB', 'midi_note': 53},
            'shaker': {'name': '沙筒', 'short': 'SH', 'midi_note': 82},
            'tambourine': {'name': '铃鼓', 'short': 'TB', 'midi_note': 54},
            'woodblock': {'name': '木鱼', 'short': 'WB', 'midi_note': 76},
            'guiro': {'name': '刮筒', 'short': 'GU', 'midi_note': 74},
            'bongo_high': {'name': '高邦戈', 'short': 'BH', 'midi_note': 60},
            'bongo_low': {'name': '低邦戈', 'short': 'BL', 'midi_note': 61},
            'timbale_high': {'name': '高天巴', 'short': 'TH', 'midi_note': 65},
            'timbale_low': {'name': '低天巴', 'short': 'TL', 'midi_note': 66},
            'agogo_high': {'name': '高阿戈戈', 'short': 'AH', 'midi_note': 67},
            'agogo_low': {'name': '低阿戈戈', 'short': 'AL', 'midi_note': 68},
            'cuica': {'name': '奎卡鼓', 'short': 'QK', 'midi_note': 78},
            'whistle': {'name': '口哨', 'short': 'WH', 'midi_note': 72},
        }
        
        # 扩展音序器模式（更多风格）
        self.patterns = {
            'classic': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            },
            'hiphop': {
                'kick': [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,1],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'hihat_open': [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            },
            'house': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
                'hihat_open': [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
            },
            'techno': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'hihat_open': [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'tom_mid': [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            },
            'breakbeat': {
                'kick': [1,0,0,1, 0,0,1,0, 0,0,1,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,1, 0,1,1,0, 1,0,1,0, 1,1,0,1],
            },
            'dubstep': {
                'kick': [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'hihat_open': [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
                'tom_low': [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
            },
            'jungle': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,1],
                'snare': [0,0,0,1, 0,0,0,0, 1,0,0,1, 0,0,0,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'hihat_open': [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
            },
            'ambient': {
                'kick': [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],
                'hihat_closed': [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
                'clap': [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],
                'tom_high': [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            },
            'trap': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'hihat_open': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
                'tom_low': [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
            },
            'drill': {
                'kick': [1,0,0,1, 0,0,0,0, 1,0,0,0, 0,0,1,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'rimshot': [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
            },
            'garage': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,1],
                'hihat_closed': [1,1,0,1, 1,1,0,1, 1,1,0,1, 1,1,0,1],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            },
            'footwork': {
                'kick': [1,0,0,1, 0,0,1,0, 1,0,0,1, 0,0,1,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'tom_high': [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
            },
            'afrobeat': {
                'kick': [1,0,0,1, 0,0,1,0, 1,0,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'conga_high': [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
                'cowbell': [0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0],
            },
            'samba': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
                'snare': [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'conga_low': [0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0],
                'maracas': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
            },
            'reggaeton': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
                'snare': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'cowbell': [0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0],
            },
            'dnb': {
                'kick': [1,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,1,0,1, 1,1,0,1, 1,1,0,1, 1,1,0,1],
                'hihat_open': [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
            },
            'trance': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'hihat_closed': [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
                'hihat_open': [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'cymbal': [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
            },
            'hardstyle': {
                'kick': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            },
            'moombahton': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'cowbell': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
            },
            'kwaito': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
            },
            'grime': {
                'kick': [1,0,0,1, 0,0,0,0, 1,0,0,0, 0,0,1,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'rimshot': [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
            },
            'crunk': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'hihat_open': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
                'tom_low': [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
            },
            'bounce': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            },
            'electro': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'tom_mid': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
            },
            'ukgarage': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,1,0, 0,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,1],
                'hihat_closed': [1,1,0,1, 1,1,0,1, 1,1,0,1, 1,1,0,1],
            },
            'dub': {
                'kick': [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'rimshot': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
            },
            'dancehall': {
                'kick': [1,0,0,0, 0,0,1,0, 0,0,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'cowbell': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
            },
            'minimal': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'hihat_closed': [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
                'rimshot': [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0],
            },
            'progressive': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'hihat_closed': [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0],
                'hihat_open': [0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,1],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'tom_mid': [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
            },
            'chicago': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0],
                'tom_high': [0,0,0,1, 0,0,0,0, 0,0,0,1, 0,0,0,0],
            },
            'detroit': {
                'kick': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'rimshot': [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,1,0],
            },
            'fidget': {
                'kick': [1,0,0,1, 0,0,1,0, 1,0,0,1, 0,0,1,0],
                'snare': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1],
                'clap': [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0],
            },
            'glitch': {
                'kick': [1,0,0,1, 0,0,0,0, 0,1,0,0, 1,0,0,0],
                'snare': [0,0,0,0, 1,0,0,1, 0,0,0,0, 1,0,0,0],
                'hihat_closed': [1,0,1,0, 1,0,1,1, 0,1,0,1, 1,0,1,0],
                'rimshot': [0,1,0,0, 0,0,1,0, 1,0,0,1, 0,0,0,1],
            },
        }
        self.current_pattern = 'classic'
        
        # 动态模式生成
        self.generated_pattern = {k: [0]*16 for k in self.drum_kits.keys()}
        self.pattern_variation_counter = 0
        self.variation_intensity = 0.3
        
        # 808音色参数（带平滑过渡）
        self.drum_params = {
            'kick': {'pitch': 1.0, 'pitch_target': 1.0, 'decay': 0.5, 'decay_target': 0.5, 'tone': 0.5, 'tone_target': 0.5},
            'snare': {'pitch': 1.0, 'pitch_target': 1.0, 'decay': 0.3, 'decay_target': 0.3, 'tone': 0.5, 'tone_target': 0.5, 'snappy': 0.5, 'snappy_target': 0.5},
            'clap': {'decay': 0.3, 'decay_target': 0.3, 'tone': 0.5, 'tone_target': 0.5},
            'hihat_closed': {'decay': 0.1, 'decay_target': 0.1, 'tone': 0.7, 'tone_target': 0.7},
            'hihat_open': {'decay': 0.4, 'decay_target': 0.4, 'tone': 0.7, 'tone_target': 0.7},
            'tom_high': {'pitch': 1.5, 'pitch_target': 1.5, 'decay': 0.3, 'decay_target': 0.3},
            'tom_mid': {'pitch': 1.0, 'pitch_target': 1.0, 'decay': 0.3, 'decay_target': 0.3},
            'tom_low': {'pitch': 0.7, 'pitch_target': 0.7, 'decay': 0.3, 'decay_target': 0.3},
            'rimshot': {'pitch': 1.0, 'pitch_target': 1.0, 'decay': 0.1, 'decay_target': 0.1},
            'cowbell': {'pitch': 1.0, 'pitch_target': 1.0, 'decay': 0.3, 'decay_target': 0.3},
            'clave': {'pitch': 1.0, 'pitch_target': 1.0, 'decay': 0.1, 'decay_target': 0.1},
            'maracas': {'decay': 0.2, 'decay_target': 0.2},
            'conga_high': {'pitch': 1.2, 'pitch_target': 1.2, 'decay': 0.35, 'decay_target': 0.35},
            'conga_low': {'pitch': 0.8, 'pitch_target': 0.8, 'decay': 0.4, 'decay_target': 0.4},
            'cymbal': {'decay': 0.6, 'decay_target': 0.6},
        }
        
        # 声音库预设参数（扩展到12种）
        self.kit_presets = {
            '808': {'kick': {'pitch': 1.0, 'decay': 0.5, 'tone': 0.5}, 'snare': {'pitch': 1.0, 'decay': 0.3, 'tone': 0.5, 'snappy': 0.5}, 'hihat_closed': {'decay': 0.1, 'tone': 0.7}, 'clap': {'decay': 0.3}, 'base_note': 48, 'scale': 'minor', 'mood': 'warm'},
            '909': {'kick': {'pitch': 1.1, 'decay': 0.4, 'tone': 0.6}, 'snare': {'pitch': 1.1, 'decay': 0.25, 'tone': 0.6, 'snappy': 0.6}, 'hihat_closed': {'decay': 0.08, 'tone': 0.8}, 'clap': {'decay': 0.25}, 'base_note': 50, 'scale': 'dorian', 'mood': 'bright'},
            'linndrum': {'kick': {'pitch': 0.95, 'decay': 0.35, 'tone': 0.4}, 'snare': {'pitch': 0.95, 'decay': 0.2, 'tone': 0.4, 'snappy': 0.7}, 'hihat_closed': {'decay': 0.12, 'tone': 0.65}, 'clap': {'decay': 0.35}, 'base_note': 52, 'scale': 'major', 'mood': 'happy'},
            'sp1200': {'kick': {'pitch': 0.9, 'decay': 0.55, 'tone': 0.45}, 'snare': {'pitch': 0.9, 'decay': 0.35, 'tone': 0.45, 'snappy': 0.55}, 'hihat_closed': {'decay': 0.15, 'tone': 0.6}, 'clap': {'decay': 0.4}, 'base_note': 45, 'scale': 'blues', 'mood': 'dark'},
            'cr78': {'kick': {'pitch': 0.85, 'decay': 0.45, 'tone': 0.35}, 'snare': {'pitch': 0.85, 'decay': 0.3, 'tone': 0.35, 'snappy': 0.4}, 'hihat_closed': {'decay': 0.18, 'tone': 0.55}, 'clap': {'decay': 0.45}, 'base_note': 47, 'scale': 'pentatonic', 'mood': 'natural'},
            'dmx': {'kick': {'pitch': 0.92, 'decay': 0.4, 'tone': 0.5}, 'snare': {'pitch': 0.92, 'decay': 0.28, 'tone': 0.5, 'snappy': 0.65}, 'hihat_closed': {'decay': 0.1, 'tone': 0.7}, 'clap': {'decay': 0.32}, 'base_note': 49, 'scale': 'mixolydian', 'mood': 'funky'},
            'mmt8': {'kick': {'pitch': 0.88, 'decay': 0.5, 'tone': 0.48}, 'snare': {'pitch': 0.88, 'decay': 0.32, 'tone': 0.48, 'snappy': 0.58}, 'hihat_closed': {'decay': 0.12, 'tone': 0.65}, 'clap': {'decay': 0.38}, 'base_note': 44, 'scale': 'harmonic_minor', 'mood': 'mysterious'},
            'tr707': {'kick': {'pitch': 1.05, 'decay': 0.35, 'tone': 0.55}, 'snare': {'pitch': 1.05, 'decay': 0.22, 'tone': 0.55, 'snappy': 0.7}, 'hihat_closed': {'decay': 0.06, 'tone': 0.85}, 'clap': {'decay': 0.2}, 'base_note': 51, 'scale': 'chromatic', 'mood': 'industrial'},
            'sr16': {'kick': {'pitch': 0.95, 'decay': 0.45, 'tone': 0.42}, 'snare': {'pitch': 0.95, 'decay': 0.3, 'tone': 0.42, 'snappy': 0.6}, 'hihat_closed': {'decay': 0.14, 'tone': 0.6}, 'clap': {'decay': 0.35}, 'base_note': 46, 'scale': 'aeolian', 'mood': 'rock'},
            'dr660': {'kick': {'pitch': 0.9, 'decay': 0.38, 'tone': 0.38}, 'snare': {'pitch': 0.9, 'decay': 0.25, 'tone': 0.38, 'snappy': 0.55}, 'hihat_closed': {'decay': 0.16, 'tone': 0.58}, 'clap': {'decay': 0.3}, 'base_note': 53, 'scale': 'jazz_minor', 'mood': 'jazz'},
            'r8': {'kick': {'pitch': 0.87, 'decay': 0.48, 'tone': 0.4}, 'snare': {'pitch': 0.87, 'decay': 0.32, 'tone': 0.4, 'snappy': 0.5}, 'hihat_closed': {'decay': 0.13, 'tone': 0.62}, 'clap': {'decay': 0.36}, 'base_note': 48, 'scale': 'arabic', 'mood': 'exotic'},
            'kpr77': {'kick': {'pitch': 1.0, 'decay': 0.42, 'tone': 0.52}, 'snare': {'pitch': 1.0, 'decay': 0.26, 'tone': 0.52, 'snappy': 0.62}, 'hihat_closed': {'decay': 0.09, 'tone': 0.75}, 'clap': {'decay': 0.28}, 'base_note': 54, 'scale': 'lydian', 'mood': 'ethereal'},
        }
        
        # 颜色到声音库的映射（12色调）
        self.color_kit_mapping = {
            (0, 30): '808', (30, 60): '909', (60, 90): 'linndrum', (90, 120): 'sp1200',
            (120, 150): 'cr78', (150, 180): 'dmx', (180, 210): 'mmt8', (210, 240): 'tr707',
            (240, 270): 'sr16', (270, 300): 'dr660', (300, 330): 'r8', (330, 360): 'kpr77',
        }
        
        # 颜色到基底旋律的映射
        self.color_melody_mapping = {
            (0, 60): {'base_notes': [48, 51, 53, 55, 58, 60], 'rhythm_density': 0.7},
            (60, 120): {'base_notes': [52, 54, 57, 59, 62, 64], 'rhythm_density': 0.8},
            (120, 180): {'base_notes': [45, 48, 50, 52, 55, 57], 'rhythm_density': 0.5},
            (180, 240): {'base_notes': [44, 47, 49, 51, 54, 56], 'rhythm_density': 0.6},
            (240, 300): {'base_notes': [43, 46, 48, 50, 53, 55], 'rhythm_density': 0.4},
            (300, 360): {'base_notes': [47, 50, 52, 54, 57, 59], 'rhythm_density': 0.65},
        }
        
        self.current_base_notes = [48, 51, 53, 55, 58, 60]
        self.melody_index = 0
        
        # 样本播放状态
        self.sample_voices = {}
        self.max_voices = 16
        
        # 音量和效果
        self.master_volume = 0.85
        self.master_volume_target = 0.85
        self.drum_volumes = {k: 0.9 for k in self.drum_kits.keys()}
        self.drum_volumes_target = {k: 0.9 for k in self.drum_kits.keys()}
        
        self.swing = 0.0
        self.swing_target = 0.0
        
        # 效果缓冲
        self.reverb_buffer = np.zeros(int(sample_rate * 0.4))
        self.reverb_index = 0
        self.reverb_amount = 0.2
        self.reverb_amount_target = 0.2
        self.delay_buffer = np.zeros(int(sample_rate * 0.5))
        self.delay_index = 0
        self.delay_amount = 0.15
        self.delay_feedback = 0.3
        
        # 滤波器
        self.filter_state = 0.0
        self.filter_cutoff = 0.8
        self.filter_resonance = 0.3
        
        # === 效果器系统 ===
        self.effects = {
            'chorus': {
                'enabled': True, 'rate': 0.6, 'depth': 0.4, 'mix': 0.4,
                'phase': 0.0, 'buffer': np.zeros(int(sample_rate * 0.08)),
                'voices': 3,
            },
            'flanger': {
                'enabled': False, 'rate': 1.0, 'depth': 0.15, 'feedback': 0.3, 'mix': 0.15,
                'phase': 0.0, 'buffer': np.zeros(int(sample_rate * 0.02)),
            },
            'phaser': {
                'enabled': False, 'rate': 0.3, 'depth': 0.3, 'stages': 4, 'mix': 0.2,
                'phase': 0.0, 'allpass_states': [0.0] * 8,
            },
            'distortion': {
                'enabled': False, 'drive': 0.3, 'tone': 0.5, 'mix': 0.15,
                'type': 'soft',
            },
            'compressor': {
                'enabled': False, 'threshold': 0.6, 'ratio': 4.0, 'attack': 0.01, 'release': 0.1,
                'gain': 1.0, 'envelope': 0.0,
            },
            'limiter': {
                'enabled': True, 'threshold': 0.95, 'release': 0.05,
                'envelope': 0.0,
            },
            'reverb': {
                'enabled': False, 'size': 0.5, 'damping': 0.5, 'mix': 0.2,
                'buffers': [np.zeros(int(sample_rate * 0.1 * (i + 1))) for i in range(6)],
                'indices': [0] * 6,
            },
            'delay': {
                'enabled': False, 'time': 0.3, 'feedback': 0.3, 'mix': 0.15,
                'buffer': np.zeros(int(sample_rate * 1.0)), 'index': 0,
            },
            'filter': {
                'enabled': False, 'type': 'lowpass', 'cutoff': 0.8, 'resonance': 0.25,
                'state': [0.0] * 4,
            },
            'eq': {
                'enabled': False,
                'low_gain': 1.0, 'low_freq': 100,
                'mid_gain': 1.0, 'mid_freq': 1000,
                'high_gain': 1.0, 'high_freq': 5000,
                'states': [0.0] * 6,
            },
            'harmonizer': {
                'enabled': False, 'semitones': [7, 12], 'mix': 0.25,
                'voices': [], 'phase_offsets': [0.0, 0.3],
            },
        }
        
        # 效果器状态
        self._chorus_buffer = np.zeros(int(sample_rate * 0.05))
        self._flanger_buffer = np.zeros(int(sample_rate * 0.02))
        self._phaser_states = [0.0] * 8
        self._compressor_envelope = 0.0
        self._limiter_envelope = 0.0
        self._harmonizer_phases = [0.0, 0.0]
        
        # 手势控制状态
        self.hand_y = 0.5
        self.hand_x = 0.5
        self.hand_area = 0.0
        self.finger_count = 0
        
        # 显示参数
        self.display_mode = 'pattern'
        
        # MIDI时钟同步
        self.midi_clock_phase = 0.0
        self.last_midi_clock_time = time.time()
        
        # === 多通道乐器系统 ===
        self.channels = {
            'drums': {'enabled': True, 'volume': 0.8, 'pan': 0.0, 'midi_channel': 9},
            'bass': {'enabled': True, 'volume': 0.7, 'pan': -0.2, 'midi_channel': 1},
            'lead': {'enabled': True, 'volume': 0.6, 'pan': 0.3, 'midi_channel': 2},
            'pad': {'enabled': True, 'volume': 0.4, 'pan': 0.0, 'midi_channel': 3},
            'keys': {'enabled': True, 'volume': 0.5, 'pan': 0.1, 'midi_channel': 4},
            'arp': {'enabled': True, 'volume': 0.5, 'pan': -0.1, 'midi_channel': 5},
        }
        
        # 音阶定义
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
            'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'arabic': [0, 1, 4, 5, 7, 8, 11],
            'japanese': [0, 1, 5, 7, 8],
            'wholetone': [0, 2, 4, 6, 8, 10],
        }
        self.current_scale = 'minor'
        self.root_note = 48
        
        # 旋律音序器
        self.melody_patterns = {
            'bass': {
                'classic': [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                'hiphop': [0, 0, -5, 0, 0, -3, 0, 0, 0, 0, -5, 0, 0, -3, 0, 0],
                'house': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'techno': [0, 0, -3, 0, 0, 0, -5, 0, 0, 0, -3, 0, 0, 0, -7, 0],
                'dnb': [0, 0, 0, -5, 0, 0, -3, 0, 0, 0, 0, -5, 0, 0, -3, 0],
                'trance': [0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0],
                'ambient': [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                'trap': [0, 0, -7, 0, 0, -5, 0, 0, 0, 0, -7, 0, 0, -5, 0, 0],
                'dubstep': [0, 0, 0, -12, 0, 0, 0, 0, 0, 0, 0, -12, 0, 0, 0, 0],
                'dub': [0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0],
                'reggaeton': [0, 0, -5, 0, 0, 0, -5, 0, 0, 0, -5, 0, 0, 0, -5, 0],
                'moombahton': [0, 0, -3, 0, 0, -5, 0, 0, 0, 0, -3, 0, 0, -5, 0, 0],
                'electro': [0, -3, 0, -5, 0, -3, 0, -5, 0, -3, 0, -5, 0, -3, 0, -5],
                'progressive': [0, 0, 0, -3, 0, 0, -5, 0, 0, 0, 0, -3, 0, 0, -5, 0],
                'minimal': [0, 0, 0, 0, 0, 0, 0, -7, 0, 0, 0, 0, 0, 0, 0, -7],
                'chicago': [0, 0, -3, 0, 0, -3, 0, 0, 0, 0, -3, 0, 0, -3, 0, 0],
                'detroit': [0, 0, -5, 0, 0, -3, 0, 0, 0, 0, -5, 0, 0, -3, 0, 0],
                'fidget': [0, -3, 0, -7, 0, -3, 0, -7, 0, -3, 0, -7, 0, -3, 0, -7],
                'glitch': [0, 0, -5, 0, -3, 0, -7, 0, 0, 0, -5, 0, -3, 0, -7, 0],
                'ukgarage': [0, 0, -3, 0, 0, -5, 0, 0, 0, 0, -3, 0, 0, -5, 0, 0],
                'grime': [0, -5, 0, 0, -3, 0, 0, -7, 0, -5, 0, 0, -3, 0, 0, -7],
                'crunk': [0, 0, -7, 0, 0, -5, 0, 0, 0, 0, -7, 0, 0, -5, 0, -12],
                'bounce': [0, 0, -5, 0, 0, -5, 0, 0, 0, 0, -5, 0, 0, -5, 0, 0],
                'hardstyle': [0, 0, 0, -5, 0, 0, 0, -5, 0, 0, 0, -5, 0, 0, 0, -5],
                'dancehall': [0, 0, -5, 0, 0, 0, -5, 0, 0, 0, -5, 0, 0, 0, -5, 0],
                'afrobeat': [0, -3, 0, -5, 0, -3, 0, -5, 0, -3, 0, -5, 0, -3, 0, -5],
                'samba': [0, 0, -3, 0, 0, -5, 0, -3, 0, 0, -3, 0, 0, -5, 0, -3],
                'jungle': [0, 0, 0, -5, 0, 0, -3, 0, 0, 0, 0, -5, 0, 0, -3, -7],
                'breakbeat': [0, 0, -5, 0, 0, -3, 0, 0, 0, 0, -5, 0, 0, -3, 0, -7],
            },
            'lead': {
                'classic': [0, -1, 2, 0, -1, 2, 0, -1, 2, 0, -1, 2, 0, -1, 2, 0],
                'hiphop': [0, -1, 0, 2, 0, -1, 0, 2, 0, -1, 0, 2, 0, -1, 0, 2],
                'house': [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0],
                'techno': [0, 2, 4, 2, 0, 2, 4, 2, 0, 2, 4, 2, 0, 2, 4, 2],
                'trance': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'ambient': [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2],
                'trap': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'dnb': [0, 5, 7, 5, 3, 5, 7, 5, 0, 5, 7, 5, 3, 5, 7, 5],
                'dubstep': [0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 12],
                'electro': [0, 4, 0, 7, 0, 4, 0, 7, 0, 4, 0, 7, 0, 4, 0, 7],
                'progressive': [0, 2, 4, 7, 4, 2, 0, 2, 0, 2, 4, 7, 4, 2, 0, 2],
                'chicago': [0, 3, 5, 7, 5, 3, 0, 3, 0, 3, 5, 7, 5, 3, 0, 3],
                'detroit': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'glitch': [0, 7, 0, 5, 0, 7, 0, 3, 0, 7, 0, 5, 0, 7, 0, 3],
                'fidget': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'ukgarage': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'grime': [0, 5, 0, 7, 0, 5, 0, 7, 0, 5, 0, 7, 0, 5, 0, 7],
                'crunk': [0, 3, 5, 7, 5, 3, 0, 3, 0, 3, 5, 7, 5, 3, 0, 3],
                'bounce': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'hardstyle': [0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7],
                'dancehall': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'afrobeat': [0, 2, 4, 2, 0, 2, 4, 7, 0, 2, 4, 2, 0, 2, 4, 7],
                'samba': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'jungle': [0, 5, 7, 10, 7, 5, 0, 5, 0, 5, 7, 10, 7, 5, 0, 5],
                'breakbeat': [0, 3, 5, 7, 5, 3, 0, 3, 0, 3, 5, 7, 5, 3, 0, 3],
                'minimal': [0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7],
                'reggaeton': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'moombahton': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
            },
            'pad': {
                'classic': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'ambient': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'trance': [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                'progressive': [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0],
                'dub': [0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0],
                'chillout': [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
                'house': [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                'techno': [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0],
                'dnb': [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0],
                'dubstep': [0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0],
                'trap': [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0],
                'electro': [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0],
                'minimal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'afrobeat': [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                'samba': [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0],
                'jungle': [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0],
                'breakbeat': [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                'reggaeton': [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0],
                'moombahton': [0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0],
            },
            'keys': {
                'classic': [0, -1, 0, 2, 0, -1, 0, 2, 0, -1, 0, 2, 0, -1, 0, 2],
                'house': [0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0],
                'jazz': [0, 2, 4, 2, 0, 2, 4, 2, 0, 2, 4, 2, 0, 2, 4, 2],
                'hiphop': [0, 0, 2, 0, -1, 0, 2, 0, 0, 0, 2, 0, -1, 0, 2, 0],
                'neo_soul': [0, 2, 3, 5, 7, 5, 3, 2, 0, 2, 3, 5, 7, 5, 3, 2],
                'gospel': [0, 4, 7, 4, 0, 4, 7, 9, 0, 4, 7, 4, 0, 4, 7, 9],
                'chicago': [0, 3, 5, 3, 0, 3, 5, 3, 0, 3, 5, 3, 0, 3, 5, 3],
                'techno': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'trance': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'dnb': [0, 5, 7, 5, 3, 5, 7, 5, 0, 5, 7, 5, 3, 5, 7, 5],
                'dubstep': [0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 7, 0],
                'trap': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'electro': [0, 4, 0, 7, 0, 4, 0, 7, 0, 4, 0, 7, 0, 4, 0, 7],
                'progressive': [0, 2, 4, 7, 4, 2, 0, 2, 0, 2, 4, 7, 4, 2, 0, 2],
                'detroit': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'glitch': [0, 7, 0, 5, 0, 7, 0, 3, 0, 7, 0, 5, 0, 7, 0, 3],
                'fidget': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'ukgarage': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'grime': [0, 5, 0, 7, 0, 5, 0, 7, 0, 5, 0, 7, 0, 5, 0, 7],
                'crunk': [0, 3, 5, 7, 5, 3, 0, 3, 0, 3, 5, 7, 5, 3, 0, 3],
                'bounce': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'hardstyle': [0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7],
                'dancehall': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'afrobeat': [0, 2, 4, 2, 0, 2, 4, 7, 0, 2, 4, 2, 0, 2, 4, 7],
                'samba': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'jungle': [0, 5, 7, 10, 7, 5, 0, 5, 0, 5, 7, 10, 7, 5, 0, 5],
                'breakbeat': [0, 3, 5, 7, 5, 3, 0, 3, 0, 3, 5, 7, 5, 3, 0, 3],
                'minimal': [0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7],
                'reggaeton': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'moombahton': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
            },
            'arp': {
                'classic': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'trance': [0, 3, 7, 10, 0, 3, 7, 10, 0, 3, 7, 10, 0, 3, 7, 10],
                'techno': [0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7],
                'ambient': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'dnb': [0, 5, 7, 10, 12, 10, 7, 5, 0, 5, 7, 10, 12, 10, 7, 5],
                'electro': [0, 12, 7, 12, 0, 12, 7, 12, 0, 12, 7, 12, 0, 12, 7, 12],
                'progressive': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'minimal': [0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7],
                'glitch': [0, 7, 12, 7, 5, 7, 12, 7, 0, 7, 12, 7, 5, 7, 12, 7],
                'house': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'hiphop': [0, 3, 5, 3, 0, 3, 5, 7, 0, 3, 5, 3, 0, 3, 5, 7],
                'trap': [0, 3, 7, 10, 12, 10, 7, 3, 0, 3, 7, 10, 12, 10, 7, 3],
                'dubstep': [0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7],
                'fidget': [0, 5, 7, 10, 7, 5, 0, 5, 0, 5, 7, 10, 7, 5, 0, 5],
                'ukgarage': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'grime': [0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7, 0, 7, 12, 7],
                'crunk': [0, 3, 7, 10, 7, 3, 0, 3, 0, 3, 7, 10, 7, 3, 0, 3],
                'bounce': [0, 5, 7, 10, 7, 5, 0, 5, 0, 5, 7, 10, 7, 5, 0, 5],
                'hardstyle': [0, 7, 12, 15, 12, 7, 0, 7, 0, 7, 12, 15, 12, 7, 0, 7],
                'dancehall': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'afrobeat': [0, 4, 7, 10, 7, 4, 0, 4, 0, 4, 7, 10, 7, 4, 0, 4],
                'samba': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'jungle': [0, 5, 7, 10, 12, 10, 7, 5, 0, 5, 7, 10, 12, 10, 7, 5],
                'breakbeat': [0, 3, 7, 10, 7, 3, 0, 3, 0, 3, 7, 10, 7, 3, 0, 3],
                'reggaeton': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'moombahton': [0, 5, 7, 10, 7, 5, 0, 5, 0, 5, 7, 10, 7, 5, 0, 5],
                'detroit': [0, 7, 12, 15, 12, 7, 0, 7, 0, 7, 12, 15, 12, 7, 0, 7],
                'chicago': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'lofi': [0, 5, 7, 5, 3, 5, 7, 5, 0, 5, 7, 5, 3, 5, 7, 5],
                'synthwave': [0, 3, 7, 10, 12, 10, 7, 3, 0, 3, 7, 10, 12, 10, 7, 3],
                'vaporwave': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'chillwave': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'future_bass': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'trap_soul': [0, 3, 5, 7, 5, 3, 0, 3, 0, 3, 5, 7, 5, 3, 0, 3],
                'rnb': [0, 2, 4, 7, 4, 2, 0, 2, 0, 2, 4, 7, 4, 2, 0, 2],
                'soul': [0, 4, 7, 4, 2, 4, 7, 4, 0, 4, 7, 4, 2, 4, 7, 4],
                'funk': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'disco': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'boogie': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'new_wave': [0, 3, 7, 10, 7, 3, 0, 3, 0, 3, 7, 10, 7, 3, 0, 3],
                'post_punk': [0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7],
                'indie': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'folk': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'country': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'blues': [0, 3, 5, 7, 5, 3, 0, 3, 0, 3, 5, 7, 5, 3, 0, 3],
                'rock': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'metal': [0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7],
                'punk': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'ska': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'reggae': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'dub_reggae': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'latin': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'salsa': [0, 3, 7, 10, 7, 3, 0, 3, 0, 3, 7, 10, 7, 3, 0, 3],
                'merengue': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'bachata': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'cumbia': [0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4, 0, 4, 7, 4],
                'flamenco': [0, 3, 7, 10, 7, 3, 0, 3, 0, 3, 7, 10, 7, 3, 0, 3],
                'tango': [0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5, 0, 5, 7, 5],
                'bossa': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
                'samba_arp': [0, 4, 7, 11, 7, 4, 0, 4, 0, 4, 7, 11, 7, 4, 0, 4],
            },
        }
        
        # 合成乐器音色参数（扩展预设）
        self.synth_voices = {}
        self.synth_presets = {
            'bass': {
                'default': {'waveform': 'saw', 'detune': 0.0, 'filter_cutoff': 0.3, 'filter_res': 0.2, 'attack': 0.01, 'decay': 0.2, 'sustain': 0.7, 'release': 0.3, 'octave': -1, 'glide': 0.02},
                'sub': {'waveform': 'sine', 'detune': 0.0, 'filter_cutoff': 0.15, 'filter_res': 0.1, 'attack': 0.005, 'decay': 0.15, 'sustain': 0.8, 'release': 0.2, 'octave': -2, 'glide': 0.01},
                'acid': {'waveform': 'saw', 'detune': 0.05, 'filter_cutoff': 0.7, 'filter_res': 0.5, 'attack': 0.001, 'decay': 0.4, 'sustain': 0.3, 'release': 0.2, 'octave': -1, 'glide': 0.08},
                ' Reese': {'waveform': 'saw', 'detune': 0.12, 'filter_cutoff': 0.25, 'filter_res': 0.15, 'attack': 0.02, 'decay': 0.25, 'sustain': 0.75, 'release': 0.35, 'octave': -1, 'glide': 0.03},
                'pluck': {'waveform': 'square', 'detune': 0.0, 'filter_cutoff': 0.5, 'filter_res': 0.2, 'attack': 0.001, 'decay': 0.3, 'sustain': 0.1, 'release': 0.15, 'octave': -1, 'glide': 0.01},
            },
            'lead': {
                'default': {'waveform': 'square', 'detune': 0.1, 'filter_cutoff': 0.6, 'filter_res': 0.3, 'attack': 0.01, 'decay': 0.1, 'sustain': 0.8, 'release': 0.2, 'octave': 0, 'glide': 0.05},
                'supersaw': {'waveform': 'saw', 'detune': 0.25, 'filter_cutoff': 0.55, 'filter_res': 0.25, 'attack': 0.02, 'decay': 0.15, 'sustain': 0.7, 'release': 0.25, 'octave': 0, 'glide': 0.04},
                'brass': {'waveform': 'saw', 'detune': 0.08, 'filter_cutoff': 0.45, 'filter_res': 0.35, 'attack': 0.08, 'decay': 0.2, 'sustain': 0.6, 'release': 0.15, 'octave': 0, 'glide': 0.02},
                'pluck': {'waveform': 'triangle', 'detune': 0.05, 'filter_cutoff': 0.7, 'filter_res': 0.2, 'attack': 0.001, 'decay': 0.25, 'sustain': 0.2, 'release': 0.1, 'octave': 1, 'glide': 0.01},
                'vocal': {'waveform': 'sine', 'detune': 0.15, 'filter_cutoff': 0.4, 'filter_res': 0.2, 'attack': 0.05, 'decay': 0.1, 'sustain': 0.85, 'release': 0.3, 'octave': 0, 'glide': 0.06},
            },
            'pad': {
                'default': {'waveform': 'sine', 'detune': 0.15, 'filter_cutoff': 0.4, 'filter_res': 0.1, 'attack': 0.5, 'decay': 0.3, 'sustain': 0.9, 'release': 1.0, 'octave': 0, 'glide': 0.1},
                'string': {'waveform': 'saw', 'detune': 0.1, 'filter_cutoff': 0.35, 'filter_res': 0.15, 'attack': 0.3, 'decay': 0.2, 'sustain': 0.85, 'release': 0.8, 'octave': 0, 'glide': 0.08},
                'choir': {'waveform': 'sine', 'detune': 0.2, 'filter_cutoff': 0.3, 'filter_res': 0.1, 'attack': 0.4, 'decay': 0.25, 'sustain': 0.8, 'release': 0.9, 'octave': 0, 'glide': 0.12},
                'ambient': {'waveform': 'triangle', 'detune': 0.18, 'filter_cutoff': 0.25, 'filter_res': 0.08, 'attack': 0.8, 'decay': 0.4, 'sustain': 0.9, 'release': 1.5, 'octave': 0, 'glide': 0.15},
                'warm': {'waveform': 'sine', 'detune': 0.08, 'filter_cutoff': 0.2, 'filter_res': 0.05, 'attack': 0.6, 'decay': 0.3, 'sustain': 0.95, 'release': 1.2, 'octave': -1, 'glide': 0.1},
            },
            'keys': {
                'default': {'waveform': 'triangle', 'detune': 0.05, 'filter_cutoff': 0.5, 'filter_res': 0.15, 'attack': 0.005, 'decay': 0.3, 'sustain': 0.5, 'release': 0.4, 'octave': 0, 'glide': 0.03},
                'piano': {'waveform': 'sine', 'detune': 0.02, 'filter_cutoff': 0.6, 'filter_res': 0.1, 'attack': 0.002, 'decay': 0.5, 'sustain': 0.3, 'release': 0.5, 'octave': 0, 'glide': 0.01},
                'epiano': {'waveform': 'sine', 'detune': 0.1, 'filter_cutoff': 0.45, 'filter_res': 0.2, 'attack': 0.01, 'decay': 0.4, 'sustain': 0.4, 'release': 0.4, 'octave': 0, 'glide': 0.02},
                'organ': {'waveform': 'sine', 'detune': 0.0, 'filter_cutoff': 0.7, 'filter_res': 0.1, 'attack': 0.01, 'decay': 0.05, 'sustain': 0.95, 'release': 0.1, 'octave': 0, 'glide': 0.0},
                'clav': {'waveform': 'square', 'detune': 0.03, 'filter_cutoff': 0.65, 'filter_res': 0.25, 'attack': 0.001, 'decay': 0.2, 'sustain': 0.3, 'release': 0.15, 'octave': 0, 'glide': 0.01},
            },
            'arp': {
                'default': {'waveform': 'saw', 'detune': 0.08, 'filter_cutoff': 0.7, 'filter_res': 0.25, 'attack': 0.001, 'decay': 0.15, 'sustain': 0.3, 'release': 0.2, 'octave': 1, 'glide': 0.01},
                'digital': {'waveform': 'square', 'detune': 0.05, 'filter_cutoff': 0.75, 'filter_res': 0.2, 'attack': 0.001, 'decay': 0.1, 'sustain': 0.2, 'release': 0.1, 'octave': 1, 'glide': 0.005},
                'glass': {'waveform': 'sine', 'detune': 0.12, 'filter_cutoff': 0.6, 'filter_res': 0.15, 'attack': 0.01, 'decay': 0.2, 'sustain': 0.4, 'release': 0.3, 'octave': 2, 'glide': 0.02},
                'pulse': {'waveform': 'pulse', 'detune': 0.06, 'filter_cutoff': 0.65, 'filter_res': 0.3, 'attack': 0.001, 'decay': 0.12, 'sustain': 0.25, 'release': 0.15, 'octave': 1, 'glide': 0.008},
            },
        }
        
        self.synth_params = {ch: self.synth_presets[ch]['default'].copy() for ch in self.synth_presets}
        
        # 和弦进行系统
        self.chord_progressions = {
            'classic': [[0, 4, 7], [5, 9, 12], [3, 7, 10], [4, 8, 11]],
            'pop': [[0, 4, 7], [5, 9, 12], [3, 7, 10], [4, 8, 11]],
            'jazz': [[0, 4, 7, 11], [5, 9, 12, 16], [7, 11, 14, 17], [3, 7, 10, 14]],
            'minor': [[0, 3, 7], [5, 8, 12], [3, 7, 10], [7, 10, 14]],
            'house': [[0, 4, 7], [0, 4, 7], [5, 9, 12], [5, 9, 12]],
            'trance': [[0, 4, 7, 11], [5, 9, 12, 16], [3, 7, 10, 14], [0, 4, 7, 11]],
            'ambient': [[0, 5, 7], [0, 5, 7], [0, 5, 7], [0, 5, 7]],
            'neo_soul': [[0, 4, 7, 11], [5, 9, 12, 15], [7, 11, 14, 17], [3, 7, 10, 14]],
            'lofi': [[0, 3, 7], [5, 8, 12], [7, 10, 14], [3, 7, 10]],
        }
        self.current_chord_progression = 'classic'
        self.chord_index = 0
        self.current_chord = [0, 4, 7]
        
        # 合成器状态
        self.synth_envelopes = {ch: 0.0 for ch in self.channels if ch != 'drums'}
        self.synth_phases = {ch: 0.0 for ch in self.channels if ch != 'drums'}
        self.synth_filter_states = {ch: 0.0 for ch in self.channels if ch != 'drums'}
        self.current_notes = {ch: 48 for ch in self.channels if ch != 'drums'}
        self.target_notes = {ch: 48 for ch in self.channels if ch != 'drums'}
        
        # 预生成合成器波形表
        self._generate_wavetables()
    
    def _init_midi(self):
        """初始化MIDI输出"""
        try:
            import mido
            self.mido = mido
            
            outputs = mido.get_output_names()
            if outputs:
                self.midi_port = mido.open_output(outputs[0])
                self.midi_enabled = True
                print(f"MIDI已连接: {outputs[0]}")
            else:
                print("未找到MIDI输出设备")
        except ImportError:
            print("mido库未安装，MIDI功能禁用。安装: pip install mido python-rtmidi")
            self.mido = None
        except Exception as e:
            print(f"MIDI初始化失败: {e}")
            self.mido = None
    
    def _generate_wavetables(self):
        """预生成合成器波形表"""
        table_size = 4096
        t = np.arange(table_size) / table_size
        
        self.wavetables = {
            'sine': np.sin(2 * np.pi * t).astype(np.float32),
            'square': np.where(t < 0.5, 1.0, -1.0).astype(np.float32),
            'saw': (2 * t - 1).astype(np.float32),
            'triangle': (4 * np.abs(t - 0.5) - 1).astype(np.float32),
            'pulse': np.where(t < 0.25, 1.0, -1.0).astype(np.float32),
            'noise': np.random.uniform(-1, 1, table_size).astype(np.float32),
        }
        
        # 复合波形
        self.wavetables['bass_rich'] = (
            self.wavetables['saw'] * 0.5 + 
            self.wavetables['square'] * 0.3 + 
            self.wavetables['sine'] * 0.2
        ).astype(np.float32)
        
        self.wavetables['lead_rich'] = (
            self.wavetables['saw'] * 0.4 + 
            self.wavetables['square'] * 0.4 + 
            np.sin(2 * np.pi * t * 2) * 0.2
        ).astype(np.float32)
        
        self.wavetables['pad_soft'] = (
            self.wavetables['sine'] * 0.6 + 
            self.wavetables['triangle'] * 0.4
        ).astype(np.float32)
    
    def _generate_sample_library(self):
        """生成高质量音频样本库"""
        duration = 2.0
        samples = int(self.sample_rate * duration)
        
        for drum_type in ['kick', 'snare', 'clap', 'hihat_closed', 'hihat_open', 
                          'tom_high', 'tom_mid', 'tom_low', 'rimshot', 'cowbell',
                          'clave', 'maracas', 'conga_high', 'conga_low', 'cymbal']:
            self.sample_library[drum_type] = self._synthesize_drum_sample(drum_type, samples)
    
    def _synthesize_drum_sample(self, drum_type, samples):
        """合成单个鼓音样本"""
        sample = np.zeros(samples, dtype=np.float32)
        t = np.arange(samples) / self.sample_rate
        
        if drum_type == 'kick':
            freq_sweep = 150 * np.exp(-t * 15) + 40
            phase = np.cumsum(2 * np.pi * freq_sweep / self.sample_rate)
            sample = np.sin(phase) * np.exp(-t * 4)
            sample += np.sin(phase * 0.5) * np.exp(-t * 3) * 0.4
            click = np.exp(-t * 100) * np.sin(2 * np.pi * 1000 * t) * 0.1
            sample += click
            
        elif drum_type == 'snare':
            noise = np.random.uniform(-1, 1, samples)
            bpf_noise = np.zeros(samples)
            state = 0
            for i in range(samples):
                state = 0.3 * state + 0.7 * noise[i]
                bpf_noise[i] = state
            tone = np.sin(2 * np.pi * 180 * t) * np.exp(-t * 10)
            sample = tone * 0.4 + bpf_noise * 0.6 * np.exp(-t * 8)
            
        elif drum_type == 'clap':
            noise = np.random.uniform(-1, 1, samples)
            bandpass = np.zeros(samples)
            state = 0
            for i in range(samples):
                state = 0.25 * state + 0.75 * noise[i]
                bandpass[i] = state
            env = np.exp(-t * 8)
            for i in range(3):
                env += np.exp(-(t - 0.01 * i) * 20) * 0.3 * (t > 0.01 * i)
            sample = bandpass * env * 0.8
            
        elif drum_type == 'hihat_closed':
            noise = np.random.uniform(-1, 1, samples)
            square_sum = np.zeros(samples)
            for f_mult in [8, 10, 12, 14, 16]:
                freq = 8000 * f_mult / 10
                square_sum += np.sign(np.sin(2 * np.pi * freq * t)) / f_mult
            hpf = np.zeros(samples)
            state = 0
            for i in range(samples):
                state = 0.1 * state + 0.9 * noise[i]
                hpf[i] = noise[i] - state
            sample = (square_sum * 0.4 + hpf * 0.6) * np.exp(-t * 30)
            
        elif drum_type == 'hihat_open':
            noise = np.random.uniform(-1, 1, samples)
            square_sum = np.zeros(samples)
            for f_mult in [8, 10, 12, 14, 16]:
                freq = 7000 * f_mult / 10
                square_sum += np.sign(np.sin(2 * np.pi * freq * t)) / f_mult
            sample = square_sum * np.exp(-t * 5) * 0.6
            
        elif drum_type.startswith('tom'):
            base_freqs = {'tom_high': 220, 'tom_mid': 150, 'tom_low': 100}
            base_freq = base_freqs.get(drum_type, 150)
            freq_sweep = base_freq * np.exp(-t * 3)
            phase = np.cumsum(2 * np.pi * freq_sweep / self.sample_rate)
            sample = np.sin(phase) * np.exp(-t * 6)
            sample += np.sin(phase * 2) * np.exp(-t * 8) * 0.3
            
        elif drum_type == 'rimshot':
            tone1 = np.sin(2 * np.pi * 750 * t) * np.exp(-t * 50)
            tone2 = np.sin(2 * np.pi * 1100 * t) * np.exp(-t * 60) * 0.5
            sample = tone1 * 0.6 + tone2 * 0.4
            
        elif drum_type == 'cowbell':
            freq1, freq2 = 560, 845
            square1 = np.sign(np.sin(2 * np.pi * freq1 * t))
            square2 = np.sign(np.sin(2 * np.pi * freq2 * t))
            sample = (square1 * 0.5 + square2 * 0.5) * np.exp(-t * 6)
            
        elif drum_type == 'clave':
            freq = 2500
            sample = np.sin(2 * np.pi * freq * t) * np.exp(-t * 40)
            
        elif drum_type == 'maracas':
            noise = np.random.uniform(-1, 1, samples)
            hpf = np.zeros(samples)
            state = 0
            for i in range(samples):
                state = 0.15 * state + 0.85 * noise[i]
                hpf[i] = (noise[i] - state) * 1.5
            sample = hpf * np.exp(-t * 15)
            
        elif drum_type.startswith('conga'):
            base_freqs = {'conga_high': 300, 'conga_low': 140}
            base_freq = base_freqs.get(drum_type, 200)
            freq_sweep = base_freq * np.exp(-t * 2)
            phase = np.cumsum(2 * np.pi * freq_sweep / self.sample_rate)
            sample = np.sin(phase) * np.exp(-t * 5)
            noise = np.random.uniform(-1, 1, samples) * np.exp(-t * 30) * 0.15
            sample += noise
            
        elif drum_type == 'cymbal':
            noise = np.random.uniform(-1, 1, samples)
            shimmer = np.sin(2 * np.pi * 6000 * t) * 0.3 + np.sin(2 * np.pi * 8000 * t) * 0.2
            sample = (noise * 0.6 + shimmer * 0.4) * np.exp(-t * 1.5)
        
        sample = np.clip(sample, -1, 1)
        peak = np.max(np.abs(sample))
        if peak > 0:
            sample = sample / peak * 0.9
        
        return sample
    
    def _load_external_samples(self, path):
        """加载外部音频样本文件"""
        try:
            import scipy.io.wavfile as wav
            if os.path.exists(path):
                for drum_type in self.drum_kits.keys():
                    file_path = os.path.join(path, f"{drum_type}.wav")
                    if os.path.exists(file_path):
                        rate, data = wav.read(file_path)
                        if rate != self.sample_rate:
                            import scipy.signal
                            data = scipy.signal.resample(data, int(len(data) * self.sample_rate / rate))
                        if len(data.shape) > 1:
                            data = data.mean(axis=1)
                        self.sample_library[drum_type] = data.astype(np.float32) / 32768.0
                print(f"已加载外部样本: {path}")
        except Exception as e:
            print(f"加载外部样本失败: {e}")
    
    def _smooth_parameter(self, current, target, speed=0.02):
        return current + (target - current) * speed
    
    def _update_smooth_parameters(self):
        self.tempo = self._smooth_parameter(self.tempo, self.tempo_target, self.tempo_smoothing)
        self.swing = self._smooth_parameter(self.swing, self.swing_target, 0.03)
        self.master_volume = self._smooth_parameter(self.master_volume, self.master_volume_target, 0.02)
        for k in self.drum_volumes:
            self.drum_volumes[k] = self._smooth_parameter(self.drum_volumes[k], self.drum_volumes_target[k], 0.02)
        self.reverb_amount = self._smooth_parameter(self.reverb_amount, self.reverb_amount_target, 0.02)
        
        for drum, params in self.drum_params.items():
            if 'pitch_target' in params:
                params['pitch'] = self._smooth_parameter(params['pitch'], params['pitch_target'], 0.02)
            if 'decay_target' in params:
                params['decay'] = self._smooth_parameter(params['decay'], params['decay_target'], 0.02)
            if 'tone_target' in params:
                params['tone'] = self._smooth_parameter(params['tone'], params['tone_target'], 0.02)
            if 'snappy_target' in params:
                params['snappy'] = self._smooth_parameter(params['snappy'], params['snappy_target'], 0.02)
        
        for style in self.style_weights:
            target = 1.0 if style == self.style_target else 0.0
            self.style_weights[style] = self._smooth_parameter(self.style_weights[style], target, self.style_transition_speed)
    
    def _blend_patterns(self):
        blended = {k: [0.0]*16 for k in self.drum_kits.keys()}
        for style, weight in self.style_weights.items():
            if weight > 0.01 and style in self.patterns:
                pattern = self.patterns[style]
                for drum, steps in pattern.items():
                    for i in range(min(len(steps), 16)):
                        blended[drum][i] += steps[i] * weight
        if self.variation_intensity > 0:
            for drum in blended:
                for i in range(16):
                    if np.random.random() < self.variation_intensity * 0.1:
                        blended[drum][i] = min(1.0, blended[drum][i] + np.random.uniform(0, 0.3))
        return blended
    
    def _generate_variation(self):
        self.pattern_variation_counter += 1
        if self.pattern_variation_counter % 4 == 0:
            extra_drums = ['tom_high', 'tom_mid', 'tom_low', 'rimshot', 'cowbell', 'clave', 'maracas', 'conga_high', 'conga_low']
            for drum in extra_drums:
                if np.random.random() < 0.25 * self.variation_intensity:
                    pos = np.random.randint(0, 16)
                    self.generated_pattern[drum][pos] = np.random.uniform(0.5, 1.0)
        if self.pattern_variation_counter % 8 == 0:
            if np.random.random() < 0.3:
                pos = np.random.randint(0, 16)
                self.generated_pattern['cymbal'][pos] = 0.7
        if self.pattern_variation_counter % 16 == 0:
            for drum in ['tom_high', 'tom_mid', 'tom_low', 'rimshot', 'cowbell', 'clave', 'maracas', 'conga_high', 'conga_low', 'cymbal']:
                self.generated_pattern[drum] = [0]*16
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
        if self.midi_port:
            try:
                self.midi_port.close()
            except:
                pass
    
    def _audio_loop(self):
        try:
            import sounddevice as sd
            with sd.OutputStream(samplerate=self.sample_rate,
                                channels=1,
                                callback=self._audio_callback,
                                blocksize=self.buffer_size,
                                dtype=np.float32):
                while self.running:
                    time.sleep(0.005)
        except ImportError:
            print("sounddevice未安装，音频禁用")
            self.running = False
    
    def _trigger_drum(self, drum_type, velocity=1.0):
        """触发鼓音（样本播放 + MIDI输出）"""
        if drum_type in self.sample_library:
            if drum_type not in self.sample_voices:
                self.sample_voices[drum_type] = []
            
            sample = self.sample_library[drum_type].copy()
            params = self.drum_params.get(drum_type, {})
            
            pitch = params.get('pitch', 1.0)
            if pitch != 1.0:
                new_len = int(len(sample) / pitch)
                indices = np.linspace(0, len(sample) - 1, new_len)
                sample = np.interp(indices, np.arange(len(sample)), sample)
            
            decay = params.get('decay', 0.5)
            decay_env = np.exp(-np.arange(len(sample)) / self.sample_rate * (10 - decay * 15))
            sample = sample * decay_env[:len(sample)]
            
            voice = {
                'sample': sample * velocity * self.drum_volumes.get(drum_type, 0.8),
                'position': 0,
                'active': True
            }
            self.sample_voices[drum_type].append(voice)
            
            if len(self.sample_voices[drum_type]) > self.max_voices:
                self.sample_voices[drum_type] = self.sample_voices[drum_type][-self.max_voices:]
        
        if self.midi_enabled and self.midi_port and drum_type in self.drum_kits:
            midi_note = self.drum_kits[drum_type].get('midi_note', 36)
            try:
                note_on = self.mido.Message('note_on', note=midi_note, velocity=int(velocity * 127), channel=9)
                self.midi_port.send(note_on)
                threading.Timer(0.1, self._send_midi_note_off, args=[midi_note]).start()
            except Exception as e:
                pass
    
    def _send_midi_note_off(self, note):
        """发送MIDI音符关闭消息"""
        if self.midi_enabled and self.midi_port:
            try:
                note_off = self.mido.Message('note_off', note=note, velocity=0, channel=9)
                self.midi_port.send(note_off)
            except:
                pass
    
    def _note_to_freq(self, note):
        """MIDI音符转频率"""
        return 440.0 * (2.0 ** ((note - 69) / 12.0))
    
    def _get_scale_note(self, interval, octave_offset=0):
        """获取音阶内的音符"""
        scale = self.scales.get(self.current_scale, self.scales['minor'])
        scale_degree = interval % len(scale)
        octave = interval // len(scale)
        return self.root_note + scale[scale_degree] + octave * 12 + octave_offset * 12
    
    def _trigger_synth(self, channel, note, velocity=1.0):
        """触发合成器音符"""
        if channel not in self.synth_params:
            return
        
        self.target_notes[channel] = note
        self.synth_envelopes[channel] = velocity
        
        if self.midi_enabled and self.midi_port:
            try:
                ch_info = self.channels.get(channel, {})
                midi_ch = ch_info.get('midi_channel', 1) - 1
                prev_note = self.current_notes.get(channel, note)
                note_off = self.mido.Message('note_off', note=int(prev_note), velocity=0, channel=midi_ch)
                self.midi_port.send(note_off)
                note_on = self.mido.Message('note_on', note=int(note), velocity=int(velocity * 127), channel=midi_ch)
                self.midi_port.send(note_on)
            except:
                pass
    
    def _generate_synth_voice(self, channel, frames):
        """生成合成器声音"""
        if channel not in self.synth_params or not self.channels[channel]['enabled']:
            return np.zeros(frames, dtype=np.float32)
        
        params = self.synth_params[channel]
        output = np.zeros(frames, dtype=np.float32)
        
        envelope = self.synth_envelopes[channel]
        if envelope < 0.001:
            return output
        
        waveform = params.get('waveform', 'saw')
        wavetable = self.wavetables.get(waveform, self.wavetables['saw'])
        table_size = len(wavetable)
        
        detune = params.get('detune', 0.0)
        filter_cutoff = params.get('filter_cutoff', 0.5) * self.filter_cutoff
        filter_res = params.get('filter_res', 0.2)
        attack = params.get('attack', 0.01)
        decay = params.get('decay', 0.1)
        sustain = params.get('sustain', 0.7)
        release = params.get('release', 0.2)
        glide = params.get('glide', 0.05)
        octave = params.get('octave', 0)
        
        current = self.current_notes[channel]
        target = self.target_notes[channel]
        
        for i in range(frames):
            current = current + (target - current) * glide
            self.current_notes[channel] = current
            
            freq = self._note_to_freq(current + octave * 12)
            phase_inc = freq * table_size / self.sample_rate
            
            phase = self.synth_phases[channel]
            idx = int(phase) % table_size
            idx2 = (idx + 1) % table_size
            frac = phase - int(phase)
            sample = wavetable[idx] * (1 - frac) + wavetable[idx2] * frac
            
            if detune > 0:
                phase2 = phase * (1 + detune * 0.01)
                idx2 = int(phase2) % table_size
                idx2b = (idx2 + 1) % table_size
                frac2 = phase2 - int(phase2)
                sample2 = wavetable[idx2] * (1 - frac2) + wavetable[idx2b] * frac2
                sample = sample * 0.5 + sample2 * 0.5
            
            filter_state = self.synth_filter_states[channel]
            cutoff = 0.1 + filter_cutoff * 0.8
            filter_state = filter_state + cutoff * (sample - filter_state)
            self.synth_filter_states[channel] = filter_state
            
            output[i] = filter_state * envelope
            
            self.synth_phases[channel] = (phase + phase_inc) % table_size
        
        ch_info = self.channels[channel]
        volume = ch_info.get('volume', 0.5)
        pan = ch_info.get('pan', 0.0)
        
        output = output * volume
        
        sub_harmonic = np.zeros(frames, dtype=np.float32)
        for i in range(frames):
            sub_phase = self.synth_phases[channel] * 0.5
            sub_idx = int(sub_phase) % table_size
            sub_idx2 = (sub_idx + 1) % table_size
            sub_frac = sub_phase - int(sub_phase)
            sub_harmonic[i] = (wavetable[sub_idx] * (1 - sub_frac) + wavetable[sub_idx2] * sub_frac) * 0.3 * envelope
        
        output = output + sub_harmonic * volume
        
        mid_boost = np.zeros(frames, dtype=np.float32)
        mid_state = getattr(self, f'_{channel}_mid_state', 0.0)
        for i in range(frames):
            mid_state = 0.85 * mid_state + 0.15 * output[i]
            mid_boost[i] = mid_state * 0.25
        setattr(self, f'_{channel}_mid_state', mid_state)
        
        output = output + mid_boost
        
        decay_rate = 0.9995 - release * 0.001
        self.synth_envelopes[channel] *= decay_rate ** frames
        
        return output
    
    def _apply_chorus(self, input_signal, frames):
        """应用合唱效果（多声部增强版）"""
        if not self.effects['chorus']['enabled']:
            return input_signal
        
        chorus = self.effects['chorus']
        rate = chorus['rate']
        depth = chorus['depth']
        mix = chorus['mix']
        voices = chorus.get('voices', 3)
        
        output = input_signal.copy()
        chorus_out = np.zeros(frames, dtype=np.float32)
        
        for voice in range(voices):
            voice_phase = chorus.get(f'voice_phase_{voice}', voice * 2.0)
            voice_rate_offset = voice * 0.1
            
            for i in range(frames):
                mod = 1 + depth * 0.005 * np.sin(voice_phase + voice * 1.5)
                delay_samples = int(len(self._chorus_buffer) * mod * 0.6)
                delay_samples = min(delay_samples, len(self._chorus_buffer) - 1)
                
                read_idx = (i - delay_samples) % len(self._chorus_buffer)
                chorus_out[i] += self._chorus_buffer[read_idx] / voices
                
                if i == 0:
                    self._chorus_buffer[i % len(self._chorus_buffer)] = input_signal[i]
                else:
                    self._chorus_buffer[i % len(self._chorus_buffer)] = input_signal[i]
                
                voice_phase += (rate + voice_rate_offset) * 2 * np.pi / self.sample_rate
            
            chorus[f'voice_phase_{voice}'] = voice_phase
        
        return output * (1 - mix) + chorus_out * mix
    
    def _apply_flanger(self, input_signal, frames):
        """应用镶边效果"""
        if not self.effects['flanger']['enabled']:
            return input_signal
        
        flanger = self.effects['flanger']
        rate = flanger['rate']
        depth = flanger['depth']
        feedback = flanger['feedback']
        mix = flanger['mix']
        
        output = input_signal.copy()
        flanger_out = np.zeros(frames, dtype=np.float32)
        
        for i in range(frames):
            mod = depth * 0.001 * (1 + np.sin(flanger['phase']))
            delay_samples = int(len(self._flanger_buffer) * mod)
            delay_samples = max(1, min(delay_samples, len(self._flanger_buffer) - 1))
            
            read_idx = (i - delay_samples) % len(self._flanger_buffer)
            flanger_out[i] = self._flanger_buffer[read_idx]
            
            self._flanger_buffer[i % len(self._flanger_buffer)] = input_signal[i] + flanger_out[i] * feedback
            flanger['phase'] += rate * 2 * np.pi / self.sample_rate
        
        return output * (1 - mix) + flanger_out * mix
    
    def _apply_phaser(self, input_signal, frames):
        """应用移相效果"""
        if not self.effects['phaser']['enabled']:
            return input_signal
        
        phaser = self.effects['phaser']
        rate = phaser['rate']
        depth = phaser['depth']
        stages = phaser['stages']
        mix = phaser['mix']
        
        output = input_signal.copy()
        phaser_out = np.zeros(frames, dtype=np.float32)
        
        for i in range(frames):
            mod = 0.5 + 0.5 * np.sin(phaser['phase'])
            sample = input_signal[i]
            
            for s in range(stages):
                coef = 0.2 + mod * 0.6
                state_idx = s * 2
                self._phaser_states[state_idx] = coef * sample + (1 - coef) * self._phaser_states[state_idx]
                sample = self._phaser_states[state_idx]
            
            phaser_out[i] = sample
            phaser['phase'] += rate * 2 * np.pi / self.sample_rate
        
        return output * (1 - mix) + phaser_out * mix
    
    def _apply_distortion(self, input_signal, frames):
        """应用失真效果"""
        if not self.effects['distortion']['enabled']:
            return input_signal
        
        dist = self.effects['distortion']
        drive = dist['drive']
        mix = dist['mix']
        dist_type = dist['type']
        
        if dist_type == 'soft':
            distorted = np.tanh(input_signal * (1 + drive * 3))
        elif dist_type == 'hard':
            distorted = np.clip(input_signal * (1 + drive * 5), -1, 1)
        elif dist_type == 'fuzz':
            sign = np.sign(input_signal)
            distorted = sign * (1 - np.exp(-np.abs(input_signal) * (1 + drive * 4)))
        else:
            distorted = np.tanh(input_signal * (1 + drive * 3))
        
        return input_signal * (1 - mix) + distorted * mix
    
    def _apply_compressor(self, input_signal, frames):
        """应用压缩器"""
        if not self.effects['compressor']['enabled']:
            return input_signal
        
        comp = self.effects['compressor']
        threshold = comp['threshold']
        ratio = comp['ratio']
        attack = comp['attack']
        release = comp['release']
        gain = comp['gain']
        
        output = input_signal.copy()
        
        for i in range(frames):
            abs_val = np.abs(input_signal[i])
            
            if abs_val > self._compressor_envelope:
                self._compressor_envelope += (abs_val - self._compressor_envelope) * attack
            else:
                self._compressor_envelope += (abs_val - self._compressor_envelope) * release
            
            if self._compressor_envelope > threshold:
                reduction = (self._compressor_envelope - threshold) * (1 - 1/ratio)
                output[i] = input_signal[i] * (self._compressor_envelope - reduction) / max(self._compressor_envelope, 0.001)
        
        return output * gain
    
    def _apply_limiter(self, input_signal, frames):
        """应用限幅器"""
        if not self.effects['limiter']['enabled']:
            return input_signal
        
        lim = self.effects['limiter']
        threshold = lim['threshold']
        release = lim['release']
        
        output = input_signal.copy()
        
        for i in range(frames):
            abs_val = np.abs(input_signal[i])
            
            if abs_val > self._limiter_envelope:
                self._limiter_envelope = abs_val
            else:
                self._limiter_envelope += (abs_val - self._limiter_envelope) * release
            
            if self._limiter_envelope > threshold:
                output[i] = input_signal[i] * threshold / max(self._limiter_envelope, 0.001)
        
        return output
    
    def _apply_filter(self, input_signal, frames):
        """应用滤波器"""
        if not self.effects['filter']['enabled']:
            return input_signal
        
        filt = self.effects['filter']
        cutoff = filt['cutoff']
        resonance = filt['resonance']
        filter_type = filt['type']
        
        output = input_signal.copy()
        state = filt['state']
        
        if filter_type == 'lowpass':
            for i in range(frames):
                state[0] = state[0] + cutoff * 0.5 * (input_signal[i] - state[0])
                state[1] = state[1] + cutoff * 0.5 * (state[0] - state[1])
                state[2] = state[2] + cutoff * 0.5 * (state[1] - state[2])
                state[3] = state[3] + cutoff * 0.5 * (state[2] - state[3])
                output[i] = state[3]
        elif filter_type == 'highpass':
            for i in range(frames):
                state[0] = input_signal[i] - state[0] * cutoff * 0.5
                state[1] = state[0] - state[1] * cutoff * 0.5
                state[2] = state[1] - state[2] * cutoff * 0.5
                state[3] = state[2] - state[3] * cutoff * 0.5
                output[i] = input_signal[i] - state[3]
        elif filter_type == 'bandpass':
            for i in range(frames):
                state[0] = state[0] + cutoff * 0.3 * (input_signal[i] - state[0])
                state[1] = state[1] + cutoff * 0.3 * (state[0] - state[1])
                output[i] = state[0] - state[1]
        
        filt['state'] = state
        return output
    
    def _apply_eq(self, input_signal, frames):
        """应用均衡器"""
        if not self.effects['eq']['enabled']:
            return input_signal
        
        eq = self.effects['eq']
        states = eq['states']
        
        output = input_signal.copy()
        
        low_freq_norm = eq['low_freq'] / self.sample_rate
        mid_freq_norm = eq['mid_freq'] / self.sample_rate
        high_freq_norm = eq['high_freq'] / self.sample_rate
        
        for i in range(frames):
            states[0] = states[0] + low_freq_norm * (input_signal[i] - states[0])
            states[1] = states[1] + low_freq_norm * (states[0] - states[1])
            low = states[1]
            
            states[2] = states[2] + mid_freq_norm * (input_signal[i] - states[2])
            states[3] = states[3] + mid_freq_norm * (states[2] - states[3])
            mid = input_signal[i] - states[2] - (input_signal[i] - states[3])
            
            states[4] = states[4] + high_freq_norm * (input_signal[i] - states[4])
            states[5] = states[5] + high_freq_norm * (states[4] - states[5])
            high = input_signal[i] - states[4]
            
            output[i] = low * eq['low_gain'] + mid * eq['mid_gain'] + high * eq['high_gain']
        
        eq['states'] = states
        return output
    
    def _apply_harmonizer(self, input_signal, frames):
        """应用合声效果器（和声生成）"""
        if not self.effects['harmonizer']['enabled']:
            return input_signal
        
        harm = self.effects['harmonizer']
        semitones = harm['semitones']
        mix = harm['mix']
        
        output = input_signal.copy()
        harmonized = np.zeros(frames, dtype=np.float32)
        
        for idx, semitone in enumerate(semitones):
            pitch_ratio = 2.0 ** (semitone / 12.0)
            
            phase_offset = harm['phase_offsets'][idx] if idx < len(harm['phase_offsets']) else 0.0
            
            for i in range(frames):
                read_pos = i / pitch_ratio + phase_offset * self.sample_rate
                read_pos_int = int(read_pos) % frames
                read_pos_frac = read_pos - int(read_pos)
                
                next_pos = (read_pos_int + 1) % frames
                
                sample = input_signal[read_pos_int] * (1 - read_pos_frac) + input_signal[next_pos] * read_pos_frac
                harmonized[i] += sample * 0.5
        
        if len(semitones) > 0:
            harmonized /= len(semitones)
        
        return output * (1 - mix) + harmonized * mix
    
    def _apply_all_effects(self, input_signal, frames):
        """应用效果器（仅合唱和限幅器）"""
        output = input_signal.copy()
        
        output = self._apply_chorus(output, frames)
        output = self._apply_limiter(output, frames)
        
        return output
    
    def _send_midi_clock(self):
        """发送MIDI时钟信号"""
        if self.midi_enabled and self.midi_port:
            try:
                clock = self.mido.Message('clock')
                self.midi_port.send(clock)
            except:
                pass
    
    def _audio_callback(self, outdata, frames, time_info, status):
        output = np.zeros(frames, dtype=np.float32)
        
        self._update_smooth_parameters()
        self._generate_variation()
        
        step_duration = 60.0 / self.tempo / 4
        step_inc = self.buffer_size / self.sample_rate / step_duration
        self.step_phase += step_inc
        
        if self.step_phase >= 1.0:
            self.step_phase -= 1.0
            
            self._send_midi_clock()
            
            blended_pattern = self._blend_patterns()
            
            for drum in self.generated_pattern:
                for i in range(16):
                    if self.generated_pattern[drum][i] > 0:
                        blended_pattern[drum][i] = max(blended_pattern[drum][i], self.generated_pattern[drum][i])
            
            for drum_type, steps in blended_pattern.items():
                if self.current_step < len(steps) and steps[self.current_step] > 0.3:
                    self._trigger_drum(drum_type, steps[self.current_step])
            
            if self.current_step % 4 == 0:
                self.chord_index = (self.current_step // 4) % 4
                progression = self.chord_progressions.get(self.current_chord_progression, self.chord_progressions['classic'])
                if self.chord_index < len(progression):
                    self.current_chord = progression[self.chord_index]
            
            for channel in ['bass', 'lead', 'pad', 'keys', 'arp']:
                if self.channels[channel]['enabled']:
                    pattern_set = self.melody_patterns.get(channel, {})
                    pattern = pattern_set.get(self.current_pattern, pattern_set.get('classic', [0]*16))
                    if self.current_step < len(pattern):
                        interval = pattern[self.current_step]
                        if interval != 0 or channel == 'bass':
                            chord_note = self.current_chord[abs(interval) % len(self.current_chord)] if interval != 0 else 0
                            note = self._get_scale_note(chord_note, self.synth_params[channel].get('octave', 0))
                            if interval < 0:
                                note -= 12
                            velocity = 0.6 + np.random.uniform(0, 0.4) * self.variation_intensity
                            self._trigger_synth(channel, note, velocity)
            
            self.current_step = (self.current_step + 1) % self.step_count
        
        for drum_type in list(self.sample_voices.keys()):
            voices = self.sample_voices[drum_type]
            active_voices = []
            for voice in voices:
                if voice['active'] and voice['position'] < len(voice['sample']):
                    remaining = len(voice['sample']) - voice['position']
                    to_read = min(remaining, frames)
                    
                    output[:to_read] += voice['sample'][voice['position']:voice['position'] + to_read]
                    voice['position'] += to_read
                    
                    if voice['position'] < len(voice['sample']):
                        active_voices.append(voice)
            self.sample_voices[drum_type] = active_voices
        
        for channel in ['bass', 'lead', 'pad', 'keys', 'arp']:
            synth_out = self._generate_synth_voice(channel, frames)
            output += synth_out * 0.7
        
        output = self._apply_all_effects(output, frames)
        
        output = output * self.master_volume * 1.5
        output = np.tanh(output * 0.85) * 0.95
        
        output = np.clip(output, -0.95, 0.95).astype(np.float32)
        outdata[:, 0] = output
    
    def update_from_gesture(self, hand_y, hand_x, hand_area, finger_count):
        self.hand_y = hand_y
        self.hand_x = hand_x
        self.hand_area = hand_area
        self.finger_count = finger_count
        
        y_curved = hand_y ** 1.3
        self.drum_params['kick']['pitch_target'] = 0.6 + y_curved * 0.7
        self.drum_params['snare']['snappy_target'] = 0.2 + y_curved * 0.8
        self.drum_params['kick']['tone_target'] = 0.25 + y_curved * 0.55
        self.drum_params['hihat_closed']['tone_target'] = 0.5 + y_curved * 0.4
        
        self.synth_params['bass']['filter_cutoff'] = 0.2 + y_curved * 0.4
        self.synth_params['lead']['filter_cutoff'] = 0.4 + y_curved * 0.5
        self.synth_params['arp']['filter_cutoff'] = 0.5 + y_curved * 0.4
        
        self.channels['bass']['volume'] = 0.5 + (1 - y_curved) * 0.4
        self.channels['lead']['volume'] = 0.4 + y_curved * 0.4
        self.channels['pad']['volume'] = 0.2 + (1 - abs(y_curved - 0.5) * 2) * 0.4
        self.channels['arp']['volume'] = 0.3 + y_curved * 0.4
        
        area_normalized = min(1.0, hand_area / 50000)
        area_curved = np.log1p(area_normalized * 2.7) / np.log1p(2.7)
        
        x_curved = hand_x ** 0.9
        base_tempo = 55 + int(x_curved * 105)
        tempo_variation = int(area_curved * 30 * (1 if hand_y > 0.5 else -1))
        self.tempo_target = max(40, min(180, base_tempo + tempo_variation))
        
        self.root_note = 36 + int((1 - x_curved) * 24) + int(area_curved * 6)
        
        scale_index = int(hand_x * len(self.scales))
        scale_names = list(self.scales.keys())
        self.current_scale = scale_names[scale_index % len(scale_names)]
        
        self.swing_target = area_curved * 0.8
        self.variation_intensity = 0.15 + area_curved * 0.55
        
        self.effects['chorus']['depth'] = 0.15 + area_curved * 0.35
        self.effects['chorus']['mix'] = 0.2 + area_curved * 0.2
        
        style_zone = int(hand_x * 3) + int(hand_y * 3) * 3
        style_zone = style_zone % len(self.patterns)
        
        if area_normalized > 0.7:
            style_zone = (style_zone + 4) % len(self.patterns)
        
        pattern_names = list(self.patterns.keys())
        self.style_target = pattern_names[style_zone]
        self.current_pattern = self.style_target
        
        progression_names = list(self.chord_progressions.keys())
        prog_index = int(hand_y * len(progression_names)) % len(progression_names)
        self.current_chord_progression = progression_names[prog_index]
        
        bass_presets = list(self.synth_presets['bass'].keys())
        lead_presets = list(self.synth_presets['lead'].keys())
        pad_presets = list(self.synth_presets['pad'].keys())
        
        bass_preset_idx = int(hand_x * len(bass_presets)) % len(bass_presets)
        lead_preset_idx = int((hand_x + hand_y) * len(lead_presets)) % len(lead_presets)
        pad_preset_idx = int(hand_y * len(pad_presets)) % len(pad_presets)
        
        self.set_synth_preset('bass', bass_presets[bass_preset_idx])
        self.set_synth_preset('lead', lead_presets[lead_preset_idx])
        self.set_synth_preset('pad', pad_presets[pad_preset_idx])
        
        self.reverb_amount_target = 0.08 + area_curved * 0.2
        self.delay_amount = 0.05 + area_curved * 0.15
        self.filter_cutoff = 0.4 + area_curved * 0.4
        
        self.effects['reverb']['size'] = 0.3 + area_curved * 0.5
        self.effects['delay']['feedback'] = 0.2 + area_curved * 0.4
        self.effects['filter']['resonance'] = 0.2 + area_curved * 0.6
        
        zone_x = int(hand_x * 3)
        zone_y = int(hand_y * 3)
        
        self.channels['drums']['enabled'] = True
        self.channels['bass']['enabled'] = area_normalized > 0.15 or zone_y >= 1
        self.channels['lead']['enabled'] = area_normalized > 0.25 or zone_x >= 1
        self.channels['pad']['enabled'] = area_normalized > 0.35 or (zone_x >= 1 and zone_y >= 1)
        self.channels['keys']['enabled'] = area_normalized > 0.5 or zone_y >= 2
        self.channels['arp']['enabled'] = area_normalized > 0.65 or (zone_x >= 2 and zone_y >= 2)
        
        self.channels['bass']['volume'] *= (0.7 + area_curved * 0.3)
        self.channels['lead']['volume'] *= (0.6 + area_curved * 0.4)
        self.channels['pad']['volume'] *= (0.5 + area_curved * 0.5)
        self.channels['keys']['volume'] *= (0.5 + area_curved * 0.5)
        self.channels['arp']['volume'] *= (0.4 + area_curved * 0.6)
        
        drum_types = ['kick', 'snare', 'clap', 'hihat_closed', 'hihat_open', 'tom_high', 'tom_mid', 'tom_low']
        zone_drum_index = (zone_x + zone_y * 3) % len(drum_types)
        self.editing_drum = drum_types[zone_drum_index]
    
    def set_sound_kit(self, kit_name):
        if kit_name in self.kit_presets:
            self.sound_kit = kit_name
            preset = self.kit_presets[kit_name]
            for drum, params in preset.items():
                if drum in self.drum_params:
                    for key, value in params.items():
                        target_key = f'{key}_target'
                        if target_key in self.drum_params[drum]:
                            self.drum_params[drum][target_key] = value
    
    def set_synth_preset(self, channel, preset_name):
        if channel in self.synth_presets and preset_name in self.synth_presets[channel]:
            self.synth_params[channel] = self.synth_presets[channel][preset_name].copy()
    
    def set_chord_progression(self, progression_name):
        if progression_name in self.chord_progressions:
            self.current_chord_progression = progression_name
    
    def update_from_color(self, colors):
        if colors is None or len(colors) == 0:
            return
        
        num_colors = min(len(colors), 5)
        r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
        saturation_sum = 0.0
        brightness_sum = 0.0
        
        for i in range(num_colors):
            color = colors[i]
            r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            r_sum += r
            g_sum += g
            b_sum += b
            
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            delta = max_c - min_c
            
            saturation_sum += (delta / max_c) if max_c > 0 else 0
            brightness_sum += max_c
        
        r = r_sum / num_colors
        g = g_sum / num_colors
        b = b_sum / num_colors
        saturation = saturation_sum / num_colors
        brightness = brightness_sum / num_colors
        
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        if delta == 0:
            hue = 0
        elif max_c == r:
            hue = 60 * (((g - b) / delta) % 6)
        elif max_c == g:
            hue = 60 * (((b - r) / delta) + 2)
        else:
            hue = 60 * (((r - g) / delta) + 4)
        
        if hue < 0:
            hue += 360
        
        saturation_enhanced = saturation ** 0.7
        brightness_enhanced = brightness ** 0.8
        
        for (hue_min, hue_max), kit_name in self.color_kit_mapping.items():
            if hue_min <= hue < hue_max:
                self.set_sound_kit(kit_name)
                break
        
        for (hue_min, hue_max), melody_info in self.color_melody_mapping.items():
            if hue_min <= hue < hue_max:
                self.current_base_notes = melody_info['base_notes']
                break
        
        hue_factor = hue / 360.0
        pitch_offset = (hue_factor - 0.5) * 0.4
        self.drum_params['kick']['pitch_target'] = 0.8 + saturation_enhanced * 0.5 + pitch_offset
        self.drum_params['snare']['pitch_target'] = 0.9 + saturation_enhanced * 0.4 + pitch_offset * 0.5
        self.drum_params['tom_high']['pitch_target'] = 1.3 + saturation_enhanced * 0.4
        self.drum_params['tom_mid']['pitch_target'] = 0.9 + saturation_enhanced * 0.3
        self.drum_params['tom_low']['pitch_target'] = 0.6 + saturation_enhanced * 0.3
        
        self.drum_params['kick']['tone_target'] = 0.2 + saturation_enhanced * 0.7
        self.drum_params['snare']['tone_target'] = 0.2 + saturation_enhanced * 0.7
        self.drum_params['snare']['snappy_target'] = 0.3 + saturation_enhanced * 0.6
        self.drum_params['hihat_closed']['tone_target'] = 0.5 + saturation_enhanced * 0.5
        
        decay_base = 0.15 + brightness_enhanced * 0.5
        for drum in self.drum_params:
            if 'decay_target' in self.drum_params[drum]:
                self.drum_params[drum]['decay_target'] = decay_base + saturation_enhanced * 0.2
        
        self.reverb_amount_target = 0.08 + saturation_enhanced * 0.5 + brightness_enhanced * 0.15
        self.delay_amount = 0.05 + saturation_enhanced * 0.25 + brightness_enhanced * 0.1
        self.filter_cutoff = 0.35 + saturation_enhanced * 0.4 + brightness_enhanced * 0.25
        self.filter_resonance = 0.2 + saturation_enhanced * 0.5
        
        self.synth_params['bass']['filter_cutoff'] = 0.15 + saturation_enhanced * 0.35
        self.synth_params['lead']['filter_cutoff'] = 0.3 + saturation_enhanced * 0.5
        self.synth_params['pad']['filter_cutoff'] = 0.2 + brightness_enhanced * 0.4
        self.synth_params['keys']['filter_cutoff'] = 0.25 + saturation_enhanced * 0.45
        self.synth_params['arp']['filter_cutoff'] = 0.4 + saturation_enhanced * 0.4
        
        self.synth_params['bass']['detune'] = saturation_enhanced * 0.15
        self.synth_params['lead']['detune'] = saturation_enhanced * 0.2
        self.synth_params['pad']['detune'] = brightness_enhanced * 0.2
        
        self.channels['bass']['volume'] = 0.5 + saturation_enhanced * 0.3
        self.channels['lead']['volume'] = 0.4 + brightness_enhanced * 0.3
        self.channels['pad']['volume'] = 0.2 + saturation_enhanced * 0.4
        self.channels['keys']['volume'] = 0.3 + brightness_enhanced * 0.3
        self.channels['arp']['volume'] = 0.3 + saturation_enhanced * 0.35
        
        self.effects['chorus']['depth'] = 0.15 + saturation_enhanced * 0.4
        self.effects['chorus']['rate'] = 0.3 + brightness_enhanced * 0.6
        self.effects['chorus']['mix'] = 0.2 + saturation_enhanced * 0.25
        
        hue_scale_index = int(hue / 30) % len(self.scales)
        scale_names = list(self.scales.keys())
        self.current_scale = scale_names[hue_scale_index]
        
        self.root_note = 36 + int(hue / 360 * 24)
        
        self.variation_intensity = 0.2 + saturation_enhanced * 0.5 + brightness_enhanced * 0.2
        self.swing_target = saturation_enhanced * 0.6 + brightness_enhanced * 0.2
        self.master_volume_target = 0.55 + saturation_enhanced * 0.25 + brightness_enhanced * 0.15
        
        self.color_hue = hue
        self.color_saturation = saturation
        self.color_brightness = brightness
    
    def get_display_info(self):
        return {
            'tempo': int(self.tempo),
            'current_step': self.current_step,
            'current_pattern': self.current_pattern,
            'editing_drum': getattr(self, 'editing_drum', 'kick'),
            'swing': self.swing,
            'step_count': self.step_count,
            'variation': self.variation_intensity,
            'sound_kit': self.sound_kit,
            'midi_enabled': self.midi_enabled,
            'channels': self.channels,
            'current_scale': self.current_scale,
            'root_note': self.root_note,
            'current_notes': self.current_notes,
            'current_chord': self.current_chord,
            'chord_progression': self.current_chord_progression,
            'synth_params': {ch: params.get('waveform', 'sine') for ch, params in self.synth_params.items()},
            'effects': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in self.effects.items()},
        }
    
    def get_pattern_for_display(self):
        return self._blend_patterns()


class AdvancedSampler:
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.thread = None
        
        # 音阶定义（扩展更多音阶）
        self.scales = {
            'pentatonic': [0, 2, 4, 7, 9],
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'blues': [0, 3, 5, 6, 7, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        }
        
        # 乐器库定义（10+种基础乐器，涵盖弦乐、管乐、打击乐等）
        self.instruments = {
            # === 弦乐器类 ===
            'piano': {
                'name': '钢琴',
                'name_en': 'Piano',
                'category': '键盘乐器',
                'range_low': 21, 'range_high': 108,  # A0-C8
                'play_style': '击弦',
                'attack': 0.005, 'decay': 0.3, 'sustain': 0.4, 'release': 0.5,
                'bright': 0.8, 'harm': [1, 0.5, 0.25, 0.125],
                'vibrato_rate': 0, 'vibrato_depth': 0,
            },
            'violin': {
                'name': '小提琴',
                'name_en': 'Violin',
                'category': '弓弦乐器',
                'range_low': 55, 'range_high': 96,  # G3-E7
                'play_style': '弓弦/拨弦',
                'attack': 0.12, 'decay': 0.15, 'sustain': 0.85, 'release': 0.4,
                'bright': 0.7, 'harm': [1, 0.6, 0.35, 0.2, 0.1],
                'vibrato_rate': 5.5, 'vibrato_depth': 0.02,
            },
            'cello': {
                'name': '大提琴',
                'name_en': 'Cello',
                'category': '弓弦乐器',
                'range_low': 36, 'range_high': 76,  # C2-C6
                'play_style': '弓弦/拨弦',
                'attack': 0.18, 'decay': 0.2, 'sustain': 0.8, 'release': 0.5,
                'bright': 0.4, 'harm': [1, 0.5, 0.3, 0.15, 0.08],
                'vibrato_rate': 4.5, 'vibrato_depth': 0.025,
            },
            'guitar': {
                'name': '吉他',
                'name_en': 'Guitar',
                'category': '拨弦乐器',
                'range_low': 40, 'range_high': 84,  # E2-E6
                'play_style': '拨弦/扫弦',
                'attack': 0.008, 'decay': 0.4, 'sustain': 0.3, 'release': 0.6,
                'bright': 0.6, 'harm': [1, 0.45, 0.25, 0.12],
                'vibrato_rate': 0, 'vibrato_depth': 0,
            },
            # === 管乐器类 ===
            'flute': {
                'name': '长笛',
                'name_en': 'Flute',
                'category': '木管乐器',
                'range_low': 60, 'range_high': 96,  # C4-C7
                'play_style': '吹奏',
                'attack': 0.08, 'decay': 0.1, 'sustain': 0.9, 'release': 0.3,
                'bright': 0.9, 'harm': [1, 0.3, 0.15, 0.05],
                'vibrato_rate': 5.0, 'vibrato_depth': 0.015,
            },
            'saxophone': {
                'name': '萨克斯',
                'name_en': 'Saxophone',
                'category': '木管乐器',
                'range_low': 49, 'range_high': 80,  # G3-A♭5
                'play_style': '吹奏',
                'attack': 0.05, 'decay': 0.15, 'sustain': 0.85, 'release': 0.35,
                'bright': 0.65, 'harm': [1, 0.7, 0.4, 0.25, 0.12],
                'vibrato_rate': 4.5, 'vibrato_depth': 0.02,
            },
            'trumpet': {
                'name': '小号',
                'name_en': 'Trumpet',
                'category': '铜管乐器',
                'range_low': 54, 'range_high': 82,  # F♯3-D6
                'play_style': '吹奏',
                'attack': 0.03, 'decay': 0.12, 'sustain': 0.8, 'release': 0.25,
                'bright': 0.95, 'harm': [1, 0.8, 0.5, 0.3, 0.15],
                'vibrato_rate': 5.5, 'vibrato_depth': 0.018,
            },
            # === 打击乐器类 ===
            'marimba': {
                'name': '马林巴',
                'name_en': 'Marimba',
                'category': '有音高打击乐器',
                'range_low': 36, 'range_high': 84,  # C2-C6
                'play_style': '敲击',
                'attack': 0.001, 'decay': 0.6, 'sustain': 0.1, 'release': 0.8,
                'bright': 0.5, 'harm': [1, 0.4, 0.2, 0.1, 0.05],
                'vibrato_rate': 0, 'vibrato_depth': 0,
            },
            'vibraphone': {
                'name': '颤音琴',
                'name_en': 'Vibraphone',
                'category': '有音高打击乐器',
                'range_low': 48, 'range_high': 84,  # C3-C6
                'play_style': '敲击',
                'attack': 0.002, 'decay': 0.7, 'sustain': 0.2, 'release': 0.9,
                'bright': 0.7, 'harm': [1, 0.5, 0.25, 0.12],
                'vibrato_rate': 6.0, 'vibrato_depth': 0.03,
            },
            # === 电子乐器类 ===
            'synth_lead': {
                'name': '合成器主音',
                'name_en': 'Synth Lead',
                'category': '电子乐器',
                'range_low': 24, 'range_high': 108,  # 全音域
                'play_style': '电子合成',
                'attack': 0.02, 'decay': 0.1, 'sustain': 0.7, 'release': 0.3,
                'bright': 0.85, 'harm': [1, 0.9, 0.5, 0.3, 0.15],
                'vibrato_rate': 4.0, 'vibrato_depth': 0.025,
            },
            'synth_pad': {
                'name': '合成器垫音',
                'name_en': 'Synth Pad',
                'category': '电子乐器',
                'range_low': 24, 'range_high': 108,
                'play_style': '电子合成',
                'attack': 0.4, 'decay': 0.2, 'sustain': 0.85, 'release': 1.2,
                'bright': 0.35, 'harm': [1, 0.7, 0.5, 0.35, 0.2],
                'vibrato_rate': 2.5, 'vibrato_depth': 0.01,
            },
            'synth_bass': {
                'name': '合成贝斯',
                'name_en': 'Synth Bass',
                'category': '电子乐器',
                'range_low': 24, 'range_high': 60,  # C1-C4
                'play_style': '电子合成',
                'attack': 0.005, 'decay': 0.15, 'sustain': 0.6, 'release': 0.2,
                'bright': 0.3, 'harm': [1, 0.4, 0.2, 0.1],
                'vibrato_rate': 0, 'vibrato_depth': 0,
            },
            # === 其他乐器 ===
            'harp': {
                'name': '竖琴',
                'name_en': 'Harp',
                'category': '拨弦乐器',
                'range_low': 23, 'range_high': 103,  # G♭1-G♯7
                'play_style': '拨弦',
                'attack': 0.01, 'decay': 0.5, 'sustain': 0.15, 'release': 0.7,
                'bright': 0.55, 'harm': [1, 0.35, 0.18, 0.08],
                'vibrato_rate': 0, 'vibrato_depth': 0,
            },
            'organ': {
                'name': '管风琴',
                'name_en': 'Organ',
                'category': '键盘乐器',
                'range_low': 24, 'range_high': 108,
                'play_style': '风鸣',
                'attack': 0.05, 'decay': 0.05, 'sustain': 0.95, 'release': 0.15,
                'bright': 0.6, 'harm': [1, 0.8, 0.6, 0.4, 0.25, 0.15],
                'vibrato_rate': 3.5, 'vibrato_depth': 0.01,
            },
        }
        
        # 乐器分类索引
        self.instrument_categories = {
            '弦乐器': ['piano', 'violin', 'cello', 'guitar', 'harp'],
            '管乐器': ['flute', 'saxophone', 'trumpet'],
            '打击乐器': ['marimba', 'vibraphone'],
            '电子乐器': ['synth_lead', 'synth_pad', 'synth_bass'],
            '键盘乐器': ['piano', 'organ'],
        }
        
        # 风格定义（颜色决定风格，风格内多种乐器组合）
        self.style_instruments = {
            'warm': {  # 温暖风格（红色系）
                'name': '温暖',
                'instruments': ['trumpet', 'saxophone', 'guitar', 'synth_pad'],
                'description': '铜管+木管+拨弦组合',
            },
            'bright': {  # 明亮风格（橙色系）
                'name': '明亮',
                'instruments': ['vibraphone', 'piano', 'flute', 'violin'],
                'description': '打击乐+键盘+木管组合',
            },
            'happy': {  # 快乐风格（黄色系）
                'name': '快乐',
                'instruments': ['piano', 'guitar', 'marimba', 'trumpet'],
                'description': '键盘+拨弦+打击乐组合',
            },
            'natural': {  # 自然风格（绿色系）
                'name': '自然',
                'instruments': ['violin', 'cello', 'guitar', 'flute'],
                'description': '弦乐+木管组合',
            },
            'calm': {  # 平静风格（青色系）
                'name': '平静',
                'instruments': ['harp', 'flute', 'synth_pad', 'vibraphone'],
                'description': '竖琴+长笛+垫音组合',
            },
            'cool': {  # 冷静风格（蓝色系）
                'name': '冷静',
                'instruments': ['cello', 'saxophone', 'synth_pad', 'piano'],
                'description': '大提琴+萨克斯+垫音组合',
            },
            'mystic': {  # 神秘风格（紫色系）
                'name': '神秘',
                'instruments': ['synth_lead', 'marimba', 'organ', 'harp'],
                'description': '合成器+打击乐+管风琴组合',
            },
        }
        
        # 当前状态
        self.base_note = 60
        self.current_scale = 'pentatonic'
        self.current_instrument = 'piano'
        self.current_style = 'bright'
        self.tempo = 120
        self.num_instruments = 1
        self.melody_index = 0
        self.beat_phase = 0.0
        
        # 多通道系统（9个通道：8个乐器+1个鼓点）
        self.num_channels = 9
        self.channel_instruments = ['piano'] * 9  # 每个通道的乐器
        self.channel_volumes = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.15]  # 最后一个是鼓点
        self.channel_active = [False] * 9
        
        # 旋律生成参数
        self.last_melody_note = 60
        self.melody_direction = 1  # 1=上行, -1=下行
        self.melody_contour_counter = 0
        self.variation_mode = 0  # 变奏模式
        
        # 节奏型定义（切分音、附点音符）
        self.rhythm_patterns = [
            [1.0, 0.5, 0.5, 1.0],  # 基础四分音符+八分音符
            [0.5, 0.5, 1.0, 0.5, 0.5],  # 切分音
            [1.5, 0.5, 1.0],  # 附点音符
            [0.5, 1.0, 0.5, 1.0],  # 反切分
            [1.0, 0.25, 0.25, 0.5, 1.0],  # 十六分音符组合
        ]
        self.current_rhythm_pattern = 0
        self.rhythm_position = 0
        self.sub_beat_phase = 0.0
        
        # 装饰音参数
        self.grace_note_active = False
        self.grace_note_freq = 0.0
        self.grace_note_env = 0.0
        self.grace_note_phase = 0.0
        self.trill_active = False
        self.trill_freq = 0.0
        self.trill_counter = 0
        self.trill_phase = 0.0
        self.glide_active = False
        self.glide_start_freq = 0.0
        self.glide_end_freq = 0.0
        self.glide_phase = 0.0
        self.glide_note_phase = 0.0
        
        # 鼓点参数（扩展节奏型）
        self.drum_patterns = [
            [1, 0, 0, 1, 0, 0, 1, 0],  # 基础4/4
            [1, 0, 0.5, 0, 1, 0, 0.5, 0],  # 带弱拍
            [1, 0.5, 0, 1, 0, 0.5, 0, 0],  # 切分鼓点
            [1, 0, 1, 0, 0.5, 0, 1, 0.5],  # 复杂节奏
        ]
        self.drum_pattern_index = 0
        self.drum_index = 0
        self.kick_phase = 0.0
        self.snare_phase = 0.0
        self.hihat_phase = 0.0
        self.kick_env = 0.0
        self.snare_env = 0.0
        self.hihat_env = 0.0
        
        # 音色参数
        self.brightness = 0.5
        self.warmth = 0.5
        self.space = 0.3
        
        # 颜色氛围
        self.color_mood = 'neutral'
        self.color_energy = 0.5
        
        # 振荡器状态
        self.phases = [0.0] * 16
        self.envelopes = [0.0] * 16
        self.note_frequencies = [440.0] * 16
        self.note_active = [False] * 16
        
        # 低通滤波器状态
        self.filter_state = [0.0] * 16
        self.filter_cutoff = 2000.0
        
        # 效果缓冲
        self.reverb_buffers = [
            np.zeros(int(sample_rate * 0.5)),
            np.zeros(int(sample_rate * 0.7)),
            np.zeros(int(sample_rate * 0.3)),
        ]
        self.reverb_indices = [0, 0, 0]
        self.delay_buffer = np.zeros(int(sample_rate * 0.4))
        self.delay_index = 0
        
        self.volume = 0.35
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
    
    def _audio_loop(self):
        try:
            import sounddevice as sd
            with sd.OutputStream(samplerate=self.sample_rate,
                                channels=1,
                                callback=self._audio_callback,
                                blocksize=self.buffer_size):
                while self.running:
                    time.sleep(0.005)
        except ImportError:
            print("sounddevice not installed, audio disabled")
            self.running = False
    
    def _generate_kick(self):
        """生成底鼓"""
        output = np.zeros(self.buffer_size)
        if self.kick_env > 0.01:
            for i in range(self.buffer_size):
                # 频率下降的正弦波
                freq = 150 * np.exp(-self.kick_phase * 8) + 50
                output[i] = np.sin(self.kick_phase * 2 * np.pi * freq / self.sample_rate)
                self.kick_phase += 1
                self.kick_env *= 0.997
        return output * self.kick_env * 1.2
    
    def _generate_snare(self):
        """生成军鼓"""
        output = np.zeros(self.buffer_size)
        if self.snare_env > 0.01:
            for i in range(self.buffer_size):
                # 正弦波 + 噪声
                tone = np.sin(self.snare_phase * 2 * np.pi * 180 / self.sample_rate)
                noise = np.random.uniform(-1, 1) * 0.6
                output[i] = (tone * 0.5 + noise * 0.5)
                self.snare_phase += 1
                self.snare_env *= 0.992
        return output * self.snare_env * 0.9
    
    def _generate_hihat(self):
        """生成踩镲"""
        output = np.zeros(self.buffer_size)
        if self.hihat_env > 0.01:
            for i in range(self.buffer_size):
                # 高频噪声
                noise = np.random.uniform(-1, 1)
                # 高通滤波效果
                output[i] = noise * 0.9
                self.hihat_env *= 0.988
        return output * self.hihat_env * 0.5
    
    def _generate_note(self, note_idx, instrument=None):
        """生成单个音符的波形（支持不同乐器特性）"""
        freq = self.note_frequencies[note_idx]
        phase = self.phases[note_idx]
        env = self.envelopes[note_idx]
        
        if instrument is None:
            instrument = self.current_instrument
        inst = self.instruments.get(instrument, self.instruments['piano'])
        
        phase_inc = 2 * np.pi * freq / self.sample_rate
        
        # 获取乐器参数
        harmonics = inst.get('harm', [1, 0.5, 0.25, 0.125])
        vibrato_rate = inst.get('vibrato_rate', 0)
        vibrato_depth = inst.get('vibrato_depth', 0)
        brightness = inst.get('bright', 0.5)
        
        wave = np.zeros(self.buffer_size)
        
        # 生成波形（带颤音效果）
        for h, amp in enumerate(harmonics, 1):
            detune = 1.0 + (h - 1) * 0.001
            
            # 应用颤音（针对管弦乐器）
            if vibrato_rate > 0:
                vibrato = np.sin(2 * np.pi * vibrato_rate * np.arange(self.buffer_size) / self.sample_rate)
                vibrato_mod = 1.0 + vibrato * vibrato_depth
            else:
                vibrato_mod = 1.0
            
            wave += amp * np.sin((phase + phase_inc * np.arange(self.buffer_size) * vibrato_mod) * h * detune)
        
        # 添加次谐波增加温暖感
        wave += np.sin((phase + phase_inc * np.arange(self.buffer_size)) * 0.5) * 0.15 * self.warmth
        
        # 根据亮度调整高频内容
        if brightness > 0.7:
            # 明亮乐器：保留更多高频
            alpha = 0.2
        elif brightness < 0.4:
            # 温暖乐器：更多低通滤波
            alpha = 0.4
        else:
            alpha = 0.3
        
        # 应用低通滤波器
        for i in range(len(wave)):
            self.filter_state[note_idx] = alpha * wave[i] + (1 - alpha) * self.filter_state[note_idx]
            wave[i] = self.filter_state[note_idx]
        
        self.phases[note_idx] = phase + phase_inc * self.buffer_size
        
        decay_rate = 0.9995 if inst['sustain'] > 0.5 else 0.999
        self.envelopes[note_idx] *= decay_rate
        
        return wave * env * self.volume
    
    def _generate_grace_note(self):
        """生成倚音（装饰音）"""
        output = np.zeros(self.buffer_size)
        if self.grace_note_env > 0.01:
            phase_inc = 2 * np.pi * self.grace_note_freq / self.sample_rate
            for i in range(self.buffer_size):
                output[i] = np.sin(self.grace_note_phase) * self.grace_note_env
                self.grace_note_phase += phase_inc
                self.grace_note_env *= 0.95  # 快速衰减
        return output * 0.3
    
    def _generate_trill(self, base_freq):
        """生成颤音"""
        output = np.zeros(self.buffer_size)
        if self.trill_active:
            trill_freq = base_freq * (2 ** (1/12))  # 上方二度音
            phase_inc_base = 2 * np.pi * base_freq / self.sample_rate
            phase_inc_trill = 2 * np.pi * trill_freq / self.sample_rate
            
            for i in range(self.buffer_size):
                # 快速交替两个音
                if (self.trill_counter // 100) % 2 == 0:
                    output[i] = np.sin(self.trill_phase) * 0.5
                    self.trill_phase += phase_inc_base
                else:
                    output[i] = np.sin(self.trill_phase) * 0.5
                    self.trill_phase += phase_inc_trill
                self.trill_counter += 1
        return output
    
    def _generate_glide(self):
        """生成滑音（portamento）"""
        output = np.zeros(self.buffer_size)
        if self.glide_active and self.glide_phase < 1.0:
            # 线性插值频率
            current_freq = self.glide_start_freq + (self.glide_end_freq - self.glide_start_freq) * self.glide_phase
            phase_inc = 2 * np.pi * current_freq / self.sample_rate
            
            for i in range(self.buffer_size):
                output[i] = np.sin(self.glide_note_phase) * 0.5
                self.glide_note_phase += phase_inc
            
            self.glide_phase += 0.02  # 滑音速度
            if self.glide_phase >= 1.0:
                self.glide_active = False
        return output
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """音频回调函数（支持多通道多乐器）"""
        output = np.zeros(frames)
        
        # 更新节拍
        beat_duration = 60.0 / self.tempo
        beat_inc = self.buffer_size / self.sample_rate / beat_duration
        self.beat_phase += beat_inc
        
        # 每拍触发新音符
        if self.beat_phase >= 1.0:
            self.beat_phase -= 1.0
            self._trigger_next_notes()
            self._trigger_drums()
        
        # 生成乐器通道（前8个通道）
        for ch in range(min(self.num_channels - 1, 8)):
            if self.channel_active[ch] and self.note_active[ch]:
                instrument = self.channel_instruments[ch]
                note_wave = self._generate_note(ch, instrument)
                output += note_wave * self.channel_volumes[ch]
        
        # 生成装饰音
        output += self._generate_grace_note() * 0.25
        output += self._generate_glide() * 0.15
        
        # 第9通道：鼓点（根据风格调整）
        drum_intensity = self.channel_volumes[8] if self.num_channels > 8 else 0.15
        output += self._generate_kick() * drum_intensity * 1.2
        output += self._generate_snare() * drum_intensity * 0.9
        output += self._generate_hihat() * drum_intensity * 0.5
        
        # 应用多层混响
        for buf_idx, reverb_buf in enumerate(self.reverb_buffers):
            delay = int(len(reverb_buf) * (0.3 + buf_idx * 0.1))
            for i in range(frames):
                rev_idx = (self.reverb_indices[buf_idx] - delay + i) % len(reverb_buf)
                output[i] += reverb_buf[rev_idx] * 0.15 * self.space
                reverb_buf[self.reverb_indices[buf_idx]] = output[i]
            self.reverb_indices[buf_idx] = (self.reverb_indices[buf_idx] + 1) % len(reverb_buf)
        
        # 应用延迟
        delay_samples = int(self.sample_rate * 0.3)
        for i in range(frames):
            del_idx = (self.delay_index - delay_samples + i) % len(self.delay_buffer)
            output[i] += self.delay_buffer[del_idx] * 0.15
            self.delay_buffer[self.delay_index] = output[i]
            self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
        
        outdata[:, 0] = np.clip(output, -1, 1).astype(np.float32)
    
    def _trigger_next_notes(self):
        """触发下一组音符（多通道多乐器系统）"""
        scale = self.scales[self.current_scale]
        scale_len = len(scale)
        
        # 获取当前风格的乐器列表
        style_instruments = self.style_instruments.get(self.current_style, self.style_instruments['bright'])
        instruments = style_instruments['instruments']
        
        # 为每个通道分配乐器和音符（前8个是乐器，第9个是鼓点）
        for ch in range(min(self.num_channels - 1, 8)):
            # 获取该通道的乐器
            inst_idx = ch % len(instruments)
            instrument = instruments[inst_idx]
            self.channel_instruments[ch] = instrument
            
            # 获取乐器信息
            inst_info = self.instruments.get(instrument, self.instruments['piano'])
            range_low = inst_info.get('range_low', 24)
            range_high = inst_info.get('range_high', 108)
            
            # 根据通道角色确定音符
            if ch == 0:
                # 主旋律通道
                base_offset = self.melody_index % scale_len
                if self.melody_contour_counter % 4 == 0:
                    jump_interval = [2, 4, -2, -4][self.melody_contour_counter % 4]
                    melody_offset = (base_offset + jump_interval) % scale_len
                else:
                    melody_offset = base_offset
                octave = (self.melody_index // scale_len) % 2
                note = self.base_note + scale[melody_offset] + octave * 12
                
                # 滑音效果
                if self.melody_index % 3 == 0:
                    self.glide_active = True
                    self.glide_start_freq = self.note_frequencies[ch]
                    self.glide_end_freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
                    self.glide_phase = 0.0
                    self.glide_note_phase = 0.0
                
                # 倚音装饰
                if np.random.random() < 0.2:
                    self.grace_note_freq = 440.0 * (2.0 ** ((note - 69) / 12.0)) * (2 ** (2/12))
                    self.grace_note_env = 0.8
                    self.grace_note_phase = 0.0
                    
            elif ch == 1:
                # 副旋律通道（三度音程）
                harmony_offset = (self.melody_index + 2) % scale_len
                note = self.base_note + scale[harmony_offset] + 12
                
            elif ch == 2:
                # 和声通道1（五度音程）
                harmony_offset = (self.melody_index + 4) % scale_len
                note = self.base_note + scale[harmony_offset]
                
            elif ch == 3:
                # 和声通道2（八度+三度）
                harmony_offset = (self.melody_index + 2) % scale_len
                note = self.base_note + scale[harmony_offset] - 12
                
            elif ch == 4:
                # 低音通道
                bass_interval = 0 if self.melody_index % 2 == 0 else 4
                note = self.base_note - 24 + scale[(self.melody_index + bass_interval) % scale_len]
                
            elif ch == 5:
                # 高音装饰通道
                note = self.base_note + scale[self.melody_index % scale_len] + 24
                
            elif ch == 6:
                # 中音填充通道
                note = self.base_note + scale[(self.melody_index + 3) % scale_len] + 7
                
            else:
                # 节奏强调通道（第8通道）
                note = self.base_note + scale[self.melody_index % scale_len] - 12
            
            # 限制在乐器音域范围内
            note = max(range_low, min(range_high, note))
            
            # 设置音符
            self.note_frequencies[ch] = 440.0 * (2.0 ** ((note - 69) / 12.0))
            self.note_active[ch] = True
            self.envelopes[ch] = 1.0
            self.channel_active[ch] = True
        
        # 第9通道标记为鼓点通道
        if self.num_channels > 8:
            self.channel_instruments[8] = 'drums'
            self.channel_active[8] = True
        
        # 更新旋律状态
        self.melody_index = (self.melody_index + 1) % (scale_len * 8)
        self.melody_contour_counter += 1
        
        # 定期改变旋律方向
        if self.melody_contour_counter > 8:
            self.melody_direction *= -1
            self.melody_contour_counter = 0
        
        # 变奏模式切换
        if self.melody_index % (scale_len * 4) == 0:
            self.variation_mode = (self.variation_mode + 1) % 3
    
    def _trigger_drums(self):
        """触发鼓点（扩展节奏型）"""
        drum_pattern = self.drum_patterns[self.drum_pattern_index]
        pattern_step = self.drum_index % len(drum_pattern)
        pattern_value = drum_pattern[pattern_step]
        
        # 底鼓
        if pattern_value >= 1:
            self.kick_env = 1.0
            self.kick_phase = 0.0
        elif pattern_value == 0.5:
            self.kick_env = 0.5
            self.kick_phase = 0.0
        
        # 军鼓：第3和第7拍
        if pattern_step in [2, 6]:
            self.snare_env = 1.0
            self.snare_phase = 0.0
        
        # 踩镲：根据模式
        if self.drum_pattern_index == 0:
            if pattern_step % 2 == 0:
                self.hihat_env = 0.6
        elif self.drum_pattern_index == 1:
            self.hihat_env = 0.4
        else:
            self.hihat_env = 0.5
        
        self.drum_index += 1
        
        # 定期切换鼓点模式
        if self.drum_index % 16 == 0:
            self.drum_pattern_index = (self.drum_pattern_index + 1) % len(self.drum_patterns)
    
    def update_from_gesture(self, hand_y, hand_x, hand_area, finger_count):
        """从手势更新采样器参数"""
        # 手的Y位置控制基础音高
        self.base_note = 48 + int((1.0 - hand_y) * 24)  # C3-C5
        
        # 手的X位置控制节奏（更大范围）
        self.tempo = 40 + int(hand_x * 180)  # 40-220 BPM
        
        # 手的面积控制乐器数量和节奏型
        self.num_instruments = max(1, min(4, int(hand_area / 12000) + 1))
        
        # 根据面积切换节奏型
        if hand_area > 40000:
            self.current_rhythm_pattern = 4  # 复杂节奏
            self.drum_pattern_index = 3
        elif hand_area > 30000:
            self.current_rhythm_pattern = 2  # 附点音符
            self.drum_pattern_index = 2
        elif hand_area > 20000:
            self.current_rhythm_pattern = 1  # 切分音
            self.drum_pattern_index = 1
        else:
            self.current_rhythm_pattern = 0  # 基础
            self.drum_pattern_index = 0
        
        # 手指数量控制音阶
        scale_names = list(self.scales.keys())
        self.current_scale = scale_names[finger_count % len(scale_names)]
        
        # 手指数量也影响装饰音
        if finger_count >= 4:
            self.trill_active = True  # 开启颤音
        else:
            self.trill_active = False
    
    def update_from_color(self, colors):
        """从颜色更新风格（颜色决定风格，风格内多乐器组合）"""
        if colors is None or len(colors) == 0:
            return
        
        main_color = colors[0]
        r, g, b = main_color[0] / 255.0, main_color[1] / 255.0, main_color[2] / 255.0
        
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        
        if delta == 0:
            hue = 0
        elif max_c == r:
            hue = 60 * (((g - b) / delta) % 6)
        elif max_c == g:
            hue = 60 * (((b - r) / delta) + 2)
        else:
            hue = 60 * (((r - g) / delta) + 4)
        
        saturation = delta / max_c if max_c > 0 else 0
        value = max_c
        
        # 根据色相选择风格（风格决定多乐器组合）
        if hue < 30 or hue >= 330:
            # 红色：温暖风格
            self.current_style = 'warm'
            self.color_mood = 'warm'
        elif hue < 60:
            # 橙色：明亮风格
            self.current_style = 'bright'
            self.color_mood = 'bright'
        elif hue < 120:
            # 黄色-黄绿：快乐风格
            self.current_style = 'happy'
            self.color_mood = 'happy'
        elif hue < 180:
            # 绿色-青绿：自然风格
            self.current_style = 'natural'
            self.color_mood = 'natural'
        elif hue < 240:
            # 青色-蓝色：平静风格
            self.current_style = 'calm'
            self.color_mood = 'calm'
        elif hue < 300:
            # 蓝紫-紫色：冷静风格
            self.current_style = 'cool'
            self.color_mood = 'cool'
        else:
            # 紫红：神秘风格
            self.current_style = 'mystic'
            self.color_mood = 'mystic'
        
        # 设置当前主乐器（风格中的第一个乐器）
        style_info = self.style_instruments.get(self.current_style, self.style_instruments['bright'])
        self.current_instrument = style_info['instruments'][0]
        
        self.brightness = 0.3 + saturation * 0.7
        self.space = 0.1 + value * 0.5
        self.color_energy = saturation * value

# 全局合成器实例
synth = None
sampler = None

def smooth_gesture(gesture_name):
    if gesture_name:
        gesture_history.append(gesture_name)
    
    if len(gesture_history) >= 3:
        from collections import Counter
        counts = Counter(gesture_history)
        most_common = counts.most_common(1)[0]
        # 只需要超过一半就返回
        if most_common[1] >= 2:
            return most_common[0]
    
    return gesture_name if gesture_name else ""

def smooth_position(x, y, w, h):
    """平滑手部位置 - 更快响应"""
    global hand_position_history, hand_size_history
    
    hand_position_history.append((x, y))
    hand_size_history.append((w, h))
    
    # 简单平均，更快响应
    if len(hand_position_history) == 0:
        return x, y, w, h
    
    sum_x = sum(p[0] for p in hand_position_history)
    sum_y = sum(p[1] for p in hand_position_history)
    sum_w = sum(s[0] for s in hand_size_history)
    sum_h = sum(s[1] for s in hand_size_history)
    n = len(hand_position_history)
    
    return sum_x // n, sum_y // n, sum_w // n, sum_h // n

def detect_skin(image):
    """多颜色空间融合的皮肤检测"""
    # HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 15, 40], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # YCrCb空间
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # LAB空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lower_lab = np.array([0, 130, 130], dtype=np.uint8)
    upper_lab = np.array([255, 160, 160], dtype=np.uint8)
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)
    
    # 融合三个颜色空间的结果
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    mask = cv2.bitwise_or(mask, mask_lab)
    
    return mask

def process_hand(image):
    """改进的手部检测算法"""
    global last_valid_contour, last_valid_time
    
    h, w = image.shape[:2]
    
    # 1. 皮肤颜色检测
    skin_mask = detect_skin(image)
    
    # 2. 背景减除
    fg_mask = bg_subtractor.apply(image, learningRate=0.001)
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    
    # 3. 融合检测结果
    combined = cv2.bitwise_and(skin_mask, fg_mask)
    
    # 如果融合结果太少，使用纯皮肤检测
    min_combined_pixels = int(1000 * (scale_factor ** 2))
    if cv2.countNonZero(combined) < min_combined_pixels:
        combined = skin_mask
    
    # 4. 形态学处理（根据分辨率调整内核大小）
    kernel_size = max(5, int(7 * scale_factor))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 如果没有检测到轮廓，检查是否可以使用上一次的有效轮廓
        current_time = time.time()
        if last_valid_contour is not None and current_time - last_valid_time < 0.5:
            return last_valid_contour, None, combined
        return None, None, combined
    
    # 6. 筛选手部轮廓
    best_contour = None
    best_score = 0
    
    # 根据分辨率缩放阈值
    min_area = int(1500 * (scale_factor ** 2))
    area_low = int(2000 * (scale_factor ** 2))
    area_high = int(80000 * (scale_factor ** 2))
    area_ext_low = int(1500 * (scale_factor ** 2))
    area_ext_high = int(100000 * (scale_factor ** 2))
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / ch if ch > 0 else 0
        
        # 计算得分
        score = 0
        
        # 面积得分 - 放宽范围（使用缩放后的阈值）
        if area_low < area < area_high:
            score += 30
        elif area_ext_low < area < area_ext_high:
            score += 20
        
        # 圆形度得分 - 放宽范围
        if 0.2 < circularity < 0.7:
            score += 25
        elif 0.15 < circularity < 0.8:
            score += 15
        
        # 实度得分
        if 0.4 < solidity < 0.95:
            score += 25
        elif 0.3 < solidity < 0.98:
            score += 15
        
        # 长宽比得分
        if 0.5 < aspect_ratio < 2.0:
            score += 20
        elif 0.3 < aspect_ratio < 2.5:
            score += 10
        
        # 如果有历史位置，给予位置接近的轮廓额外分数
        if len(hand_position_history) > 0:
            last_x, last_y = hand_position_history[-1]
            last_w, last_h = hand_size_history[-1]
            center_x = x + cw // 2
            center_y = y + ch // 2
            last_center_x = last_x + last_w // 2
            last_center_y = last_y + last_h // 2
            
            distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
            max_distance = np.sqrt(w**2 + h**2) * 0.4  # 放宽最大允许移动距离
            
            if distance < max_distance:
                score += int(30 * (1 - distance / max_distance))
        
        if score > best_score:
            best_score = score
            best_contour = cnt
    
    if best_contour is not None:
        last_valid_contour = best_contour
        last_valid_time = time.time()
    
    defects = None
    if best_contour is not None:
        hull = cv2.convexHull(best_contour, returnPoints=False)
        defects = cv2.convexityDefects(best_contour, hull)
    
    return best_contour, defects, combined

def recognize_gesture(contour, defects):
    if contour is None:
        return "", 0
    
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    solidity = area / hull_area if hull_area > 0 else 0
    
    x, y, cw, ch = cv2.boundingRect(contour)
    
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 计算手指数量 - 使用凸包顶点检测
    finger_count = 0
    
    if defects is not None and defects.shape[0] > 0:
        # 获取凸包点
        hull_points = cv2.convexHull(contour, returnPoints=True)
        
        # 计算手的中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + cw // 2, y + ch // 2
        
        # 找出凸包的顶点（手指尖）
        finger_tips = []
        
        for i, pt in enumerate(hull_points):
            pt = pt[0]
            
            # 只考虑手部上半部分的点
            if pt[1] < cy + ch * 0.1:
                # 计算该点与中心的距离
                dist = np.sqrt((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
                
                # 如果距离足够远，可能是手指尖
                if dist > ch * 0.25:
                    finger_tips.append(pt)
        
        # 过滤相邻的重复点
        if len(finger_tips) > 0:
            filtered_tips = [finger_tips[0]]
            for tip in finger_tips[1:]:
                # 检查是否与已有点太近
                is_duplicate = False
                for ft in filtered_tips:
                    if np.sqrt((tip[0] - ft[0]) ** 2 + (tip[1] - ft[1]) ** 2) < cw * 0.2:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    filtered_tips.append(tip)
            
            finger_count = len(filtered_tips)
        
        # 使用缺陷点验证手指数量
        valid_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 400:  # 深度阈值
                cnt_f = tuple(contour[f][0])
                # 缺陷点应该在手的下半部分
                if cnt_f[1] > cy - ch * 0.1:
                    valid_defects += 1
        
        # 手指数量 = 缺陷数量 + 1（如果缺陷检测更可靠）
        if valid_defects > 0 and valid_defects < 5:
            finger_count = valid_defects + 1
        
        finger_count = min(max(finger_count, 0), 5)
    
    # 手势判断
    gesture_name = ""
    
    # 握拳：实度低
    if solidity < 0.35:
        gesture_name = "Fist"
    # 手掌：实度高或手指多
    elif solidity > 0.75 or finger_count >= 4:
        gesture_name = "Paper"
    # 剪刀：2个手指
    elif finger_count == 2:
        gesture_name = "Victory"
    # 一：1个手指
    elif finger_count == 1:
        gesture_name = "One"
    # 三：3个手指
    elif finger_count == 3:
        gesture_name = "Three"
    # 默认
    else:
        gesture_name = "Point"
    
    gesture_name = smooth_gesture(gesture_name)
    
    return gesture_name, finger_count

def draw_3d_cube(image, center_x, center_y, scale, rot_x, rot_y):
    h, w = image.shape[:2]
    size = int(60 * scale)
    
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]) * size
    
    angle_x = np.radians(rot_x)
    angle_y = np.radians(rot_y)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    vertices = vertices @ Ry.T @ Rx.T
    
    vertices[:, 0] += center_x
    vertices[:, 1] += center_y
    
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    faces = [
        ([0,1,2,3], (200, 100, 100)),
        ([4,5,6,7], (100, 200, 100)),
        ([0,1,5,4], (100, 100, 200)),
        ([2,3,7,6], (200, 200, 100)),
        ([1,2,6,5], (100, 200, 200)),
        ([0,3,7,4], (200, 100, 200))
    ]
    
    camera_pos = np.array([0, 0, 500])
    
    for face_indices, color in faces:
        face_center = np.mean(vertices[face_indices], axis=0)
        v1 = vertices[face_indices[1]] - vertices[face_indices[0]]
        v2 = vertices[face_indices[2]] - vertices[face_indices[0]]
        normal = np.cross(v1, v2)
        view_vec = camera_pos - face_center
        if np.dot(normal, view_vec) > 0:
            pts = vertices[face_indices].reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(image, [pts], color)
    
    for edge in edges:
        pt1 = tuple(vertices[edge[0]][:2].astype(int))
        pt2 = tuple(vertices[edge[1]][:2].astype(int))
        cv2.line(image, pt1, pt2, (50, 50, 50), 2)
    
    return image

def control_cube(gesture_name, finger_count, hand_center, w, h):
    global cube_rotation, cube_scale, cube_auto_rotate
    
    if hand_center is None:
        return
    
    cx, cy = hand_center
    
    if gesture_name == "One":
        cube_auto_rotate = False
        dx = (cx - w // 2) / w
        dy = (cy - h // 2) / h
        cube_rotation[0] += dy * 5
        cube_rotation[1] += dx * 5
    
    elif gesture_name == "Victory":
        cube_auto_rotate = False
        dy = (cy - h // 2) / h
        cube_scale = max(0.3, min(2.5, cube_scale - dy * 0.05))
    
    elif gesture_name == "Paper":
        cube_auto_rotate = True

# ASCII艺术字符集（按亮度排序，增加多样性）
ASCII_CHARS = " .',:;ilI1|\\/()[]{}?_-+~=<>!@#$%&*#"
ASCII_CHARS_EXTENDED = " .',:;ilI1|\\/()[]{}?_-+~=<>!@#$%&*#O0O8&@#*+"  # 扩展字符集（更多基础字符）
ASCII_COLORS = [
    (50, 50, 50),
    (70, 70, 70),
    (90, 90, 90),
    (110, 110, 110),
    (130, 130, 130),
    (150, 150, 150),
    (170, 170, 170),
    (190, 190, 190),
    (210, 210, 210),
    (230, 230, 230),
    (255, 255, 255),
    (255, 255, 150),
    (255, 230, 100),
    (255, 200, 50),
]

def create_ascii_art(image, contour, mask, synth=None, sampler=None):
    """在检测到的手部区域创建ASCII艺术效果，符号颜色根据明暗渐变，合成器/采样器影响字符"""
    if contour is None:
        return image
    
    h, w = image.shape[:2]
    
    # 获取手部边界框
    x, y, cw, ch = cv2.boundingRect(contour)
    
    # 验证边界框有效性
    if cw <= 0 or ch <= 0:
        return image
    
    # 扩展边界框
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    cw = min(w - x, cw + 2 * padding)
    ch = min(h - y, ch + 2 * padding)
    
    # 再次验证
    if cw <= 0 or ch <= 0 or x >= w or y >= h:
        return image
    
    # 裁剪手部区域
    try:
        hand_region = image[y:y+ch, x:x+cw].copy()
    except Exception:
        return image
    
    # 验证裁剪结果
    if hand_region is None or not isinstance(hand_region, np.ndarray):
        return image
    if hand_region.size == 0 or len(hand_region.shape) < 2:
        return image
    
    # 验证mask_region
    if mask is None or not isinstance(mask, np.ndarray):
        return image
    
    mask_region = mask[y:y+ch, x:x+cw]
    
    # 验证mask_region有效性
    if mask_region.size == 0:
        return image
    
    # 转换为灰度图
    try:
        if len(hand_region.shape) == 3:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_region.copy()  # 已经是灰度图，复制一份
    except Exception:
        return image
    
    # 确保gray是有效的numpy数组
    if not isinstance(gray, np.ndarray) or gray.size == 0 or len(gray.shape) < 2:
        return image
    
    # 音色颜色映射（扩展乐器）
    instrument_colors = {
        # 弦乐器
        'piano': {'base': (180, 200, 255), 'bright': (220, 240, 255)},      # 淡金色
        'violin': {'base': (200, 180, 150), 'bright': (230, 200, 170)},    # 木色
        'cello': {'base': (150, 120, 100), 'bright': (180, 150, 130)},     # 深木色
        'guitar': {'base': (180, 140, 100), 'bright': (210, 170, 130)},    # 棕色
        'harp': {'base': (200, 220, 255), 'bright': (230, 240, 255)},      # 金白色
        # 管乐器
        'flute': {'base': (200, 255, 255), 'bright': (230, 255, 255)},     # 银白色
        'saxophone': {'base': (255, 200, 100), 'bright': (255, 220, 140)}, # 金色
        'trumpet': {'base': (255, 180, 80), 'bright': (255, 200, 120)},    # 亮金色
        # 打击乐器
        'marimba': {'base': (180, 160, 120), 'bright': (210, 190, 150)},   # 木纹色
        'vibraphone': {'base': (180, 200, 220), 'bright': (210, 230, 250)},# 银蓝色
        # 电子乐器
        'synth_lead': {'base': (255, 100, 200), 'bright': (255, 150, 230)},# 粉紫色
        'synth_pad': {'base': (100, 150, 255), 'bright': (150, 200, 255)}, # 柔和蓝紫
        'synth_bass': {'base': (100, 100, 200), 'bright': (150, 150, 230)},# 深蓝紫
        # 其他
        'organ': {'base': (150, 130, 100), 'bright': (180, 160, 130)},     # 棕色
    }
    
    # 根据模式设置不同参数
    if sampler is not None and hasattr(sampler, 'drum_kits'):
        # 模式6：TR-808音序器模式
        base_char_size = max(14, min(20, cw // 12))
        # 当前步影响字符大小
        step_factor = 1.0 - sampler.current_step / 16 * 0.3
        char_size = int(base_char_size * step_factor)
        char_size = max(8, min(28, char_size))
        jitter_freq = 0
        # 节奏影响字符偏移
        char_offset = int((sampler.tempo - 60) / 30)
        chars_to_use = ASCII_CHARS_EXTENDED
        skip_rate = 0.25
        # 808音序器颜色（根据当前模式）
        pattern_colors = {
            'classic': {'base': (100, 150, 200), 'bright': (150, 200, 255)},
            'hiphop': {'base': (200, 100, 150), 'bright': (255, 150, 200)},
            'house': {'base': (100, 200, 150), 'bright': (150, 255, 200)},
            'techno': {'base': (150, 100, 200), 'bright': (200, 150, 255)},
            'breakbeat': {'base': (200, 150, 100), 'bright': (255, 200, 150)},
        }
        inst_colors = pattern_colors.get(sampler.current_pattern, pattern_colors['classic'])
    elif sampler is not None:
        # 模式6：采样器模式（旧版兼容）
        base_char_size = max(14, min(20, cw // 12))
        # 音高影响字符大小
        note_factor = 1.0 - (getattr(sampler, 'base_note', 60) - 48) / 24 * 0.4
        char_size = int(base_char_size * note_factor)
        char_size = max(8, min(28, char_size))
        jitter_freq = 0
        # 节奏影响字符偏移
        char_offset = int((sampler.tempo - 60) / 30)
        chars_to_use = ASCII_CHARS_EXTENDED
        skip_rate = 0.25
        # 获取当前音色颜色
        inst_colors = instrument_colors.get(getattr(sampler, 'current_instrument', 'piano'), instrument_colors['piano'])
    elif synth is not None:
        # 模式5：合成器模式，更大更稀疏，扩展字符集，无震荡
        base_char_size = max(14, min(20, cw // 12))
        # 频率影响字符大小：更灵敏（高频=小字符，低频=大字符）
        freq_factor = 1.0 - (synth.frequency - 80) / 700 * 0.5  # 更灵敏
        char_size = int(base_char_size * freq_factor)
        char_size = max(8, min(28, char_size))  # 扩大范围
        jitter_freq = 0  # 无震荡
        char_offset = int(synth.envelope_target * 30)  # 更多样的字符偏移
        chars_to_use = ASCII_CHARS_EXTENDED  # 扩展字符集
        skip_rate = 0.25  # 更稀疏
        inst_colors = None
    else:
        # 模式3：普通ASCII模式，回到之前的大小
        char_size = max(4, min(8, cw // 30))
        jitter_freq = 0
        char_offset = 0
        chars_to_use = ASCII_CHARS
        skip_rate = 0
        inst_colors = None
    
    # 创建ASCII艺术图像
    ascii_h = ch // char_size
    ascii_w = cw // char_size
    
    if ascii_h < 3 or ascii_w < 3:
        return image
    
    # 创建输出图像
    result = image.copy()
    
    # 当前时间用于抖动
    current_time = time.time()
    
    # 为每个字符生成随机相位偏移（基于位置，保持一致性）
    np.random.seed(42)
    phase_offsets = np.random.random((ascii_h, ascii_w)) * 2 * np.pi
    
    # 在手部区域绘制ASCII字符（稀疏采样）
    for row in range(ascii_h):
        for col in range(ascii_w):
            # 随机跳过一些字符，让显示更稀疏多样
            if skip_rate > 0 and np.random.random() < skip_rate:
                continue
            
            # 计算当前块的平均亮度
            y_start = row * char_size
            y_end = min((row + 1) * char_size, ch)
            x_start = col * char_size
            x_end = min((col + 1) * char_size, cw)
            
            block = gray[y_start:y_end, x_start:x_end]
            block_mask = mask_region[y_start:y_end, x_start:x_end]
            
            # 只处理在手部区域内的块
            if cv2.countNonZero(block_mask) < (block_mask.size * 0.3):
                continue
            
            if block.size == 0:
                continue
            
            # 计算平均亮度
            avg_brightness = np.mean(block)
            
            # 合成器抖动影响（模式5/6无震荡）
            brightness_jitter = avg_brightness
            
            # 映射到ASCII字符
            base_char_idx = int(brightness_jitter / 255 * (len(chars_to_use) - 1))
            
            # 模式6：采样器模式 - 字符种类随采样器参数变化
            if sampler is not None and hasattr(sampler, 'drum_kits'):
                # TR-808音序器模式
                pos_variety = (row * 7 + col * 11 + int(phase_offsets[row, col] * 100)) % 8
                
                # 音序器参数影响字符种类
                step_variety = sampler.current_step % 8  # 当前步影响
                tempo_variety = int((sampler.tempo - 60) / 20) % 6  # 节奏影响
                pattern_variety = hash(sampler.current_pattern) % 5  # 模式影响
                swing_variety = int(sampler.swing * 10) % 4  # 摇摆影响
                
                # 组合偏移
                total_variety = pos_variety + step_variety + tempo_variety + pattern_variety + swing_variety
                char_idx = (base_char_idx + total_variety) % len(chars_to_use)
            elif sampler is not None:
                # 旧版采样器模式
                pos_variety = (row * 7 + col * 11 + int(phase_offsets[row, col] * 100)) % 8
                
                # 采样器参数影响字符种类
                note_variety = (getattr(sampler, 'base_note', 60) - 48) % 12  # 音高影响
                tempo_variety = int((sampler.tempo - 60) / 20) % 6  # 节奏影响
                scale_variety = hash(getattr(sampler, 'current_scale', 'pentatonic')) % 5  # 音阶影响
                inst_variety = hash(getattr(sampler, 'current_instrument', 'piano')) % 4  # 音色影响
                
                # 组合偏移
                total_variety = pos_variety + note_variety + tempo_variety + scale_variety + inst_variety
                char_idx = (base_char_idx + total_variety) % len(chars_to_use)
            # 模式5：多种字符显示，种类随合成器参数变化
            elif synth is not None:
                # 基于位置的多样性（使用模运算确保在有效范围内）
                pos_variety = (row * 7 + col * 11 + int(phase_offsets[row, col] * 100)) % 8
                
                # 合成器参数影响字符种类（更灵敏）
                freq_variety = int((synth.frequency - 80) / 50) % 6  # 频率影响
                lfo_variety = int(synth.lfo_rate * 2) % 5  # LFO速率影响
                envelope_variety = int(synth.envelope_target * 10) % 4  # 包络影响
                
                # 组合偏移（确保多样性且不会超出范围太多）
                total_variety = pos_variety + freq_variety + lfo_variety + envelope_variety
                
                # 使用模运算确保字符种类多样性
                char_idx = (base_char_idx + total_variety) % len(chars_to_use)
            else:
                char_idx = base_char_idx
            
            char_idx = max(0, min(len(chars_to_use) - 1, char_idx))
            
            # 字符颜色：根据模式选择（降低明度与饱和度）
            if sampler is not None and inst_colors is not None:
                # 模式6：音色颜色渐变（降低明度与饱和度）
                brightness_factor = brightness_jitter / 255.0
                base_color = inst_colors['base']
                bright_color = inst_colors['bright']
                # 降低明度（乘以0.6）和饱和度（混合灰色）
                blue_val = int((base_color[0] + (bright_color[0] - base_color[0]) * brightness_factor) * 0.5)
                green_val = int((base_color[1] + (bright_color[1] - base_color[1]) * brightness_factor) * 0.5)
                red_val = int((base_color[2] + (bright_color[2] - base_color[2]) * brightness_factor) * 0.5)
                # 降低饱和度：混合灰色
                gray_mix = int((blue_val + green_val + red_val) / 3)
                blue_val = int(blue_val * 0.5 + gray_mix * 0.5)
                green_val = int(green_val * 0.5 + gray_mix * 0.5)
                red_val = int(red_val * 0.5 + gray_mix * 0.5)
                gradient_color = (blue_val, green_val, red_val)
            else:
                # 模式3/5：蓝色渐变（降低明度与饱和度）
                blue_val = int((50 + brightness_jitter * 0.4) * 0.5)  # 降低明度
                green_val = int((25 + brightness_jitter * 0.3) * 0.5)
                red_val = int((5 + brightness_jitter * 0.2) * 0.5)
                # 降低饱和度：混合灰色
                gray_mix = int((blue_val + green_val + red_val) / 3)
                blue_val = int(blue_val * 0.5 + gray_mix * 0.5)
                green_val = int(green_val * 0.5 + gray_mix * 0.5)
                red_val = int(red_val * 0.5 + gray_mix * 0.5)
                gradient_color = (blue_val, green_val, red_val)
            
            # 获取字符
            char = chars_to_use[char_idx]
            
            # 计算绘制位置
            draw_x = x + x_start + char_size // 2
            draw_y = y + y_start + char_size
            
            # 绘制ASCII字符
            font_scale = char_size / 18 if (synth is not None or sampler is not None) else char_size / 20
            cv2.putText(result, char, (draw_x, draw_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       gradient_color, 1, cv2.LINE_AA)
    
    return result

def extract_8_dominant_colors(image):
    """提取图像中像素数量最多的8种主色调（极速版）"""
    h, w = image.shape[:2]
    
    # 缩小图像以加快处理速度
    small = cv2.resize(image, (w // 12, h // 12))
    
    # 转换为RGB
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    
    # 重塑为像素列表
    pixels = rgb.reshape(-1, 3)
    
    # 随机采样以加快速度
    if len(pixels) > 500:
        indices = np.random.choice(len(pixels), 500, replace=False)
        pixels = pixels[indices]
    
    # 转换为float32
    pixels = np.float32(pixels)
    
    # K-means聚类 - 极速参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 3.0)
    _, labels, centers = cv2.kmeans(pixels, 8, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    
    # 统计每个聚类的像素数量
    unique, counts = np.unique(labels, return_counts=True)
    
    # 按像素数量排序（从多到少）
    sorted_indices = np.argsort(-counts)
    
    # 转换为整数并按像素数量排序
    centers = np.uint8(centers)
    sorted_centers = centers[sorted_indices]
    
    return sorted_centers

def smooth_color_centers(new_centers, history):
    """平滑颜色采样，减缓颜色过渡速度"""
    if not history:
        return new_centers
    
    # 将历史颜色转换为数组
    history_array = np.array(list(history), dtype=np.float32)  # (n, 8, 3)
    new_array = np.array(new_centers, dtype=np.float32)  # (8, 3)
    
    # 计算平滑后的颜色
    smoothed = np.zeros_like(new_array)
    
    for i in range(8):
        current = new_array[i]
        
        # 在历史中找到最相近的颜色
        min_dist = float('inf')
        best_match = current.copy()
        
        for hist in history_array:
            for j in range(8):
                dist = np.sum((hist[j] - current) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = hist[j]
        
        # 平滑混合：50%新颜色 + 50%历史颜色，减缓过渡
        if min_dist < 8000:  # 放宽匹配阈值
            smoothed[i] = current * 0.5 + best_match * 0.5
        else:
            smoothed[i] = current
    
    return np.uint8(smoothed)

def create_soft_color_palette(centers):
    """根据主色调创建生动鲜艳的调色板"""
    palette = []
    
    # 明暗梯度 - 保持层次感
    factors = [0.30, 0.42, 0.54, 0.66, 0.78, 0.90, 1.02, 1.15]
    
    for color in centers:
        r, g, b = color
        
        # 计算原始颜色的亮度和饱和度
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        
        for factor in factors:
            # 根据亮度调整
            if luminance < 80:
                # 暗色系：提升亮度
                adjusted_factor = factor * 1.15 + 0.1
            elif luminance < 150:
                # 中间调：保持层次
                adjusted_factor = factor * 1.05 + 0.05
            else:
                # 亮色系：保持高光
                adjusted_factor = factor
            
            # 大幅增强饱和度，保持画面生动
            if saturation < 0.5:
                # 低饱和度：大幅提升
                sat_boost = 1.3
            elif saturation < 0.7:
                # 中等饱和度：适度提升
                sat_boost = 1.15
            else:
                # 高饱和度：保持
                sat_boost = 1.0
            
            # 应用调整
            new_r = int(min(255, max(0, r * adjusted_factor)))
            new_g = int(min(255, max(0, g * adjusted_factor)))
            new_b = int(min(255, max(0, b * adjusted_factor)))
            
            # 增强饱和度
            gray_val = 0.299 * new_r + 0.587 * new_g + 0.114 * new_b
            new_r = int(gray_val + (new_r - gray_val) * sat_boost)
            new_g = int(gray_val + (new_g - gray_val) * sat_boost)
            new_b = int(gray_val + (new_b - gray_val) * sat_boost)
            
            # 确保在有效范围内
            new_r = min(255, max(0, new_r))
            new_g = min(255, max(0, new_g))
            new_b = min(255, max(0, new_b))
            
            palette.append((new_b, new_g, new_r))  # BGR格式
    
    return palette

def apply_soft_color_mapping_fast(image, centers):
    """应用生动鲜艳的色彩映射到图像（含输出滤波和色彩增强）"""
    h, w = image.shape[:2]
    
    # 创建调色板
    palette = create_soft_color_palette(centers)
    palette = np.array(palette, dtype=np.uint8)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 转换为RGB用于颜色匹配
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 预计算距离矩阵 - 使用广播优化
    centers_array = np.array(centers, dtype=np.float32)  # (8, 3)
    
    # 重塑为二维数组
    pixels = rgb.reshape(-1, 3)  # (N, 3)
    
    # 使用广播计算距离 - 更高效
    diff = pixels[:, np.newaxis, :] - centers_array[np.newaxis, :, :]  # (N, 8, 3)
    distances = np.sum(diff ** 2, axis=2)  # (N, 8)
    
    # 找到最近的主色调索引
    color_indices = np.argmin(distances, axis=1)
    
    # 计算明度级别
    gray_flat = gray.reshape(-1).astype(np.float32)
    levels = (gray_flat / 255 * 7).astype(np.int32)
    levels = np.clip(levels, 0, 7)
    
    # 计算调色板索引
    palette_indices = color_indices * 8 + levels
    
    # 应用调色板
    result_flat = palette[palette_indices]
    result = result_flat.reshape(h, w, 3)
    
    # 输出滤波：轻微高斯模糊减少噪点
    result = cv2.GaussianBlur(result, (3, 3), 0.5)
    
    # 色彩增强：提升对比度和饱和度
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)  # 饱和度+15%
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)  # 亮度+5%
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result

def apply_hand_mosaic(image, contour, synth, hand_area):
    """为手部添加马赛克效果"""
    if contour is None:
        return image
    
    h, w = image.shape[:2]
    result = image.copy()
    
    # 获取合成器参数
    frequency = getattr(synth, 'frequency', 440.0)
    lfo_rate = getattr(synth, 'lfo_rate', 1.0)
    lfo_depth = getattr(synth, 'lfo_depth', 0.3)
    envelope = getattr(synth, 'envelope_target', 0.0)
    
    # 合成器参数影响马赛克大小
    blend_intensity = min(1.0, max(0.1, envelope))
    freq_factor = (frequency - 80) / 700.0
    freq_factor = np.clip(freq_factor, 0.0, 1.0)
    
    base_block_size = 2 + int(blend_intensity * 6) + int(freq_factor * 4)
    
    # 增加错乱效果 - 使用LFO参数
    chaos_factor = int(lfo_depth * 4) + int(lfo_rate % 4 * 1)
    
    # 获取手部边界框
    x, y, cw, ch = cv2.boundingRect(contour)
    
    # 记录当前位置并平滑
    global mosaic_position_history
    current_pos = (x, y)
    mosaic_position_history.append(current_pos)
    
    # 计算平滑后的位置（增加黏度，使用加权平均）
    if len(mosaic_position_history) > 1:
        # 加权平均，给最近的历史更高权重，但整体更慢
        weights = np.linspace(0.1, 1.0, len(mosaic_position_history))
        weights = weights / np.sum(weights)
        
        positions = np.array(mosaic_position_history)
        smooth_x = int(np.sum(positions[:, 0] * weights))
        smooth_y = int(np.sum(positions[:, 1] * weights))
        x, y = smooth_x, smooth_y
    
    # 扩展边界框
    padding = 8
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(w, x + cw + padding)
    y_end = min(h, y + ch + padding)
    
    hand_w = x_end - x_start
    hand_h = y_end - y_start
    
    if hand_w <= 0 or hand_h <= 0:
        return result
    
    # 提取手部区域
    hand_region = result[y_start:y_end, x_start:x_end].copy()
    
    # 应用马赛克 - 使用更小的处理区域提高性能
    block_size = max(4, base_block_size + chaos_factor)
    
    # 像素化手部区域
    small_w = max(2, hand_w // block_size)
    small_h = max(2, hand_h // block_size)
    
    small = cv2.resize(hand_region, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (hand_w, hand_h), interpolation=cv2.INTER_NEAREST)
    
    # 增加错乱颜色偏移（根据LFO）
    if int(lfo_rate) % 2 == 0:
        shift_amount = chaos_factor
        if shift_amount > 0 and mosaic.shape[1] > shift_amount * 2:
            mosaic_shifted = mosaic.copy()
            mosaic_shifted[:, :, 0] = np.roll(mosaic_shifted[:, :, 0], shift_amount, axis=1)
            mosaic_shifted[:, :, 2] = np.roll(mosaic_shifted[:, :, 2], -shift_amount, axis=1)
            mosaic = cv2.addWeighted(mosaic, 0.6, mosaic_shifted, 0.4, 0)
    
    # 创建边缘流体效果
    mask_region = np.zeros((hand_h, hand_w), dtype=np.uint8)
    # 调整contour坐标到相对于hand_region的坐标
    contour_shifted = contour.copy()
    contour_shifted[:, :, 0] -= x_start
    contour_shifted[:, :, 1] -= y_start
    cv2.drawContours(mask_region, [contour_shifted], 0, 255, -1)
    
    # 边缘平滑 - 使用形态学操作和高斯模糊（增加黏度）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, kernel)
    mask_region = cv2.GaussianBlur(mask_region, (21, 21), 3)
    
    # 扩展一点边缘
    mask_region = mask_region.astype(np.float32) / 255.0
    
    # 使用alpha混合
    for c in range(3):
        result[y_start:y_end, x_start:x_end, c] = (
            result[y_start:y_end, x_start:x_end, c] * (1 - mask_region) + 
            mosaic[:, :, c] * mask_region
        )
    
    # 提升整体画面的饱和度和对比度
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 提升对比度
    alpha = 1.1
    beta = 5
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    return result

def apply_pioneer_art_effect(image, synth, contour, prev_gray):
    """先锋艺术效果：光流色彩爆发与动态分割（自然柔和版）"""
    h, w = image.shape[:2]
    
    result = image.copy()
    
    # 获取合成器参数
    if synth is not None:
        frequency = getattr(synth, 'frequency', 440.0)
        lfo_rate = getattr(synth, 'lfo_rate', 1.0)
        lfo_depth = getattr(synth, 'lfo_depth', 0.3)
        envelope = getattr(synth, 'envelope_target', 0.0)
    else:
        frequency = 440.0
        lfo_rate = 1.0
        lfo_depth = 0.3
        envelope = 0.0
    
    # 当前帧灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 光流检测 - 使用降采样提高性能
    flow = None
    if prev_gray is not None:
        try:
            # 降采样图像
            scale_factor = 0.4
            small_prev = cv2.resize(prev_gray, (int(w * scale_factor), int(h * scale_factor)))
            small_gray = cv2.resize(gray, (int(w * scale_factor), int(h * scale_factor)))
            
            # 计算光流 - 更平滑的参数
            small_flow = cv2.calcOpticalFlowFarneback(small_prev, small_gray, None, 0.5, 5, 15, 3, 7, 1.5, 0)
            
            # 上采样回原尺寸
            flow = cv2.resize(small_flow, (w, h))
            flow[:, :, 0] /= scale_factor
            flow[:, :, 1] /= scale_factor
        except:
            pass
    
    # 创建艺术效果
    # 1. 基础层：柔和色彩增强
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 饱和度和亮度柔和增强
    sat_boost = 1.2 + envelope * 0.6
    val_boost = 1.05 + envelope * 0.2
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_boost, 0, 255)
    
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 2. 如果有光流，添加柔和的运动色彩爆发
    if flow is not None:
        # 计算光流强度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 光流先做高斯模糊，更自然
        mag = cv2.GaussianBlur(mag, (15, 15), 3)
        
        # 合成器参数影响爆发强度 - 更柔和
        flow_intensity = np.clip(mag * (0.3 + lfo_depth * 0.8), 0, 30)
        
        # 创建色彩爆发层 - 向量化计算
        ang_deg = ang * 180 / np.pi
        
        # 角度也做平滑
        ang_smooth = cv2.GaussianBlur(ang_deg, (15, 15), 3)
        
        # 创建色相映射 - 调整到OpenCV HSV范围 0-179
        hue_map = np.zeros((h, w), dtype=np.uint8)
        
        # 根据角度分配色相 - 转换到0-179
        mask = ang_smooth < 60
        hue_map[mask] = 0
        mask = (ang_smooth >= 60) & (ang_smooth < 120)
        hue_map[mask] = 30
        mask = (ang_smooth >= 120) & (ang_smooth < 180)
        hue_map[mask] = 60
        mask = (ang_smooth >= 180) & (ang_smooth < 240)
        hue_map[mask] = 90
        mask = (ang_smooth >= 240) & (ang_smooth < 300)
        hue_map[mask] = 120
        mask = ang_smooth >= 300
        hue_map[mask] = 150
        
        # 创建HSV爆发层 - 柔和饱和度
        burst_hsv = np.zeros((h, w, 3), dtype=np.float32)
        burst_hsv[:, :, 0] = hue_map
        burst_hsv[:, :, 1] = 180
        burst_hsv[:, :, 2] = np.clip(flow_intensity * 3, 0, 200)
        
        # 转换为BGR
        burst_layer = cv2.cvtColor(burst_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 高斯模糊爆发层，更自然
        burst_layer = cv2.GaussianBlur(burst_layer, (7, 7), 2)
        
        # 只保留有强度的区域
        mask = flow_intensity > 0.5
        burst_masked = np.zeros_like(burst_layer)
        burst_masked[mask] = burst_layer[mask]
        
        # 混合爆发层 - 更柔和的混合
        blend_factor = min(0.35, 0.1 + envelope * 0.25)
        result = cv2.addWeighted(result, 1.0 - blend_factor, burst_masked, blend_factor, 0)
    
    # 3. 柔和边缘高亮
    if contour is not None:
        # 创建手部高亮
        highlight_mask = np.zeros((h, w), dtype=np.float32)
        cv2.drawContours(highlight_mask, [contour], 0, 1.0, -1)
        
        # 更柔的高斯模糊
        highlight_mask = cv2.GaussianBlur(highlight_mask, (51, 51), 10)
        
        # 高亮色 - 随LFO柔和变化
        hue_shift = int(lfo_rate * 15 % 180)
        highlight_color = np.array([
            255 * (0.5 + 0.3 * np.sin(hue_shift * np.pi / 180)),
            255 * (0.5 + 0.3 * np.sin((hue_shift + 120) * np.pi / 180)),
            255 * (0.5 + 0.3 * np.sin((hue_shift + 240) * np.pi / 180))
        ])
        
        # 应用柔和高亮
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c].astype(np.float32) * (1.0 - highlight_mask * 0.15) +
                highlight_color[c] * highlight_mask * (0.3 + envelope * 0.2),
                0, 255
            ).astype(np.uint8)
    
    # 4. 对比度和饱和度柔和增强
    alpha = 1.1 + envelope * 0.15
    beta = 5 + int(envelope * 10)
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    return result, gray.copy()

def draw_color_palette_ui(image, centers):
    """绘制颜色调色板UI"""
    h, w = image.shape[:2]
    
    # 在右上角绘制调色板
    num_colors = len(centers)
    palette_h = 15
    palette_w = 20
    start_x = w - (num_colors * palette_w + 10)
    start_y = 70
    
    # 绘制背景
    cv2.rectangle(image, (start_x - 5, start_y - 5), 
                 (w - 5, start_y + palette_h + 5), (0, 0, 0), -1)
    
    # 绘制每种颜色
    for i, color in enumerate(centers):
        px = start_x + i * palette_w
        # color是RGB格式，转换为BGR
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.rectangle(image, (px, start_y), (px + palette_w - 1, start_y + palette_h), bgr, -1)
    
    return image

def draw_ui(image, fps, current_mode, color_centers=None, synth_params=None, sampler_params=None, sampler=None, synth_visual_params=None, synth_art_pioneer_params=None):
    h, w = image.shape[:2]
    
    font_scale = 0.5 if h < 600 else 0.85
    font_scale_small = 0.4 if h < 600 else 0.6
    
    # 顶部UI
    cv2.rectangle(image, (0, 0), (w, int(h * 0.08)), (0, 0, 0), -1)
    cv2.putText(image, 'Hand Gesture', (10, int(h * 0.055)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    cv2.putText(image, f'FPS:{int(fps)}', (w - 80, int(h * 0.055)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
    
    # 合成器模式特殊UI - 信息显示在顶部
    if current_mode == MODE_SYNTH and synth_params is not None:
        freq, lfo_rate, lfo_depth, detune, lfo_type, hue, sat, val = synth_params
        lfo_names = ['Sin', 'Sqr', 'Tri']
        # 在顶部右侧显示合成器参数
        info_x = w - 300
        cv2.putText(image, f'Freq:{int(freq)}Hz Detune:{detune:.1f}', (info_x, int(h * 0.035)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
        cv2.putText(image, f'LFO:{lfo_rate:.1f}Hz {lfo_names[lfo_type]} D:{lfo_depth:.2f}', (info_x, int(h * 0.065)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (100, 255, 200), 1)
    
    # 采样器模式特殊UI - TR-808音序器显示
    if current_mode == MODE_SAMPLER and sampler_params is not None:
        tempo, current_step, pattern_name, editing_drum, swing, step_count = sampler_params[:6]
        variation = sampler_params[6] if len(sampler_params) > 6 else 0.3
        sound_kit = sampler_params[7] if len(sampler_params) > 7 else '808'
        # 在顶部右侧显示音序器参数
        info_x = w - 400
        cv2.putText(image, f'{sound_kit.upper()} {tempo}BPM Step:{current_step+1}/{step_count}', (info_x, int(h * 0.035)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
        cv2.putText(image, f'{pattern_name.upper()} Swing:{swing:.0%} Kit:{sound_kit}', (info_x, int(h * 0.065)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (100, 255, 200), 1)
        
        # 绘制16步音序器网格
        grid_x = w - 380
        grid_y = int(h * 0.09)
        step_width = 20
        step_height = 12
        
        # 获取当前模式
        if hasattr(sampler, 'get_pattern_for_display'):
            pattern = sampler.get_pattern_for_display()
            drum_types = ['kick', 'snare', 'clap', 'hihat_closed', 'hihat_open']
            drum_colors = [(255, 100, 100), (100, 255, 100), (255, 200, 100), (100, 200, 255), (200, 100, 255)]
            
            for row, drum_type in enumerate(drum_types):
                if drum_type in pattern:
                    for step in range(16):
                        x = grid_x + step * step_width
                        y = grid_y + row * (step_height + 2)
                        
                        # 背景
                        if step == current_step:
                            cv2.rectangle(image, (x, y), (x + step_width - 2, y + step_height), (80, 80, 80), -1)
                        else:
                            cv2.rectangle(image, (x, y), (x + step_width - 2, y + step_height), (40, 40, 40), -1)
                        
                        # 激活的步骤（带强度显示）
                        intensity = pattern[drum_type][step]
                        if intensity > 0.1:
                            color = tuple(int(c * min(1.0, intensity)) for c in drum_colors[row])
                            cv2.rectangle(image, (x, y), (x + step_width - 2, y + step_height), color, -1)
                        
                        # 边框
                        cv2.rectangle(image, (x, y), (x + step_width - 2, y + step_height), (100, 100, 100), 1)
    
    # 合成视觉模式特殊UI - 与合成器模式相同
    if current_mode == MODE_SYNTH_VISUAL and synth_visual_params is not None:
        freq, lfo_rate, lfo_depth, detune, lfo_type, hue, sat, val = synth_visual_params
        lfo_names = ['Sin', 'Sqr', 'Tri']
        # 在顶部右侧显示合成器参数
        info_x = w - 300
        cv2.putText(image, f'Freq:{int(freq)}Hz Detune:{detune:.1f}', (info_x, int(h * 0.035)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
        cv2.putText(image, f'LFO:{lfo_rate:.1f}Hz {lfo_names[lfo_type]} D:{lfo_depth:.2f}', (info_x, int(h * 0.065)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (100, 255, 200), 1)
    
    # 先锋艺术模式特殊UI - 与合成器模式相同
    if current_mode == MODE_ART_PIONEER and synth_art_pioneer_params is not None:
        freq, lfo_rate, lfo_depth, detune, lfo_type, hue, sat, val = synth_art_pioneer_params
        lfo_names = ['Sin', 'Sqr', 'Tri']
        # 在顶部右侧显示合成器参数
        info_x = w - 300
        cv2.putText(image, f'Freq:{int(freq)}Hz Detune:{detune:.1f}', (info_x, int(h * 0.035)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
        cv2.putText(image, f'LFO:{lfo_rate:.1f}Hz {lfo_names[lfo_type]} D:{lfo_depth:.2f}', (info_x, int(h * 0.065)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 100, 200), 1)
    
    # 底部UI
    cv2.rectangle(image, (0, int(h * 0.92)), (w, h), (0, 0, 0), -1)
    cv2.putText(image, 'q:Quit 1:Mouse 2:Gesture 3:ASCII 4:Color 5:Synth 6:Sampler 7:SynthVisual 8:PioneerArt r:Reset', (10, int(h * 0.96)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (180, 180, 180), 1)
    
    if current_mode == MODE_ASCII:
        cv2.putText(image, 'ASCII Art Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
    elif current_mode == MODE_COLOR:
        cv2.putText(image, '8-Color Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 100, 255), 1)
        if color_centers is not None:
            image = draw_color_palette_ui(image, color_centers)
    elif current_mode == MODE_SYNTH:
        cv2.putText(image, 'Synthesizer Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 100, 100), 1)
    elif current_mode == MODE_SAMPLER:
        cv2.putText(image, 'TR-808 Sequencer Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (100, 255, 255), 1)
    elif current_mode == MODE_SYNTH_VISUAL:
        cv2.putText(image, 'Synth Visual Blend Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 100, 255), 1)
    elif current_mode == MODE_ART_PIONEER:
        cv2.putText(image, 'Pioneer Art Mode', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
    else:
        mode_color = (0, 255, 0) if current_mode == MODE_GESTURE else (0, 150, 255)
        mode_text = 'Gesture' if current_mode == MODE_GESTURE else 'Mouse'
        cv2.putText(image, f'Mode: {mode_text}', (10, int(h * 0.90)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, mode_color, 1)
    
    return image

def main():
    global current_mode
    
    print("Initializing...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # 尝试设置高分辨率，然后获取摄像头实际支持的规格
    # 先尝试常见的高分辨率
    resolutions_to_try = [
        (1920, 1080),
        (1280, 720),
        (640, 480),
    ]
    
    for width, height in resolutions_to_try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w >= width * 0.9:  # 如果实际分辨率接近目标
            break
    
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 获取摄像头实际规格
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera specs: {actual_w}x{actual_h} @ {actual_fps}fps")
    
    # 根据分辨率自动适配参数
    global scale_factor
    if actual_w >= 1280:
        # 高分辨率：使用更大的处理区域
        scale_factor = actual_w / 640
        print(f"High resolution detected, scale factor: {scale_factor:.2f}")
    else:
        scale_factor = 1.0
    
    print("Starting... Press 'q' to quit or click X to close")
    
    prev_time = 0
    fps_smooth = 30
    
    # 颜色模式变量
    color_centers = None
    color_history = deque(maxlen=10)
    
    # 合成器模式变量
    global synth
    synth_params = None
    
    # 采样器模式变量
    global sampler
    sampler_params = None
    
    # 合成视觉模式变量
    global synth_visual
    synth_visual_params = None
    
    # 先锋艺术模式变量
    global synth_art_pioneer
    global prev_frame_gray
    synth_art_pioneer_params = None
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        image = cv2.flip(image, 1)
        
        h, w = image.shape[:2]
        
        contour, defects, mask = process_hand(image)
        
        display_image = image.copy()
        
        gesture_name, finger_count = recognize_gesture(contour, defects)
        
        hand_center = None
        
        if contour is not None:
            # 手势模式：只勾勒手部线条
            if current_mode == MODE_GESTURE:
                cv2.drawContours(display_image, [contour], 0, (0, 255, 0), 2)
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # 使用平滑位置
            smooth_x, smooth_y, smooth_cw, smooth_ch = smooth_position(x, y, cw, ch)
            
            hand_center = (smooth_x + smooth_cw // 2, smooth_y + smooth_ch // 2)
            
            if current_mode == MODE_GESTURE:
                if gesture_name:
                    cv2.putText(display_image, gesture_name, (smooth_x, smooth_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            elif current_mode == MODE_MOUSE:
                if hand_center:
                    screen_width, screen_height = pyautogui.size()
                    screen_x = int((hand_center[0] / w) * screen_width)
                    screen_y = int((hand_center[1] / h) * screen_height)
                    pyautogui.moveTo(screen_x, screen_y, duration=0.05)
                    cv2.circle(display_image, hand_center, 8, (0, 200, 255), -1)
        
        # ASCII艺术模式
        if current_mode == MODE_ASCII and contour is not None:
            # 确保mask是有效的numpy数组
            if mask is not None and isinstance(mask, np.ndarray) and mask.size > 0:
                display_image = create_ascii_art(display_image, contour, mask)
        
        # 颜色调色板模式（8色）
        if current_mode == MODE_COLOR:
            # 每帧更新颜色（30Hz），实时响应
            new_colors = extract_8_dominant_colors(display_image)
            
            # 平滑颜色采样
            color_centers = smooth_color_centers(new_colors, color_history)
            
            # 更新历史
            color_history.append(color_centers.copy())
            
            # 应用生动鲜艳色彩映射
            display_image = apply_soft_color_mapping_fast(display_image, color_centers)
        
        # 合成器模式
        if current_mode == MODE_SYNTH:
            # 启动合成器
            if synth is None:
                synth = AudioSynthesizer()
                synth.start()
            
            # 使用ASCII艺术作为背景（传入合成器影响字符）
            if contour is not None:
                # 先更新合成器参数
                x, y, cw, ch = cv2.boundingRect(contour)
                hand_y = y / h
                hand_x = x / w
                hand_area = cw * ch
                
                new_colors = extract_8_dominant_colors(display_image)
                synth.update_from_gesture(hand_y, hand_area, hand_x, new_colors, finger_count)
                
                # 然后创建受合成器影响的ASCII艺术
                if mask is not None and isinstance(mask, np.ndarray) and mask.size > 0:
                    display_image = create_ascii_art(display_image, contour, mask, synth, None)
                
                # 显示参数
                synth_params = (synth.frequency, synth.lfo_rate, synth.lfo_depth, synth.detune, synth.lfo_type, synth.color_hue, synth.color_sat, synth.envelope_target)
            else:
                # 没有检测到手时，降低音量
                if synth is not None:
                    synth.envelope_target = 0.0
        
        # 停止合成器（切换模式时）
        if current_mode != MODE_SYNTH and synth is not None:
            synth.stop()
            synth = None
        
        # 采样器模式（模式6）- TR-808音序器
        if current_mode == MODE_SAMPLER:
            # 启动音序器
            if sampler is None:
                sampler = TR808Sequencer()
                sampler.start()
            
            if contour is not None:
                # 更新音序器参数
                x, y, cw, ch = cv2.boundingRect(contour)
                hand_y = y / h
                hand_x = x / w
                hand_area = cw * ch
                
                # 提取颜色用于音色调整
                new_colors = extract_8_dominant_colors(display_image)
                
                # 更新手势参数
                sampler.update_from_gesture(hand_y, hand_x, hand_area, finger_count)
                
                # 更新颜色参数
                sampler.update_from_color(new_colors)
                
                # 创建ASCII艺术背景
                if mask is not None and isinstance(mask, np.ndarray) and mask.size > 0:
                    display_image = create_ascii_art(display_image, contour, mask, None, sampler)
                
                # 获取显示信息
                info = sampler.get_display_info()
                sampler_params = (info['tempo'], info['current_step'], info['current_pattern'], 
                                 info['editing_drum'], info['swing'], info['step_count'], info['variation'], info['sound_kit'])
        else:
            # 停止采样器（切换模式时）
            if sampler is not None:
                sampler.stop()
                sampler = None
        
        # 合成视觉模式（模式7）- 手部马赛克效果
        if current_mode == MODE_SYNTH_VISUAL:
            # 启动合成器
            if synth_visual is None:
                synth_visual = AudioSynthesizer()
                synth_visual.start()
            
            # 记录手部信息
            hand_area_val = 0
            if contour is not None:
                x, y, cw, ch = cv2.boundingRect(contour)
                hand_y = y / h
                hand_x = x / w
                hand_area_val = cw * ch
                
                # 提取颜色用于音色调整
                new_colors = extract_8_dominant_colors(display_image)
                
                # 更新合成器参数（与模式5相同）
                synth_visual.update_from_gesture(hand_y, hand_area_val, hand_x, new_colors, finger_count)
                
                # 显示参数
                synth_visual_params = (synth_visual.frequency, synth_visual.lfo_rate, synth_visual.lfo_depth, 
                                     synth_visual.detune, synth_visual.lfo_type, synth_visual.color_hue, 
                                     synth_visual.color_sat, synth_visual.envelope_target)
            else:
                # 没有检测到手时，降低音量
                if synth_visual is not None:
                    synth_visual.envelope_target = 0.0
            
            # 应用手部马赛克效果
            display_image = apply_hand_mosaic(display_image, contour, synth_visual, hand_area_val)
        else:
            # 停止视觉合成器（切换模式时）
            if synth_visual is not None:
                synth_visual.stop()
                synth_visual = None
        
        # 先锋艺术模式（模式8）- 光流色彩爆发
        if current_mode == MODE_ART_PIONEER:
            # 启动合成器
            if synth_art_pioneer is None:
                synth_art_pioneer = AudioSynthesizer()
                synth_art_pioneer.start()
            
            # 记录手部信息
            if contour is not None:
                x, y, cw, ch = cv2.boundingRect(contour)
                hand_y = y / h
                hand_x = x / w
                hand_area_val = cw * ch
                
                # 提取颜色用于音色调整
                new_colors = extract_8_dominant_colors(display_image)
                
                # 更新合成器参数
                synth_art_pioneer.update_from_gesture(hand_y, hand_area_val, hand_x, new_colors, finger_count)
                
                # 显示参数
                synth_art_pioneer_params = (synth_art_pioneer.frequency, synth_art_pioneer.lfo_rate, 
                                           synth_art_pioneer.lfo_depth, synth_art_pioneer.detune, 
                                           synth_art_pioneer.lfo_type, synth_art_pioneer.color_hue, 
                                           synth_art_pioneer.color_sat, synth_art_pioneer.envelope_target)
            else:
                # 没有检测到手时，降低音量
                if synth_art_pioneer is not None:
                    synth_art_pioneer.envelope_target = 0.0
            
            # 应用先锋艺术效果
            display_image, new_gray = apply_pioneer_art_effect(display_image, synth_art_pioneer, contour, prev_frame_gray)
            prev_frame_gray = new_gray
        else:
            # 停止先锋艺术合成器（切换模式时）
            if synth_art_pioneer is not None:
                synth_art_pioneer.stop()
                synth_art_pioneer = None
                prev_frame_gray = None
        
        curr_time = time.time()
        frame_time = curr_time - prev_time
        if frame_time > 0:
            fps = 1 / frame_time
            fps_smooth = fps_smooth * 0.9 + fps * 0.1
        prev_time = curr_time
        
        display_image = draw_ui(display_image, fps_smooth, current_mode, color_centers, synth_params, sampler_params, sampler, synth_visual_params, synth_art_pioneer_params)
        
        cv2.imshow('Hand Gesture Recognition', display_image)
        
        # 检查窗口是否被关闭（点击叉叉）
        if cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('x'):
            break
        elif key == ord('1'):
            current_mode = MODE_MOUSE
        elif key == ord('2'):
            current_mode = MODE_GESTURE
        elif key == ord('3'):
            current_mode = MODE_ASCII
        elif key == ord('4'):
            current_mode = MODE_COLOR
            color_centers = None
        elif key == ord('5'):
            current_mode = MODE_SYNTH
        elif key == ord('6'):
            current_mode = MODE_SAMPLER
        elif key == ord('7'):
            current_mode = MODE_SYNTH_VISUAL
        elif key == ord('8'):
            current_mode = MODE_ART_PIONEER
        elif key == ord('r'):
            # 重置所有历史
            gesture_history.clear()
            hand_position_history.clear()
            hand_size_history.clear()
            print("History reset")
    
    print("Releasing resources...")
    # 停止合成器
    if synth is not None:
        synth.stop()
    # 停止采样器
    if sampler is not None:
        sampler.stop()
    # 停止视觉合成器
    if synth_visual is not None:
        synth_visual.stop()
    # 停止先锋艺术合成器
    if synth_art_pioneer is not None:
        synth_art_pioneer.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
