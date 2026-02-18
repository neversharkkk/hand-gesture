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
current_mode = MODE_GESTURE

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
    """TR-808风格音序器采样器 - 高品质版"""
    def __init__(self, sample_rate=48000, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.running = False
        self.thread = None
        
        # 高品质噪声表（预生成，避免实时随机噪声）
        self._noise_table_size = sample_rate * 2
        self._noise_table = np.random.uniform(-1, 1, self._noise_table_size).astype(np.float32)
        self._noise_index = 0
        
        # 音序器参数（带平滑过渡）
        self.tempo = 120
        self.tempo_target = 120
        self.tempo_smoothing = 0.02
        
        self.step_count = 16
        self.current_step = 0
        self.step_phase = 0.0
        
        # 风格混合系统
        self.style_weights = {
            'classic': 1.0, 'hiphop': 0.0, 'house': 0.0,
            'techno': 0.0, 'breakbeat': 0.0, 'dubstep': 0.0,
            'jungle': 0.0, 'ambient': 0.0,
        }
        self.style_target = 'classic'
        self.style_transition_speed = 0.05
        
        # 808音色定义（扩展）
        self.drum_kits = {
            'kick': {'name': '底鼓', 'short': 'KD'},
            'snare': {'name': '军鼓', 'short': 'SD'},
            'clap': {'name': '拍手', 'short': 'CP'},
            'hihat_closed': {'name': '闭镲', 'short': 'CH'},
            'hihat_open': {'name': '开镲', 'short': 'OH'},
            'tom_high': {'name': '高通', 'short': 'HT'},
            'tom_mid': {'name': '中通', 'short': 'MT'},
            'tom_low': {'name': '低通', 'short': 'LT'},
            'rimshot': {'name': '边击', 'short': 'RS'},
            'cowbell': {'name': '牛铃', 'short': 'CB'},
            'clave': {'name': '克拉维', 'short': 'CL'},
            'maracas': {'name': '沙锤', 'short': 'MA'},
            'conga_high': {'name': '高康加', 'short': 'CG'},
            'conga_low': {'name': '低康加', 'short': 'CL'},
            'cymbal': {'name': '镲片', 'short': 'CY'},
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
        
        # 声音状态
        self.drum_envelopes = {k: 0.0 for k in self.drum_kits.keys()}
        self.drum_phases = {k: 0.0 for k in self.drum_kits.keys()}
        
        # 音量和效果（带平滑）
        self.master_volume = 0.65
        self.master_volume_target = 0.65
        self.drum_volumes = {k: 0.8 for k in self.drum_kits.keys()}
        self.drum_volumes_target = {k: 0.8 for k in self.drum_kits.keys()}
        
        self.swing = 0.0
        self.swing_target = 0.0
        
        # 混响和压缩
        self.reverb_buffer = np.zeros(int(sample_rate * 0.4))
        self.reverb_index = 0
        self.reverb_amount = 0.2
        self.reverb_amount_target = 0.2
        self.compression_ratio = 0.8
        
        # 延迟效果
        self.delay_buffer = np.zeros(int(sample_rate * 0.5))
        self.delay_index = 0
        self.delay_amount = 0.15
        self.delay_feedback = 0.3
        
        # 滤波器
        self.filter_state = 0.0
        self.filter_cutoff = 0.8
        self.filter_resonance = 0.3
        
        # 手势控制状态
        self.hand_y = 0.5
        self.hand_x = 0.5
        self.hand_area = 0.0
        self.finger_count = 0
        
        # 显示参数
        self.display_mode = 'pattern'
        
    def _smooth_parameter(self, current, target, speed=0.02):
        """平滑参数过渡"""
        return current + (target - current) * speed
    
    def _update_smooth_parameters(self):
        """更新所有平滑参数"""
        # BPM平滑
        self.tempo = self._smooth_parameter(self.tempo, self.tempo_target, self.tempo_smoothing)
        
        # 摇摆平滑
        self.swing = self._smooth_parameter(self.swing, self.swing_target, 0.03)
        
        # 音量平滑
        self.master_volume = self._smooth_parameter(self.master_volume, self.master_volume_target, 0.02)
        for k in self.drum_volumes:
            self.drum_volumes[k] = self._smooth_parameter(self.drum_volumes[k], self.drum_volumes_target[k], 0.02)
        
        # 混响平滑
        self.reverb_amount = self._smooth_parameter(self.reverb_amount, self.reverb_amount_target, 0.02)
        
        # 鼓参数平滑
        for drum, params in self.drum_params.items():
            if 'pitch_target' in params:
                params['pitch'] = self._smooth_parameter(params['pitch'], params['pitch_target'], 0.02)
            if 'decay_target' in params:
                params['decay'] = self._smooth_parameter(params['decay'], params['decay_target'], 0.02)
            if 'tone_target' in params:
                params['tone'] = self._smooth_parameter(params['tone'], params['tone_target'], 0.02)
            if 'snappy_target' in params:
                params['snappy'] = self._smooth_parameter(params['snappy'], params['snappy_target'], 0.02)
        
        # 风格权重平滑过渡
        for style in self.style_weights:
            if style == self.style_target:
                target = 1.0
            else:
                target = 0.0
            self.style_weights[style] = self._smooth_parameter(self.style_weights[style], target, self.style_transition_speed)
    
    def _blend_patterns(self):
        """混合多个风格模式"""
        blended = {k: [0.0]*16 for k in self.drum_kits.keys()}
        
        for style, weight in self.style_weights.items():
            if weight > 0.01 and style in self.patterns:
                pattern = self.patterns[style]
                for drum, steps in pattern.items():
                    for i in range(min(len(steps), 16)):
                        blended[drum][i] += steps[i] * weight
        
        # 添加随机变化
        if self.variation_intensity > 0:
            for drum in blended:
                for i in range(16):
                    if np.random.random() < self.variation_intensity * 0.1:
                        blended[drum][i] = min(1.0, blended[drum][i] + np.random.uniform(0, 0.3))
        
        return blended
    
    def _generate_variation(self):
        """生成节奏变奏（增强版）"""
        self.pattern_variation_counter += 1
        
        # 每4个循环添加一些变化
        if self.pattern_variation_counter % 4 == 0:
            # 随机添加一些打击乐
            extra_drums = ['tom_high', 'tom_mid', 'tom_low', 'rimshot', 'cowbell', 'clave', 'maracas', 'conga_high', 'conga_low']
            for drum in extra_drums:
                if np.random.random() < 0.25 * self.variation_intensity:
                    pos = np.random.randint(0, 16)
                    self.generated_pattern[drum][pos] = np.random.uniform(0.5, 1.0)
        
        # 每8个循环添加镲片
        if self.pattern_variation_counter % 8 == 0:
            if np.random.random() < 0.3:
                pos = np.random.randint(0, 16)
                self.generated_pattern['cymbal'][pos] = 0.7
        
        # 每16个循环清除变化
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
    
    def _generate_808_kick(self):
        """生成TR-808风格底鼓（增强版）"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes['kick'] > 0.01:
            params = self.drum_params['kick']
            pitch = params['pitch']
            decay = params['decay']
            tone = params['tone']
            
            for i in range(self.buffer_size):
                # 多层频率下滑
                freq1 = (45 + tone * 80) * pitch * np.exp(-self.drum_phases['kick'] * (2 + decay * 4))
                freq2 = (80 + tone * 60) * pitch * np.exp(-self.drum_phases['kick'] * (4 + decay * 6))
                freq1 = max(20, freq1)
                freq2 = max(30, freq2)
                
                # 主音层
                main_tone = np.sin(self.drum_phases['kick'] * 2 * np.pi * freq1 / self.sample_rate)
                # 次谐波层
                sub_tone = np.sin(self.drum_phases['kick'] * 2 * np.pi * freq2 / self.sample_rate) * 0.6
                # 点击音
                click = np.sin(self.drum_phases['kick'] * 2 * np.pi * 1200 / self.sample_rate) * np.exp(-self.drum_phases['kick'] * 80)
                # 额外 punch
                punch = np.sin(self.drum_phases['kick'] * 2 * np.pi * 200 / self.sample_rate) * np.exp(-self.drum_phases['kick'] * 30) * 0.3
                
                output[i] = (main_tone * 0.5 + sub_tone * 0.35 + click * 0.08 + punch * 0.07) * self.drum_envelopes['kick']
                self.drum_phases['kick'] += 1
                self.drum_envelopes['kick'] *= (0.9985 - decay * 0.002)
        return output * self.drum_volumes['kick']
    
    def _generate_808_snare(self):
        """生成TR-808风格军鼓（增强版）"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes['snare'] > 0.01:
            params = self.drum_params['snare']
            pitch = params['pitch']
            decay = params['decay']
            snappy = params['snappy']
            tone_val = params.get('tone', 0.5)
            
            # 噪声滤波状态
            if not hasattr(self, '_snare_noise_state'):
                self._snare_noise_state = 0.0
            
            for i in range(self.buffer_size):
                # 多层主体音调
                freq1 = 160 * pitch
                freq2 = 200 * pitch * (1 + tone_val * 0.3)
                freq3 = 320 * pitch
                
                tone1 = np.sin(self.drum_phases['snare'] * 2 * np.pi * freq1 / self.sample_rate)
                tone2 = np.sin(self.drum_phases['snare'] * 2 * np.pi * freq2 / self.sample_rate) * 0.4
                tone3 = np.sin(self.drum_phases['snare'] * 2 * np.pi * freq3 / self.sample_rate) * 0.2
                
                # 带通噪声层
                raw_noise = np.random.uniform(-1, 1)
                # 简单带通滤波
                self._snare_noise_state = 0.4 * self._snare_noise_state + 0.6 * raw_noise
                bpf_noise = self._snare_noise_state * snappy * 0.8
                
                # 高频噪声
                hpf_noise = (raw_noise - self._snare_noise_state) * snappy * 0.5
                
                output[i] = ((tone1 + tone2 + tone3) * (1 - snappy * 0.4) + bpf_noise + hpf_noise) * self.drum_envelopes['snare']
                self.drum_phases['snare'] += 1
                self.drum_envelopes['snare'] *= (0.996 - decay * 0.008)
        return output * self.drum_volumes['snare']
    
    def _generate_808_clap(self):
        """生成TR-808风格拍手（增强版）"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes['clap'] > 0.01:
            decay = self.drum_params['clap']['decay']
            tone_val = self.drum_params['clap'].get('tone', 0.5)
            
            # 噪声滤波状态
            if not hasattr(self, '_clap_noise_state'):
                self._clap_noise_state = [0.0, 0.0]
            
            for i in range(self.buffer_size):
                # 多层噪声模拟拍手
                noise1 = np.random.uniform(-1, 1)
                noise2 = np.random.uniform(-1, 1)
                noise3 = np.random.uniform(-1, 1)
                
                # 多级带通滤波
                self._clap_noise_state[0] = 0.3 * self._clap_noise_state[0] + 0.7 * noise1
                self._clap_noise_state[1] = 0.5 * self._clap_noise_state[1] + 0.5 * noise2
                
                bandpass1 = self._clap_noise_state[0] * 0.5
                bandpass2 = self._clap_noise_state[1] * 0.3
                bandpass3 = (noise1 + noise3 * 0.5) * 0.2
                
                # 多个短促点击
                click1 = np.exp(-self.drum_phases['clap'] * 120) * np.random.uniform(-1, 1) * 0.2
                click2 = np.exp(-self.drum_phases['clap'] * 80) * np.random.uniform(-1, 1) * 0.15
                
                # 音调成分
                tone = np.sin(self.drum_phases['clap'] * 2 * np.pi * (800 + tone_val * 400) / self.sample_rate) * np.exp(-self.drum_phases['clap'] * 50) * 0.1
                
                output[i] = (bandpass1 + bandpass2 + bandpass3 + click1 + click2 + tone) * self.drum_envelopes['clap']
                self.drum_phases['clap'] += 1
                self.drum_envelopes['clap'] *= (0.995 - decay * 0.008)
        return output * self.drum_volumes['clap']
    
    def _generate_808_hihat(self, is_open=False):
        """生成TR-808风格踩镲（增强版）"""
        drum_key = 'hihat_open' if is_open else 'hihat_closed'
        output = np.zeros(self.buffer_size)
        
        if self.drum_envelopes[drum_key] > 0.01:
            params = self.drum_params[drum_key]
            decay = params['decay']
            tone = params['tone']
            
            # 滤波状态
            if not hasattr(self, '_hihat_filter'):
                self._hihat_filter = [0.0, 0.0, 0.0]
            
            for i in range(self.buffer_size):
                # 多个高频方波叠加
                square_sum = 0
                for f_mult in [5, 6, 8, 10, 12, 14, 16, 18]:
                    freq = 6000 * tone * f_mult / 10
                    phase = self.drum_phases[drum_key] * 2 * np.pi * freq / self.sample_rate
                    square_sum += np.sign(np.sin(phase)) * (1.0 / (f_mult * 0.5))
                
                # 多层噪声
                noise1 = np.random.uniform(-1, 1)
                noise2 = np.random.uniform(-1, 1)
                
                # 高通滤波
                self._hihat_filter[0] = 0.1 * self._hihat_filter[0] + 0.9 * noise1
                self._hihat_filter[1] = 0.2 * self._hihat_filter[1] + 0.8 * noise2
                hpf_noise = (noise1 - self._hihat_filter[0]) * 0.5 + (noise2 - self._hihat_filter[1]) * 0.3
                
                output[i] = (square_sum * 0.5 + hpf_noise * 0.5) * self.drum_envelopes[drum_key]
                self.drum_phases[drum_key] += 1
                
                decay_rate = 0.988 - decay * 0.015 if is_open else 0.965 - decay * 0.02
                self.drum_envelopes[drum_key] *= decay_rate
        return output * self.drum_volumes[drum_key]
    
    def _generate_808_tom(self, tom_type='mid'):
        """生成TR-808风格通鼓（增强版）"""
        drum_key = f'tom_{tom_type}'
        output = np.zeros(self.buffer_size)
        
        if self.drum_envelopes[drum_key] > 0.01:
            params = self.drum_params[drum_key]
            pitch = params['pitch']
            decay = params['decay']
            
            # 不同通鼓的基础频率
            base_freqs = {'high': 220, 'mid': 150, 'low': 100}
            base_freq = base_freqs[tom_type] * pitch
            
            for i in range(self.buffer_size):
                # 多层频率下滑
                freq1 = base_freq * np.exp(-self.drum_phases[drum_key] * 1.5)
                freq2 = base_freq * 0.8 * np.exp(-self.drum_phases[drum_key] * 2.5)
                freq1 = max(50, freq1)
                freq2 = max(40, freq2)
                
                # 主音 + 泛音
                tone1 = np.sin(self.drum_phases[drum_key] * 2 * np.pi * freq1 / self.sample_rate)
                tone2 = np.sin(self.drum_phases[drum_key] * 2 * np.pi * freq2 / self.sample_rate) * 0.5
                tone3 = np.sin(self.drum_phases[drum_key] * 2 * np.pi * freq1 * 2 / self.sample_rate) * 0.2
                
                # 轻微噪声
                noise = np.random.uniform(-1, 1) * 0.1
                
                output[i] = (tone1 * 0.6 + tone2 * 0.25 + tone3 * 0.1 + noise) * self.drum_envelopes[drum_key]
                self.drum_phases[drum_key] += 1
                self.drum_envelopes[drum_key] *= (0.997 - decay * 0.004)
        return output * self.drum_volumes[drum_key]
    
    def _generate_808_rimshot(self):
        """生成TR-808风格边击（增强版）"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes['rimshot'] > 0.01:
            pitch = self.drum_params['rimshot']['pitch']
            
            for i in range(self.buffer_size):
                # 多层频率
                freq1 = 750 * pitch
                freq2 = 1100 * pitch
                
                tone1 = np.sin(self.drum_phases['rimshot'] * 2 * np.pi * freq1 / self.sample_rate)
                tone2 = np.sin(self.drum_phases['rimshot'] * 2 * np.pi * freq2 / self.sample_rate) * 0.4
                # 点击成分
                click = np.exp(-self.drum_phases['rimshot'] * 150) * np.random.uniform(-1, 1) * 0.2
                
                output[i] = (tone1 * 0.5 + tone2 * 0.35 + click * 0.15) * self.drum_envelopes['rimshot']
                self.drum_phases['rimshot'] += 1
                self.drum_envelopes['rimshot'] *= 0.94
        return output * self.drum_volumes['rimshot']
    
    def _generate_808_cowbell(self):
        """生成TR-808风格牛铃（增强版）"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes['cowbell'] > 0.01:
            pitch = self.drum_params['cowbell']['pitch']
            decay = self.drum_params['cowbell']['decay']
            
            for i in range(self.buffer_size):
                # 多个失谐的方波
                freq1 = 560 * pitch
                freq2 = 845 * pitch
                freq3 = 1120 * pitch
                
                square1 = np.sign(np.sin(self.drum_phases['cowbell'] * 2 * np.pi * freq1 / self.sample_rate))
                square2 = np.sign(np.sin(self.drum_phases['cowbell'] * 2 * np.pi * freq2 / self.sample_rate))
                square3 = np.sin(self.drum_phases['cowbell'] * 2 * np.pi * freq3 / self.sample_rate) * 0.3
                
                output[i] = (square1 * 0.4 + square2 * 0.4 + square3 * 0.2) * self.drum_envelopes['cowbell']
                self.drum_phases['cowbell'] += 1
                self.drum_envelopes['cowbell'] *= (0.996 - decay * 0.004)
        return output * self.drum_volumes['cowbell']
    
    def _generate_808_clave(self):
        """生成TR-808风格克拉维（增强版）"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes['clave'] > 0.01:
            pitch = self.drum_params['clave']['pitch']
            
            for i in range(self.buffer_size):
                freq1 = 2500 * pitch
                freq2 = 3750 * pitch
                
                tone1 = np.sin(self.drum_phases['clave'] * 2 * np.pi * freq1 / self.sample_rate)
                tone2 = np.sin(self.drum_phases['clave'] * 2 * np.pi * freq2 / self.sample_rate) * 0.3
                
                output[i] = (tone1 * 0.7 + tone2 * 0.3) * self.drum_envelopes['clave']
                self.drum_phases['clave'] += 1
                self.drum_envelopes['clave'] *= 0.91
        return output * self.drum_volumes['clave']
    
    def _generate_808_maracas(self):
        """生成TR-808风格沙锤（增强版）"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes['maracas'] > 0.01:
            decay = self.drum_params['maracas']['decay']
            
            # 滤波状态
            if not hasattr(self, '_maracas_filter'):
                self._maracas_filter = 0.0
            
            for i in range(self.buffer_size):
                noise = np.random.uniform(-1, 1)
                # 高通滤波
                self._maracas_filter = 0.15 * self._maracas_filter + 0.85 * noise
                hpf_noise = (noise - self._maracas_filter) * 1.2
                
                output[i] = hpf_noise * self.drum_envelopes['maracas']
                self.drum_phases['maracas'] += 1
                self.drum_envelopes['maracas'] *= (0.975 - decay * 0.008)
        return output * self.drum_volumes['maracas']
    
    def _generate_808_conga(self, conga_type='high'):
        """生成TR-808风格康加鼓"""
        drum_key = f'conga_{conga_type}'
        if drum_key not in self.drum_envelopes:
            return np.zeros(self.buffer_size)
        
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes[drum_key] > 0.01:
            base_freqs = {'high': 300, 'mid': 200, 'low': 140}
            base_freq = base_freqs.get(conga_type, 200)
            
            for i in range(self.buffer_size):
                freq = base_freq * np.exp(-self.drum_phases[drum_key] * 1.5)
                freq = max(60, freq)
                
                tone = np.sin(self.drum_phases[drum_key] * 2 * np.pi * freq / self.sample_rate)
                tone += np.sin(self.drum_phases[drum_key] * 2 * np.pi * freq * 2.5 / self.sample_rate) * 0.2
                # 轻微拍击声
                slap = np.exp(-self.drum_phases[drum_key] * 80) * np.random.uniform(-1, 1) * 0.15
                
                output[i] = (tone * 0.85 + slap * 0.15) * self.drum_envelopes[drum_key]
                self.drum_phases[drum_key] += 1
                self.drum_envelopes[drum_key] *= 0.994
        return output * self.drum_volumes.get(drum_key, 0.6)
    
    def _generate_808_cymbal(self):
        """生成TR-808风格镲片"""
        output = np.zeros(self.buffer_size)
        if self.drum_envelopes.get('cymbal', 0) > 0.01:
            decay = self.drum_params.get('cymbal', {}).get('decay', 0.5)
            
            for i in range(self.buffer_size):
                # 多个高频成分
                noise = np.random.uniform(-1, 1)
                shimmer = np.sin(self.drum_phases['cymbal'] * 2 * np.pi * 6000 / self.sample_rate) * 0.3
                shimmer += np.sin(self.drum_phases['cymbal'] * 2 * np.pi * 8000 / self.sample_rate) * 0.2
                
                output[i] = (noise * 0.6 + shimmer * 0.4) * self.drum_envelopes['cymbal']
                self.drum_phases['cymbal'] += 1
                self.drum_envelopes['cymbal'] *= (0.992 - decay * 0.003)
        return output * self.drum_volumes.get('cymbal', 0.5)
    
    def _trigger_drum(self, drum_type):
        """触发鼓音"""
        if drum_type in self.drum_envelopes:
            self.drum_envelopes[drum_type] = 1.0
            self.drum_phases[drum_type] = 0.0
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """音频回调（先锋版）"""
        output = np.zeros(frames)
        
        # 更新平滑参数
        self._update_smooth_parameters()
        
        # 生成变奏
        self._generate_variation()
        
        # 计算步进
        step_duration = 60.0 / self.tempo / 4
        step_inc = self.buffer_size / self.sample_rate / step_duration
        self.step_phase += step_inc
        
        # 步进触发
        if self.step_phase >= 1.0:
            self.step_phase -= 1.0
            
            # 应用摇摆
            swing_delay = 0
            if self.current_step % 2 == 1:
                swing_delay = self.swing * 0.3
            
            # 获取混合模式
            blended_pattern = self._blend_patterns()
            
            # 合并生成的变奏
            for drum in self.generated_pattern:
                for i in range(16):
                    if self.generated_pattern[drum][i] > 0:
                        blended_pattern[drum][i] = max(blended_pattern[drum][i], self.generated_pattern[drum][i])
            
            # 触发当前步的所有鼓
            for drum_type, steps in blended_pattern.items():
                if self.current_step < len(steps) and steps[self.current_step] > 0.3:
                    self._trigger_drum(drum_type)
            
            self.current_step = (self.current_step + 1) % self.step_count
        
        # 生成所有鼓音
        output += self._generate_808_kick()
        output += self._generate_808_snare()
        output += self._generate_808_clap()
        output += self._generate_808_hihat(is_open=False)
        output += self._generate_808_hihat(is_open=True)
        output += self._generate_808_tom('high')
        output += self._generate_808_tom('mid')
        output += self._generate_808_tom('low')
        output += self._generate_808_rimshot()
        output += self._generate_808_cowbell()
        output += self._generate_808_clave()
        output += self._generate_808_maracas()
        output += self._generate_808_conga('high')
        output += self._generate_808_conga('low')
        output += self._generate_808_cymbal()
        
        # 应用延迟效果（立体声模拟）
        for i in range(frames):
            # 左声道延迟
            delay_idx_l = (self.delay_index - int(len(self.delay_buffer) * 0.25) + i) % len(self.delay_buffer)
            # 右声道延迟（稍长）
            delay_idx_r = (self.delay_index - int(len(self.delay_buffer) * 0.35) + i) % len(self.delay_buffer)
            
            delay_l = self.delay_buffer[delay_idx_l] * self.delay_amount
            delay_r = self.delay_buffer[delay_idx_r] * self.delay_amount * 0.7
            
            output[i] += (delay_l + delay_r) * 0.5
            self.delay_buffer[self.delay_index] = output[i] * self.delay_feedback + self.delay_buffer[self.delay_index] * 0.6
            self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
        
        # 应用多层混响
        for i in range(frames):
            # 多个延迟线模拟混响
            rev_idx1 = (self.reverb_index - int(len(self.reverb_buffer) * 0.3) + i) % len(self.reverb_buffer)
            rev_idx2 = (self.reverb_index - int(len(self.reverb_buffer) * 0.45) + i) % len(self.reverb_buffer)
            rev_idx3 = (self.reverb_index - int(len(self.reverb_buffer) * 0.6) + i) % len(self.reverb_buffer)
            
            reverb_sum = (self.reverb_buffer[rev_idx1] * 0.5 + 
                         self.reverb_buffer[rev_idx2] * 0.3 + 
                         self.reverb_buffer[rev_idx3] * 0.2)
            output[i] += reverb_sum * self.reverb_amount
            self.reverb_buffer[self.reverb_index] = output[i] * 0.7
            self.reverb_index = (self.reverb_index + 1) % len(self.reverb_buffer)
        
        # 应用动态滤波器
        alpha = 1.0 - self.filter_cutoff * 0.25
        for i in range(frames):
            # 低通
            self.filter_state = alpha * self.filter_state + (1 - alpha) * output[i]
            # 共振峰
            resonance = (output[i] - self.filter_state) * self.filter_resonance * 0.3
            output[i] = self.filter_state + resonance
        
        # 多段压缩（低频、中频、高频）
        max_val = np.max(np.abs(output))
        if max_val > 0.65:
            # 软压缩曲线
            ratio = self.compression_ratio
            output = np.sign(output) * (0.65 + (np.abs(output) - 0.65) * (1 - ratio) * 0.5)
        
        # 最终限制器和软削波
        output = np.tanh(output * 1.3) * 0.85
        
        # 直流偏移消除
        output = output - np.mean(output)
        
        outdata[:, 0] = np.clip(output * self.master_volume, -0.99, 0.99).astype(np.float32)
    
    def update_from_gesture(self, hand_y, hand_x, hand_area, finger_count):
        """从手势更新参数（平滑过渡）"""
        self.hand_y = hand_y
        self.hand_x = hand_x
        self.hand_area = hand_area
        self.finger_count = finger_count
        
        # Y位置控制音高/音色参数（设置目标值）
        self.drum_params['kick']['pitch_target'] = 0.7 + hand_y * 0.6
        self.drum_params['snare']['snappy_target'] = 0.3 + hand_y * 0.7
        self.drum_params['kick']['tone_target'] = 0.3 + hand_y * 0.5
        
        # X位置控制节奏速度（设置目标值）
        self.tempo_target = 60 + int(hand_x * 160)  # 60-220 BPM
        
        # 面积控制摇摆感和变化强度
        self.swing_target = min(1.0, hand_area / 40000)
        self.variation_intensity = 0.2 + min(0.5, hand_area / 60000)
        
        # 手指数量切换模式（设置目标风格）
        pattern_names = list(self.patterns.keys())
        self.style_target = pattern_names[finger_count % len(pattern_names)]
        self.current_pattern = self.style_target
        
        # 手指数量也切换编辑的鼓
        drum_types = ['kick', 'snare', 'clap', 'hihat_closed', 'hihat_open', 'tom_high', 'tom_mid', 'tom_low']
        self.editing_drum = drum_types[finger_count % len(drum_types)]
    
    def update_from_color(self, colors):
        """从颜色更新音色（平滑过渡）"""
        if colors is None or len(colors) == 0:
            return
        
        main_color = colors[0]
        r, g, b = main_color[0] / 255.0, main_color[1] / 255.0, main_color[2] / 255.0
        
        # 亮度影响衰减（设置目标值）
        brightness = (r + g + b) / 3
        for drum in self.drum_params:
            if 'decay_target' in self.drum_params[drum]:
                self.drum_params[drum]['decay_target'] = 0.2 + brightness * 0.4
        
        # 饱和度影响音色（设置目标值）
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        saturation = (max_c - min_c) / max_c if max_c > 0 else 0
        
        self.drum_params['kick']['tone_target'] = 0.3 + saturation * 0.5
        self.drum_params['snare']['tone_target'] = 0.3 + saturation * 0.5
        
        # 混响量（设置目标值）
        self.reverb_amount_target = 0.1 + saturation * 0.4
        
        # 延迟量
        self.delay_amount = 0.1 + saturation * 0.2
        
        # 滤波器截止
        self.filter_cutoff = 0.5 + brightness * 0.4
    
    def get_display_info(self):
        """获取显示信息"""
        return {
            'tempo': int(self.tempo),
            'current_step': self.current_step,
            'current_pattern': self.current_pattern,
            'editing_drum': getattr(self, 'editing_drum', 'kick'),
            'swing': self.swing,
            'step_count': self.step_count,
            'variation': self.variation_intensity,
        }
    
    def get_pattern_for_display(self):
        """获取当前混合模式用于显示"""
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

def draw_ui(image, fps, current_mode, color_centers=None, synth_params=None, sampler_params=None, sampler=None):
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
        # 在顶部右侧显示音序器参数
        info_x = w - 380
        cv2.putText(image, f'TR-808 {tempo}BPM Step:{current_step+1}/{step_count}', (info_x, int(h * 0.035)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 200, 100), 1)
        cv2.putText(image, f'{pattern_name.upper()} Swing:{swing:.0%} Var:{variation:.0%}', (info_x, int(h * 0.065)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (100, 255, 200), 1)
        
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
    
    # 底部UI
    cv2.rectangle(image, (0, int(h * 0.92)), (w, h), (0, 0, 0), -1)
    cv2.putText(image, 'q:Quit 1:Mouse 2:Gesture 3:ASCII 4:Color 5:Synth 6:Sampler r:Reset', (10, int(h * 0.96)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (180, 180, 180), 1)
    
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
                                 info['editing_drum'], info['swing'], info['step_count'], info['variation'])
        else:
            # 停止采样器（切换模式时）
            if sampler is not None:
                sampler.stop()
                sampler = None
        
        curr_time = time.time()
        frame_time = curr_time - prev_time
        if frame_time > 0:
            fps = 1 / frame_time
            fps_smooth = fps_smooth * 0.9 + fps * 0.1
        prev_time = curr_time
        
        display_image = draw_ui(display_image, fps_smooth, current_mode, color_centers, synth_params, sampler_params, sampler)
        
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
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
