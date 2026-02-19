import numpy as np
import time
import threading
from collections import deque
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.properties import StringProperty
from kivy.metrics import dp, sp

MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2
MODE_SYNTH = 3
MODE_SAMPLER = 4

ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
ASCII_CHARS_EXTENDED = " .',:;ilI1|\\/()[]{}?_-+~=<>!@#$%&*#O0O8&@#*+"

instrument_colors = {
    'piano': {'base': (150, 180, 220), 'bright': (200, 220, 255)},
    'violin': {'base': (200, 150, 150), 'bright': (255, 200, 200)},
    'guitar': {'base': (150, 200, 150), 'bright': (200, 255, 200)},
    'saxophone': {'base': (200, 180, 140), 'bright': (255, 230, 180)},
    'default': {'base': (100, 100, 200), 'bright': (150, 150, 255)},
}


def _is_android():
    try:
        import android  # noqa: F401
        return True
    except ImportError:
        return False


class AndroidCamera:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.frame = None
        self.frame_lock = threading.Lock()
        
    def start(self):
        try:
            import cv2
            # Android 使用 CAP_ANDROID 提高兼容性
            if _is_android():
                try:
                    self.cap = cv2.VideoCapture(0, cv2.CAP_ANDROID)
                except (TypeError, Exception):
                    self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()
                Logger.info("Camera started")
                return True
            # Android 备选：尝试其他相机索引
            if _is_android():
                for idx in [1, 2, 3]:
                    try:
                        self.cap = cv2.VideoCapture(idx, cv2.CAP_ANDROID)
                    except (TypeError, Exception):
                        self.cap = cv2.VideoCapture(idx)
                    if self.cap and self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.running = True
                        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                        self.thread.start()
                        Logger.info(f"Camera {idx} started")
                        return True
        except Exception as e:
            Logger.error(f"Camera init error: {e}")
        return False
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
    
    def _capture_loop(self):
        import cv2
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame.copy()
            time.sleep(0.01)
    
    def get_frame(self):
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
        return None


class HandDetector:
    def __init__(self):
        import cv2
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)
        self.last_valid_contour = None
        self.last_valid_time = 0
        self.gesture_history = deque(maxlen=5)
        
    def detect(self, frame):
        import cv2
        h, w = frame.shape[:2]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        lower_lab = np.array([20, 120, 120], dtype=np.uint8)
        upper_lab = np.array([235, 160, 160], dtype=np.uint8)
        mask_lab = cv2.inRange(lab, lower_lab, upper_lab)
        
        mask_skin = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask_skin = cv2.bitwise_and(mask_skin, mask_lab)
        
        fg_mask = self.bg_subtractor.apply(frame)
        
        mask = cv2.bitwise_and(mask_skin, fg_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        max_area = 0
        min_area = 2000
        max_area_limit = 150000
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area_limit:
                if area > max_area:
                    max_area = area
                    best_contour = cnt
        
        if best_contour is not None:
            self.last_valid_contour = best_contour
            self.last_valid_time = time.time()
        elif self.last_valid_contour is not None and (time.time() - self.last_valid_time) < 0.5:
            best_contour = self.last_valid_contour
        
        return best_contour, mask, max_area
    
    def recognize_gesture(self, frame, contour):
        import cv2
        h, w = frame.shape[:2]
        
        if contour is None:
            return 'None', 0
        
        area = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        rect_w, rect_h = rect[1]
        aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 1
        
        hull = cv2.convexHull(contour, returnPoints=False)
        
        finger_count = 0
        if len(hull) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 8000:
                            finger_count += 1
            except:
                pass
        
        solidity = area / (rect_w * rect_h) if (rect_w * rect_h) > 0 else 0
        
        gesture = 'Paper'
        if solidity > 0.8:
            gesture = 'Fist'
        elif finger_count == 1:
            gesture = 'Victory'
        elif finger_count == 0 and solidity < 0.6:
            gesture = 'One'
        elif finger_count >= 2:
            gesture = 'Three'
        
        self.gesture_history.append(gesture)
        if len(self.gesture_history) >= 3:
            from collections import Counter
            counter = Counter(self.gesture_history)
            gesture = counter.most_common(1)[0][0]
        
        return gesture, finger_count


class AndroidAudioSynthesizer:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        self.frequency = 440.0
        self.lfo_rate = 1.0
        self.lfo_depth = 0.3
        self.envelope = 0.0
        self.envelope_target = 0.0
        self.phase = 0.0
        self.volume = 0.2
        self.audio_track = None
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=0.5)
            except:
                pass
        if self.audio_track:
            try:
                self.audio_track.stop()
                self.audio_track.release()
            except:
                pass
            self.audio_track = None
    
    def _audio_loop(self):
        try:
            from jnius import autoclass
            AudioTrack = autoclass('android.media.AudioTrack')
            AudioFormat = autoclass('android.media.AudioFormat')
            AudioManager = autoclass('android.media.AudioManager')
            
            channel_config = AudioFormat.CHANNEL_OUT_MONO
            audio_format = AudioFormat.ENCODING_PCM_16BIT
            
            min_buf = AudioTrack.getMinBufferSize(
                self.sample_rate, channel_config, audio_format)
            buffer_size = max(4096, min_buf)
            
            self.audio_track = AudioTrack(
                AudioManager.STREAM_MUSIC,
                self.sample_rate,
                channel_config,
                audio_format,
                buffer_size,
                AudioTrack.MODE_STREAM
            )
            
            self.audio_track.play()
            Logger.info("Audio started")
            
            while self.running:
                frames = buffer_size // 2
                wave = self._generate_wave(frames)
                audio_data = (np.clip(wave, -1, 1) * 32767).astype(np.int16).tobytes()
                self.audio_track.write(audio_data, 0, len(audio_data))
                
        except Exception as e:
            Logger.error(f"Audio error: {e}")
        self.running = False
    
    def _generate_wave(self, frames):
        t = np.arange(frames) / self.sample_rate
        
        lfo = np.sin(2 * np.pi * self.lfo_rate * t)
        lfo_mod = 1.0 + lfo * self.lfo_depth
        
        freq = self.frequency * lfo_mod
        phase_inc = 2 * np.pi * freq / self.sample_rate
        self.phase = self.phase + np.cumsum(phase_inc)
        wave = np.sin(self.phase)
        
        if self.envelope_target > self.envelope:
            self.envelope += 0.01
        else:
            self.envelope -= 0.005
        self.envelope = max(0, min(1, self.envelope))
        
        wave = wave * self.envelope * self.volume
        self.phase = self.phase[-1] % (2 * np.pi * 100)
        
        return wave
    
    def update(self, x, y, amount):
        self.frequency = 150 + (1.0 - y) * 500
        self.lfo_rate = 0.5 + x * 4.0
        self.envelope_target = min(0.8, amount * 1.0)


class AndroidTR808:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        self.tempo = 120
        self.current_step = 0
        self.step_count = 16
        self.current_pattern = 'classic'
        self.swing = 0.0
        self.audio_track = None
        self.patterns = {
            'classic': {
                'kick': [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                'snare': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                'hihat_closed': [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
            },
            'hiphop': {
                'kick': [1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
                'snare': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                'hihat_closed': [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1],
            },
            'house': {
                'kick': [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                'snare': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                'hihat_closed': [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
            },
        }
        self.pattern_colors = {
            'classic': {'base': (100, 150, 200), 'bright': (150, 200, 255)},
            'hiphop': {'base': (200, 100, 150), 'bright': (255, 150, 200)},
            'house': {'base': (100, 200, 150), 'bright': (150, 255, 200)},
        }
        self.drum_envelopes = {'kick': 0.0, 'snare': 0.0, 'hihat_closed': 0.0}
        self.volume = 0.3
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=0.5)
            except:
                pass
        if self.audio_track:
            try:
                self.audio_track.stop()
                self.audio_track.release()
            except:
                pass
            self.audio_track = None
    
    def _audio_loop(self):
        try:
            from jnius import autoclass
            AudioTrack = autoclass('android.media.AudioTrack')
            AudioFormat = autoclass('android.media.AudioFormat')
            AudioManager = autoclass('android.media.AudioManager')
            
            channel_config = AudioFormat.CHANNEL_OUT_MONO
            audio_format = AudioFormat.ENCODING_PCM_16BIT
            min_buf = AudioTrack.getMinBufferSize(self.sample_rate, channel_config, audio_format)
            buffer_size = max(4096, min_buf)
            
            self.audio_track = AudioTrack(AudioManager.STREAM_MUSIC, self.sample_rate,
                channel_config, audio_format, buffer_size, AudioTrack.MODE_STREAM)
            self.audio_track.play()
            Logger.info("TR-808 started")
            
            step_duration = 60.0 / self.tempo / 4.0
            samples_per_step = int(step_duration * self.sample_rate)
            sample_pos = 0
            last_step_time = time.time()
            
            while self.running:
                curr_time = time.time()
                if curr_time - last_step_time >= step_duration:
                    self.current_step = (self.current_step + 1) % self.step_count
                    last_step_time = curr_time
                    
                    pattern = self.patterns.get(self.current_pattern, self.patterns['classic'])
                    for drum in pattern:
                        if pattern[drum][self.current_step] > 0:
                            self.drum_envelopes[drum] = 1.0
                
                frames = min(buffer_size // 2, samples_per_step)
                wave = self._generate_drum_wave(frames)
                audio_data = (np.clip(wave, -1, 1) * 32767).astype(np.int16).tobytes()
                self.audio_track.write(audio_data, 0, len(audio_data))
                
        except Exception as e:
            Logger.error(f"TR-808 error: {e}")
        self.running = False
    
    def _generate_drum_wave(self, frames):
        t = np.arange(frames) / self.sample_rate
        wave = np.zeros(frames)
        
        if self.drum_envelopes['kick'] > 0.01:
            freq = 150 * np.exp(-t * 20)
            kick_wave = np.sin(2 * np.pi * np.cumsum(freq) / self.sample_rate)
            kick_env = self.drum_envelopes['kick'] * np.exp(-t * 15)
            wave += kick_wave * kick_env
            self.drum_envelopes['kick'] *= (1.0 - 0.05)
        
        if self.drum_envelopes['snare'] > 0.01:
            noise = np.random.uniform(-1, 1, frames)
            snare_env = self.drum_envelopes['snare'] * np.exp(-t * 10)
            wave += noise * snare_env * 0.5
            self.drum_envelopes['snare'] *= (1.0 - 0.04)
        
        if self.drum_envelopes['hihat_closed'] > 0.01:
            noise = np.random.uniform(-1, 1, frames)
            hihat_env = self.drum_envelopes['hihat_closed'] * np.exp(-t * 40)
            wave += noise * hihat_env * 0.3
            self.drum_envelopes['hihat_closed'] *= (1.0 - 0.08)
        
        return wave * self.volume
    
    def update_from_gesture(self, hand_x, hand_y, hand_area):
        self.tempo = 80 + hand_x * 120
        pattern_list = list(self.patterns.keys())
        pattern_idx = min(int(hand_y * len(pattern_list)), len(pattern_list) - 1)
        self.current_pattern = pattern_list[pattern_idx]
        self.swing = min(0.5, hand_area * 0.00001)


def create_ascii_art(image, contour, mask, synth=None, sampler=None):
    import cv2
    if contour is None:
        return image
    
    h, w = image.shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    cw = min(cw, w - x)
    ch = min(ch, h - y)
    
    if cw < 10 or ch < 10:
        return image
    
    roi = image[y:y+ch, x:x+cw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    mask_region = np.zeros((ch, cw), dtype=np.uint8)
    contour_shifted = contour - [x, y]
    cv2.drawContours(mask_region, [contour_shifted], -1, 255, -1)
    
    if sampler is not None and hasattr(sampler, 'drum_envelopes'):
        base_char_size = max(14, min(20, cw // 12))
        step_factor = 1.0 - sampler.current_step / 16 * 0.3
        char_size = int(base_char_size * step_factor)
        char_size = max(8, min(28, char_size))
        char_offset = int((sampler.tempo - 60) / 30)
        chars_to_use = ASCII_CHARS_EXTENDED
        skip_rate = 0.25
        inst_colors = sampler.pattern_colors.get(sampler.current_pattern, sampler.pattern_colors['classic'])
    elif synth is not None:
        base_char_size = max(14, min(20, cw // 12))
        freq_factor = 1.0 - (synth.frequency - 80) / 700 * 0.5
        char_size = int(base_char_size * freq_factor)
        char_size = max(8, min(28, char_size))
        char_offset = int(synth.envelope * 30)
        chars_to_use = ASCII_CHARS_EXTENDED
        skip_rate = 0.25
        inst_colors = None
    else:
        char_size = max(4, min(8, cw // 30))
        char_offset = 0
        chars_to_use = ASCII_CHARS
        skip_rate = 0
        inst_colors = None
    
    ascii_h = ch // char_size
    ascii_w = cw // char_size
    
    if ascii_h < 3 or ascii_w < 3:
        return image
    
    result = image.copy()
    current_time = time.time()
    
    np.random.seed(42)
    phase_offsets = np.random.random((ascii_h, ascii_w)) * 2 * np.pi
    
    for row in range(ascii_h):
        for col in range(ascii_w):
            if skip_rate > 0 and np.random.random() < skip_rate:
                continue
            
            y_start = row * char_size
            y_end = min((row + 1) * char_size, ch)
            x_start = col * char_size
            x_end = min((col + 1) * char_size, cw)
            
            block = gray[y_start:y_end, x_start:x_end]
            block_mask = mask_region[y_start:y_end, x_start:x_end]
            
            if cv2.countNonZero(block_mask) < (block_mask.size * 0.3):
                continue
            
            if block.size == 0:
                continue
            
            avg_brightness = np.mean(block)
            brightness_jitter = avg_brightness
            
            base_char_idx = int(brightness_jitter / 255 * (len(chars_to_use) - 1))
            
            if sampler is not None and hasattr(sampler, 'drum_envelopes'):
                pos_variety = (row * 7 + col * 11 + int(phase_offsets[row, col] * 100)) % 8
                step_variety = sampler.current_step % 8
                tempo_variety = int((sampler.tempo - 60) / 20) % 6
                pattern_variety = hash(sampler.current_pattern) % 5
                swing_variety = int(sampler.swing * 10) % 4
                total_variety = pos_variety + step_variety + tempo_variety + pattern_variety + swing_variety
                char_idx = (base_char_idx + total_variety) % len(chars_to_use)
            elif synth is not None:
                pos_variety = (row * 7 + col * 11 + int(phase_offsets[row, col] * 100)) % 8
                freq_variety = int((synth.frequency - 80) / 50) % 6
                lfo_variety = int(synth.lfo_rate * 2) % 5
                envelope_variety = int(synth.envelope * 10) % 4
                total_variety = pos_variety + freq_variety + lfo_variety + envelope_variety
                char_idx = (base_char_idx + total_variety) % len(chars_to_use)
            else:
                char_idx = base_char_idx
            
            char_idx = max(0, min(len(chars_to_use) - 1, char_idx))
            
            if sampler is not None and inst_colors is not None:
                brightness_factor = brightness_jitter / 255.0
                base_color = inst_colors['base']
                bright_color = inst_colors['bright']
                blue_val = int((base_color[0] + (bright_color[0] - base_color[0]) * brightness_factor) * 0.5)
                green_val = int((base_color[1] + (bright_color[1] - base_color[1]) * brightness_factor) * 0.5)
                red_val = int((base_color[2] + (bright_color[2] - base_color[2]) * brightness_factor) * 0.5)
                gray_mix = int((blue_val + green_val + red_val) / 3)
                blue_val = int(blue_val * 0.5 + gray_mix * 0.5)
                green_val = int(green_val * 0.5 + gray_mix * 0.5)
                red_val = int(red_val * 0.5 + gray_mix * 0.5)
                gradient_color = (blue_val, green_val, red_val)
            else:
                blue_val = int((50 + brightness_jitter * 0.4) * 0.5)
                green_val = int((25 + brightness_jitter * 0.3) * 0.5)
                red_val = int((5 + brightness_jitter * 0.2) * 0.5)
                gray_mix = int((blue_val + green_val + red_val) / 3)
                blue_val = int(blue_val * 0.5 + gray_mix * 0.5)
                green_val = int(green_val * 0.5 + gray_mix * 0.5)
                red_val = int(red_val * 0.5 + gray_mix * 0.5)
                gradient_color = (blue_val, green_val, red_val)
            
            char = chars_to_use[char_idx]
            draw_x = x + x_start + char_size // 2
            draw_y = y + y_start + char_size
            font_scale = char_size / 18 if (synth is not None or sampler is not None) else char_size / 20
            cv2.putText(result, char, (draw_x, draw_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       gradient_color, 1, cv2.LINE_AA)
    
    return result


class HandGestureApp(App):
    fps_text = StringProperty('FPS: 0')
    mode_text = StringProperty('Camera')
    info_text = StringProperty('')
    gesture_text = StringProperty('--')
    fingers_text = StringProperty('0')
    
    def build(self):
        self.current_mode = MODE_CAMERA
        self.synth = None
        self.sampler = None
        self.camera = AndroidCamera()
        self.detector = HandDetector()
        self.fps = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.hand_position_history = deque(maxlen=3)
        
        root = BoxLayout(orientation='vertical', padding=dp(8), spacing=dp(6))
        
        # 顶部：标题 + FPS
        header = BoxLayout(size_hint=(1, None), height=dp(44), spacing=dp(8))
        title = Label(text='[b]Hand Gesture[/b]', markup=True, font_size=sp(20),
                      halign='left', size_hint_x=0.7)
        title.bind(size=title.setter('text_size'))
        self.fps_label = Label(text='FPS: 0', font_size=sp(14), color=(0.5, 0.9, 0.5, 1),
                               halign='right', size_hint_x=0.3)
        self.fps_label.bind(size=self.fps_label.setter('text_size'))
        header.add_widget(title)
        header.add_widget(self.fps_label)
        root.add_widget(header)
        
        # 信息栏：手势、手指、模式参数
        info_bar = BoxLayout(size_hint=(1, None), height=dp(52), spacing=dp(8))
        info_left = BoxLayout(orientation='vertical', spacing=dp(2), size_hint_x=0.6)
        self.gesture_label = Label(text='Gesture: --', font_size=sp(15), bold=True,
                                   color=(0.3, 0.9, 1.0, 1), halign='left')
        self.gesture_label.bind(size=self.gesture_label.setter('text_size'))
        self.fingers_label = Label(text='Fingers: 0', font_size=sp(13),
                                   color=(0.8, 0.8, 0.8, 1), halign='left')
        self.fingers_label.bind(size=self.fingers_label.setter('text_size'))
        info_left.add_widget(self.gesture_label)
        info_left.add_widget(self.fingers_label)
        info_right = BoxLayout(orientation='vertical', spacing=dp(2), size_hint_x=0.4)
        self.mode_label = Label(text='Camera', font_size=sp(14),
                                color=(1.0, 0.85, 0.4, 1), halign='right')
        self.mode_label.bind(size=self.mode_label.setter('text_size'))
        self.params_label = Label(text='', font_size=sp(11),
                                  color=(0.6, 0.9, 0.6, 1), halign='right')
        self.params_label.bind(size=self.params_label.setter('text_size'))
        info_right.add_widget(self.mode_label)
        info_right.add_widget(self.params_label)
        info_bar.add_widget(info_left)
        info_bar.add_widget(info_right)
        root.add_widget(info_bar)
        
        # 摄像头显示区
        self.display = Image(allow_stretch=True, keep_ratio=True, size_hint=(1, 0.6))
        root.add_widget(self.display)
        
        # 模式切换按钮
        mode_names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
        btn_row = GridLayout(cols=5, size_hint=(1, None), height=dp(52), spacing=dp(4))
        self.buttons = {}
        for name, mode in zip(mode_names, [MODE_CAMERA, MODE_GESTURE, MODE_ASCII, MODE_SYNTH, MODE_SAMPLER]):
            btn = ToggleButton(text=name, font_size=sp(12), group='mode',
                               size_hint=(None, 1), width=dp(70),
                               background_color=(0.25, 0.5, 0.7, 1) if mode != MODE_CAMERA else (0.2, 0.6, 0.4, 1),
                               background_normal='', background_down='')
            if mode == MODE_CAMERA:
                btn.state = 'down'
            btn.bind(on_press=lambda x, m=mode: self.set_mode(m))
            btn_row.add_widget(btn)
            self.buttons[mode] = btn
        root.add_widget(btn_row)
        
        # 底部：Reset
        btn_reset = Button(text='Reset', font_size=sp(14), size_hint=(1, None), height=dp(48),
                          background_color=(0.7, 0.35, 0.3, 1), background_normal='', background_down='')
        btn_reset.bind(on_press=self.reset)
        root.add_widget(btn_reset)
        
        Clock.schedule_once(self._request_permissions, 0.3)
        Clock.schedule_once(self._init_camera, 0.5)
        Clock.schedule_interval(self.update_loop, 1.0 / 30.0)
        
        return root
    
    def _request_permissions(self, dt):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.RECORD_AUDIO])
            Logger.info('Permissions requested')
        except ImportError:
            pass
    
    def _init_camera(self, dt):
        self.camera.start()
    
    def set_mode(self, mode):
        if mode == MODE_SYNTH:
            if self.synth is None:
                try:
                    self.synth = AndroidAudioSynthesizer()
                    self.synth.start()
                except Exception as e:
                    Logger.error(f"Synth error: {e}")
            if self.sampler:
                self.sampler.stop()
                self.sampler = None
        elif mode == MODE_SAMPLER:
            if self.sampler is None:
                try:
                    self.sampler = AndroidTR808()
                    self.sampler.start()
                except Exception as e:
                    Logger.error(f"Sampler error: {e}")
            if self.synth:
                self.synth.stop()
                self.synth = None
        else:
            if self.synth:
                self.synth.stop()
                self.synth = None
            if self.sampler:
                self.sampler.stop()
                self.sampler = None
        
        self.current_mode = mode
        for m, btn in self.buttons.items():
            if m == mode:
                btn.state = 'down'
            else:
                btn.state = 'normal'
    
    def reset(self, instance):
        self.detector.last_valid_contour = None
        self.hand_position_history.clear()
    
    def update_loop(self, dt):
        import cv2
        try:
            curr_time = time.time()
            frame_time = curr_time - self.prev_time
            if frame_time > 0:
                self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
            self.prev_time = curr_time
            
            frame = self.camera.get_frame()
            gesture, finger_count = '--', 0
            if frame is None:
                display = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display, 'Camera Initializing...', (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                display = frame.copy()
                h, w = display.shape[:2]
                
                contour, mask, area = self.detector.detect(display)
                gesture, finger_count = self.detector.recognize_gesture(display, contour)
                
                hand_x = 0.5
                hand_y = 0.5
                
                if contour is not None:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        hand_x = cx / w
                        hand_y = cy / h
                        self.hand_position_history.append((hand_x, hand_y))
                    
                    if self.current_mode in [MODE_GESTURE, MODE_ASCII, MODE_SYNTH, MODE_SAMPLER]:
                        cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                        
                        try:
                            hull = cv2.convexHull(contour)
                            cv2.drawContours(display, [hull], -1, (0, 200, 200), 1)
                        except:
                            pass
                    
                    if self.current_mode == MODE_ASCII or self.current_mode == MODE_SYNTH or self.current_mode == MODE_SAMPLER:
                        synth_obj = self.synth if self.current_mode == MODE_SYNTH else None
                        sampler_obj = self.sampler if self.current_mode == MODE_SAMPLER else None
                        display = create_ascii_art(display, contour, mask, synth_obj, sampler_obj)
                
                if self.current_mode == MODE_SYNTH and self.synth:
                    if len(self.hand_position_history) > 0:
                        avg_x = np.mean([p[0] for p in self.hand_position_history])
                        avg_y = np.mean([p[1] for p in self.hand_position_history])
                        self.synth.update(avg_x, avg_y, area / 50000.0 if area > 0 else 0.1)
                
                if self.current_mode == MODE_SAMPLER and self.sampler:
                    if len(self.hand_position_history) > 0:
                        avg_x = np.mean([p[0] for p in self.hand_position_history])
                        avg_y = np.mean([p[1] for p in self.hand_position_history])
                        self.sampler.update_from_gesture(avg_x, avg_y, area)
                
                self._draw_ui(display, gesture, finger_count, area)
            
            # 同步更新 Kivy 信息栏
            self.gesture_label.text = f'Gesture: {gesture}'
            self.fingers_label.text = f'Fingers: {finger_count}'
            mode_names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
            self.mode_label.text = mode_names[self.current_mode]
            if self.current_mode == MODE_SYNTH and self.synth:
                self.params_label.text = f'{int(self.synth.frequency)}Hz'
            elif self.current_mode == MODE_SAMPLER and self.sampler:
                self.params_label.text = f'{self.sampler.current_pattern} {int(self.sampler.tempo)} BPM'
            else:
                self.params_label.text = ''
            
            buf = cv2.flip(display, 0).tobytes()
            texture = Texture.create(size=(display.shape[1], display.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.display.texture = texture
            
            self.frame_count += 1
            if self.frame_count % 5 == 0:
                self.fps_text = f'FPS: {int(self.fps)}'
                self.fps_label.text = self.fps_text
            
        except Exception as e:
            Logger.error(f"Update error: {e}")
            import traceback
            Logger.error(traceback.format_exc())
    
    def _draw_ui(self, display, gesture, finger_count, area):
        import cv2
        h, w = display.shape[:2]
        
        mode_names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
        
        cv2.putText(display, f'Mode: {mode_names[self.current_mode]}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, self.fps_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if self.current_mode in [MODE_GESTURE, MODE_ASCII, MODE_SYNTH, MODE_SAMPLER]:
            cv2.putText(display, f'Gesture: {gesture}', (10, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f'Fingers: {finger_count}', (10, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.current_mode == MODE_SYNTH and self.synth:
            cv2.putText(display, f'Freq: {int(self.synth.frequency)}Hz', (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        if self.current_mode == MODE_SAMPLER and self.sampler:
            cv2.putText(display, f'Tempo: {int(self.sampler.tempo)}', (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 200), 1)
            cv2.putText(display, f'Pattern: {self.sampler.current_pattern}', (w - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 200), 1)
            cv2.putText(display, f'Step: {self.sampler.current_step + 1}/{self.sampler.step_count}', (w - 150, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 200), 1)
    
    def on_stop(self):
        self.camera.stop()
        if self.synth:
            self.synth.stop()
        if self.sampler:
            self.sampler.stop()


if __name__ == '__main__':
    HandGestureApp().run()
