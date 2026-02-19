from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import StringProperty, NumericProperty
from kivy.metrics import dp, sp
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Color, Rectangle, Line
import time
import math
from collections import deque

MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2
MODE_SYNTH = 3
MODE_SAMPLER = 4

GESTURE_TYPES = [
    'Open Hand', 'Fist', 'Pointing', 'Victory', 'Three Fingers',
    'Four Fingers', 'Thumb Up', 'Thumb Down', 'OK Sign', 'Rock Sign',
    'Call Me', 'Shaka', 'Peace', 'Stop', 'Go',
    'Left', 'Right', 'Up', 'Down', 'Circle',
    'Wave', 'Snap', 'Clap', 'Pinch', 'Spread'
]

ASCII_CHARS = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


class AdvancedGestureRecognizer:
    def __init__(self):
        self.history = deque(maxlen=10)
        self.feature_history = deque(maxlen=5)
        self.calibration_frames = 0
        self.calibration_data = []
        self.is_calibrated = False
        self.baseline_features = None
        self.sensitivity = 0.5
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.1
        self.confidence_threshold = 0.7
        
    def calibrate(self, features):
        self.calibration_data.append(features)
        self.calibration_frames += 1
        
        if self.calibration_frames >= 30:
            self.baseline_features = self._compute_baseline()
            self.is_calibrated = True
            Logger.info('GestureRecognizer: Calibrated')
            return True
        return False
    
    def _compute_baseline(self):
        if not self.calibration_data:
            return None
        
        n = len(self.calibration_data)
        baseline = {}
        
        keys = self.calibration_data[0].keys()
        for key in keys:
            values = [d[key] for d in self.calibration_data if key in d]
            if values:
                baseline[key] = {
                    'mean': sum(values) / len(values),
                    'std': math.sqrt(sum((v - sum(values)/len(values))**2 for v in values) / len(values)) if len(values) > 1 else 0
                }
        return baseline
    
    def extract_features(self, frame_data):
        features = {
            'brightness': frame_data.get('brightness', 128),
            'brightness_std': frame_data.get('brightness_std', 0),
            'motion_x': frame_data.get('motion_x', 0),
            'motion_y': frame_data.get('motion_y', 0),
            'motion_magnitude': frame_data.get('motion_magnitude', 0),
            'color_variance': frame_data.get('color_variance', 0),
            'edge_density': frame_data.get('edge_density', 0),
            'region_count': frame_data.get('region_count', 1),
            'aspect_ratio': frame_data.get('aspect_ratio', 1.0),
            'center_x': frame_data.get('center_x', 0.5),
            'center_y': frame_data.get('center_y', 0.5),
            'spread_x': frame_data.get('spread_x', 0.3),
            'spread_y': frame_data.get('spread_y', 0.3),
            'symmetry': frame_data.get('symmetry', 0.5),
            'compactness': frame_data.get('compactness', 0.5),
            'texture_complexity': frame_data.get('texture_complexity', 0.5),
            'dominant_direction': frame_data.get('dominant_direction', 0),
            'temporal_change': frame_data.get('temporal_change', 0),
        }
        
        self.feature_history.append(features)
        return features
    
    def classify_gesture(self, features):
        if not self.is_calibrated:
            return 'Calibrating...', 0, 0
        
        gesture_scores = {}
        
        brightness = features['brightness']
        motion = features['motion_magnitude']
        spread = (features['spread_x'] + features['spread_y']) / 2
        compactness = features['compactness']
        symmetry = features['symmetry']
        aspect = features['aspect_ratio']
        direction = features['dominant_direction']
        temporal = features['temporal_change']
        
        gesture_scores['Open Hand'] = spread * 0.8 + symmetry * 0.6 + (1 - compactness) * 0.5
        gesture_scores['Fist'] = compactness * 0.9 + (1 - spread) * 0.7 + symmetry * 0.4
        gesture_scores['Pointing'] = (aspect - 1) * 0.3 + compactness * 0.5 + (1 - spread) * 0.6
        gesture_scores['Victory'] = spread * 0.5 + symmetry * 0.3 + 0.4
        gesture_scores['Three Fingers'] = spread * 0.6 + symmetry * 0.4 + 0.3
        gesture_scores['Four Fingers'] = spread * 0.7 + symmetry * 0.5 + 0.3
        gesture_scores['Thumb Up'] = (aspect - 1) * 0.4 + compactness * 0.6 + 0.2
        gesture_scores['Thumb Down'] = (aspect - 1) * 0.4 + compactness * 0.6 + 0.2
        gesture_scores['OK Sign'] = compactness * 0.8 + symmetry * 0.7 + 0.3
        gesture_scores['Rock Sign'] = spread * 0.5 + (1 - symmetry) * 0.4 + 0.3
        gesture_scores['Stop'] = spread * 0.9 + symmetry * 0.8 + compactness * 0.3
        gesture_scores['Go'] = motion * 0.8 + direction * 0.3 + 0.2
        gesture_scores['Left'] = motion * 0.6 + (1 if direction < 0.3 else 0) * 0.5
        gesture_scores['Right'] = motion * 0.6 + (1 if direction > 0.7 else 0) * 0.5
        gesture_scores['Up'] = motion * 0.6 + (1 if features['motion_y'] < 0 else 0) * 0.5
        gesture_scores['Down'] = motion * 0.6 + (1 if features['motion_y'] > 0 else 0) * 0.5
        gesture_scores['Circle'] = temporal * 0.7 + motion * 0.5 + 0.2
        gesture_scores['Wave'] = temporal * 0.9 + motion * 0.7 + spread * 0.3
        gesture_scores['Pinch'] = (1 - spread) * 0.8 + compactness * 0.5 + 0.2
        gesture_scores['Spread'] = spread * 0.8 + temporal * 0.4 + 0.2
        
        if len(self.feature_history) >= 3:
            recent = list(self.feature_history)[-3:]
            motion_trend = sum(f['motion_magnitude'] for f in recent) / 3
            
            if motion_trend > 0.5:
                gesture_scores['Wave'] += 0.3
                gesture_scores['Circle'] += 0.2
        
        best_gesture = max(gesture_scores, key=gesture_scores.get)
        confidence = min(1.0, gesture_scores[best_gesture])
        
        self.history.append(best_gesture)
        
        if len(self.history) >= 5:
            from collections import Counter
            counter = Counter(list(self.history)[-5:])
            most_common = counter.most_common(1)[0]
            if most_common[1] >= 3:
                best_gesture = most_common[0]
                confidence = min(1.0, confidence + 0.1)
        
        finger_count = self._estimate_finger_count(best_gesture, features)
        
        return best_gesture, confidence, finger_count
    
    def _estimate_finger_count(self, gesture, features):
        finger_map = {
            'Fist': 0, 'Pointing': 1, 'Victory': 2, 'Three Fingers': 3,
            'Four Fingers': 4, 'Open Hand': 5, 'Stop': 5, 'Thumb Up': 1,
            'Thumb Down': 1, 'OK Sign': 2, 'Rock Sign': 2, 'Shaka': 2,
            'Call Me': 2, 'Peace': 2
        }
        return finger_map.get(gesture, 0)


class SimpleSynthesizer:
    def __init__(self):
        self.frequency = 440.0
        self.lfo_rate = 1.0
        self.volume = 0.5
        self.active = False
        self.waveform = 'sine'
        self.phase = 0.0
        self.lfo_phase = 0.0
        self.audio_buffer = []
    
    def update(self, x, y, intensity, gesture=''):
        self.frequency = 200 + (1.0 - y) * 600
        self.lfo_rate = 0.5 + x * 4.0
        self.volume = min(0.8, intensity * 0.8)
        
        if 'Pointing' in gesture:
            self.waveform = 'sine'
        elif 'Victory' in gesture:
            self.waveform = 'square'
        elif 'Three' in gesture:
            self.waveform = 'triangle'
        elif 'Fist' in gesture:
            self.waveform = 'sawtooth'
        else:
            self.waveform = 'sine'
    
    def generate_wave(self, duration=0.1, sample_rate=44100):
        import numpy as np
        
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        if self.lfo_rate > 0:
            lfo = np.sin(2 * np.pi * self.lfo_rate * (self.lfo_phase + t))
        else:
            lfo = 0
        
        freq = self.frequency * (1.0 + lfo * 0.2)
        phase_inc = 2 * np.pi * freq / sample_rate
        
        if self.waveform == 'sine':
            wave = np.sin(np.cumsum(phase_inc))
        elif self.waveform == 'square':
            wave = np.sign(np.sin(np.cumsum(phase_inc)))
        elif self.waveform == 'triangle':
            wave = 2 * np.abs(2 * (np.cumsum(phase_inc) / (2 * np.pi) % 1 - 0.5)) - 1
        elif self.waveform == 'sawtooth':
            wave = 2 * (np.cumsum(phase_inc) / (2 * np.pi) % 1) - 1
        else:
            wave = np.sin(np.cumsum(phase_inc))
        
        wave = wave * self.volume
        
        self.phase = (self.phase + np.cumsum(phase_inc)[-1]) % (2 * np.pi * 1000)
        self.lfo_phase = (self.lfo_phase + self.lfo_rate * duration) % 1.0
        
        return wave
    
    def get_params_text(self):
        return f'Freq: {int(self.frequency)}Hz | Wave: {self.waveform}'


class SimpleDrumMachine:
    def __init__(self):
        self.tempo = 120
        self.current_step = 0
        self.step_count = 16
        self.pattern = 'classic'
        self.patterns = {
            'classic': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
            'hiphop': [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,1,0],
            'house': [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0],
            'breakbeat': [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0],
        }
        self.last_step_time = 0
        self.kick_samples = []
    
    def update(self, x, y, gesture=''):
        self.tempo = 80 + int(x * 120)
        
        if 'Fist' in gesture:
            self.pattern = 'hiphop'
        elif 'Open Hand' in gesture:
            self.pattern = 'house'
        elif 'Victory' in gesture:
            self.pattern = 'breakbeat'
        else:
            self.pattern = 'classic'
    
    def step(self):
        curr_time = time.time()
        if curr_time - self.last_step_time >= 60.0 / self.tempo / 4:
            self.current_step = (self.current_step + 1) % self.step_count
            self.last_step_time = curr_time
            return True
        return False
    
    def generate_kick(self, duration=0.1, sample_rate=44100):
        import numpy as np
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freq = 150 * np.exp(-t * 20)
        wave = np.sin(2 * np.pi * np.cumsum(freq) / sample_rate)
        envelope = np.exp(-t * 10)
        return wave * envelope * 0.8
    
    def get_params_text(self):
        return f'Tempo: {self.tempo} BPM | Step: {self.current_step + 1}/{self.step_count}'


def generate_ascii_art(frame_data, gesture=''):
    brightness = frame_data.get('brightness', 128)
    char_idx = int(brightness / 255 * (len(ASCII_CHARS) - 1))
    char = ASCII_CHARS[max(0, min(len(ASCII_CHARS) - 1, char_idx))]
    
    ascii_text = f'''
    ╔═══════════════════════════════╗
    ║   ASCII ART MODE               ║
    ╠═══════════════════════════╣
    ║  Gesture: {gesture[:12]:<12}      ║
    ║  Brightness: {brightness:>3}           ║
    ╠═══════════════════════════╣
    ║                               ║
    ║        {char * 12}           ║
    ║       {char * 14}          ║
    ║      {char * 16}         ║
    ║       {char * 14}          ║
    ║        {char * 12}           ║
    ║                               ║
    ╚═══════════════════════════╝
    '''
    return ascii_text


def generate_hand_contour(cx, cy, radius=100, fingers=5):
    points = []
    num_points = 100
    
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        r = radius
        
        if fingers > 0:
            finger_interval = num_points / (fingers + 1)
            finger_phase = (i % finger_interval) / finger_interval
            finger_wave = math.sin(finger_phase * math.pi) * 0.3
            r *= (1 + finger_wave)
        
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.extend([x, y])
    
    return points


class HandGestureWidget(BoxLayout):
    fps_value = NumericProperty(0)
    status_text = StringProperty('Ready')
    gesture_text = StringProperty('None')
    
    def __init__(self, **kwargs):
        super(HandGestureWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(3)
        self.padding = dp(5)
        
        self.current_mode = MODE_CAMERA
        self.camera_index = 0
        self.fps = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.camera_active = False
        self.camera = None
        self.image_widget = None
        
        self.gesture_recognizer = AdvancedGestureRecognizer()
        self.synth = SimpleSynthesizer()
        self.drum = SimpleDrumMachine()
        
        self.prev_frame_data = None
        self.frame_buffer = deque(maxlen=5)
        self.hand_center = (0.5, 0.5)
        self.hand_radius = 0.2
        self.finger_count = 5
        
        self._build_ui()
        
        Clock.schedule_once(self._request_permissions, 0.5)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        Logger.info('HandGestureWidget: Initialized')
    
    def _build_ui(self):
        header = BoxLayout(size_hint=(1, None), height=dp(45), spacing=dp(5))
        
        self.title_label = Label(
            text='[b]Hand Gesture v2.3[/b]',
            markup=True,
            font_size=sp(16),
            halign='center'
        )
        header.add_widget(self.title_label)
        
        self.fps_label = Label(
            text='FPS: 0',
            font_size=sp(12),
            size_hint_x=0.25
        )
        header.add_widget(self.fps_label)
        
        self.add_widget(header)
        
        self.status_label = Label(
            text='Status: Initializing...',
            font_size=sp(12),
            size_hint=(1, None),
            height=dp(25),
            color=(0.5, 0.8, 0.5, 1)
        )
        self.add_widget(self.status_label)
        
        self.camera_container = BoxLayout(size_hint=(1, 0.5), spacing=dp(2))
        self.add_widget(self.camera_container)
        
        self.info_panel = BoxLayout(
            size_hint=(1, None),
            height=dp(80),
            spacing=dp(3),
            padding=dp(3)
        )
        
        info_inner = BoxLayout(orientation='vertical', spacing=dp(2))
        
        self.gesture_label = Label(
            text='Gesture: Calibrating...',
            font_size=sp(15),
            bold=True,
            color=(0.3, 0.7, 1, 1)
        )
        info_inner.add_widget(self.gesture_label)
        
        self.confidence_label = Label(
            text='Confidence: 0% | Fingers: 0',
            font_size=sp(12),
            color=(0.7, 0.7, 0.7, 1)
        )
        info_inner.add_widget(self.confidence_label)
        
        self.mode_label = Label(
            text='Mode: Camera',
            font_size=sp(12)
        )
        info_inner.add_widget(self.mode_label)
        
        self.params_label = Label(
            text='',
            font_size=sp(11),
            color=(0.8, 0.8, 0.3, 1)
        )
        info_inner.add_widget(self.params_label)
        
        self.info_panel.add_widget(info_inner)
        self.add_widget(self.info_panel)
        
        self.mode_panel = GridLayout(
            cols=5,
            size_hint=(1, None),
            height=dp(40),
            spacing=dp(2)
        )
        
        modes = [
            ('Camera', MODE_CAMERA),
            ('Gesture', MODE_GESTURE),
            ('ASCII', MODE_ASCII),
            ('Synth', MODE_SYNTH),
            ('TR-808', MODE_SAMPLER)
        ]
        
        self.mode_buttons = {}
        for text, mode in modes:
            btn = ToggleButton(
                text=text,
                font_size=sp(11),
                group='mode'
            )
            if mode == MODE_CAMERA:
                btn.state = 'down'
            btn.bind(on_press=lambda x, m=mode: self.set_mode(m))
            self.mode_panel.add_widget(btn)
            self.mode_buttons[mode] = btn
        
        self.add_widget(self.mode_panel)
        
        control_panel = BoxLayout(
            size_hint=(1, None),
            height=dp(45),
            spacing=dp(3)
        )
        
        self.btn_camera = Button(
            text='Start Camera',
            font_size=sp(12)
        )
        self.btn_camera.bind(on_press=self.toggle_camera)
        control_panel.add_widget(self.btn_camera)
        
        self.btn_switch = Button(
            text='Switch',
            font_size=sp(12),
            size_hint_x=0.35
        )
        self.btn_switch.bind(on_press=self.switch_camera)
        control_panel.add_widget(self.btn_switch)
        
        self.btn_reset = Button(
            text='Reset',
            font_size=sp(12),
            size_hint_x=0.35
        )
        self.btn_reset.bind(on_press=self.reset)
        control_panel.add_widget(self.btn_reset)
        
        self.add_widget(control_panel)
    
    def _request_permissions(self, dt):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([
                Permission.CAMERA,
                Permission.RECORD_AUDIO,
                Permission.WRITE_EXTERNAL_STORAGE
            ])
            self.status_label.text = 'Status: Permissions requested'
            Logger.info('HandGestureWidget: Permissions requested')
        except ImportError:
            self.status_label.text = 'Status: Ready (desktop mode)'
            Logger.info('HandGestureWidget: Desktop mode')
        except Exception as e:
            Logger.error(f'HandGestureWidget: Permission error - {e}')
    
    def set_mode(self, mode):
        self.current_mode = mode
        mode_names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
        self.mode_label.text = f'Mode: {mode_names[mode]}'
        self.status_label.text = f'Status: {mode_names[mode]} mode'
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == mode else 'normal'
        
        Logger.info(f'HandGestureWidget: Mode set to {mode}')
        
        if mode == MODE_ASCII:
            self._update_ascii_display()
    
    def toggle_camera(self, instance):
        if self.camera_active:
            self.stop_camera()
            instance.text = 'Start Camera'
        else:
            self.start_camera()
            instance.text = 'Stop Camera'
    
    def start_camera(self):
        try:
            self.camera_container.clear_widgets()
            self.camera = Camera(
                index=self.camera_index,
                resolution=(640, 480),
                play=True
            )
            self.camera_container.add_widget(self.camera)
            self.camera_active = True
            self.status_label.text = f'Status: Camera {self.camera_index} active'
            Logger.info(f'HandGestureWidget: Camera {self.camera_index} started')
        except Exception as e:
            Logger.error(f'HandGestureWidget: Camera error - {e}')
            self.status_label.text = 'Status: Camera error - trying fallback'
            self._start_camera_fallback()
    
    def _start_camera_fallback(self):
        try:
            self.camera_container.clear_widgets()
            self.camera = Camera(
                index=self.camera_index,
                play=True
            )
            self.camera_container.add_widget(self.camera)
            self.camera_active = True
            self.status_label.text = f'Status: Camera {self.camera_index} active (fallback)'
            Logger.info(f'HandGestureWidget: Camera fallback started')
        except Exception as e:
            Logger.error(f'HandGestureWidget: Fallback camera error - {e}')
            self.status_label.text = 'Status: Camera error'
    
    def stop_camera(self):
        if self.camera:
            self.camera.play = False
            self.camera_container.clear_widgets()
            self.camera = None
        self.camera_active = False
        self.status_label.text = 'Status: Camera stopped'
    
    def switch_camera(self, instance):
        self.stop_camera()
        self.camera_index = 1 - self.camera_index
        self.start_camera()
    
    def reset(self, instance):
        self.stop_camera()
        self.camera_index = 0
        self.current_mode = MODE_CAMERA
        self.gesture_label.text = 'Gesture: None'
        self.confidence_label.text = 'Confidence: 0% | Fingers: 0'
        self.params_label.text = ''
        self.status_label.text = 'Status: Reset'
        self.gesture_recognizer = AdvancedGestureRecognizer()
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == MODE_CAMERA else 'normal'
    
    def update(self, dt):
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
        self.prev_time = curr_time
        
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            self.fps_label.text = f'FPS: {int(self.fps)}'
        
        if self.camera_active and self.camera:
            self._process_frame()
    
    def _process_frame(self):
        frame_data = self._extract_frame_features()
        
        if self.current_mode == MODE_GESTURE:
            features = self.gesture_recognizer.extract_features(frame_data)
            
            if not self.gesture_recognizer.is_calibrated:
                if self.gesture_recognizer.calibrate(features):
                    self.status_label.text = 'Status: Calibrated!'
            else:
                gesture, confidence, fingers = self.gesture_recognizer.classify_gesture(features)
                self.gesture_label.text = f'Gesture: {gesture}'
                self.confidence_label.text = f'Confidence: {int(confidence*100)}% | Fingers: {fingers}'
                self.finger_count = fingers
                self.hand_center = (frame_data.get('center_x', 0.5), frame_data.get('center_y', 0.5))
                self.hand_radius = 0.2 + frame_data.get('motion_magnitude', 0) * 0.1
                
                self._draw_hand_contour()
        
        elif self.current_mode == MODE_ASCII:
            self._update_ascii_display(frame_data)
        
        elif self.current_mode == MODE_SYNTH:
            x = frame_data.get('center_x', 0.5)
            y = frame_data.get('center_y', 0.5)
            intensity = frame_data.get('motion_magnitude', 0)
            gesture = self.gesture_label.text.replace('Gesture: ', '')
            self.synth.update(x, y, intensity, gesture)
            self.params_label.text = self.synth.get_params_text()
        
        elif self.current_mode == MODE_SAMPLER:
            x = frame_data.get('brightness', 128) / 255.0
            y = frame_data.get('motion_magnitude', 0)
            gesture = self.gesture_label.text.replace('Gesture: ', '')
            self.drum.update(x, y, gesture)
            self.drum.step()
            self.params_label.text = self.drum.get_params_text()
        
        self.prev_frame_data = frame_data
    
    def _extract_frame_features(self):
        features = {
            'brightness': 128,
            'brightness_std': 30,
            'motion_x': 0,
            'motion_y': 0,
            'motion_magnitude': 0,
            'color_variance': 0,
            'edge_density': 0.3,
            'region_count': 1,
            'aspect_ratio': 1.0,
            'center_x': 0.5,
            'center_y': 0.5,
            'spread_x': 0.3,
            'spread_y': 0.3,
            'symmetry': 0.5,
            'compactness': 0.5,
            'texture_complexity': 0.5,
            'dominant_direction': 0.5,
            'temporal_change': 0,
        }
        
        if self.prev_frame_data:
            time_diff = time.time() - self.prev_frame_data.get('timestamp', time.time())
            if time_diff > 0:
                features['temporal_change'] = abs(
                    features['brightness'] - self.prev_frame_data.get('brightness', 128)
                ) / 255.0
        
        features['timestamp'] = time.time()
        
        t = time.time()
        features['brightness'] = 128 + int(40 * math.sin(t * 2))
        features['motion_magnitude'] = 0.1 + 0.2 * abs(math.sin(t * 3))
        features['center_x'] = 0.5 + 0.1 * math.sin(t * 1.5)
        features['center_y'] = 0.5 + 0.1 * math.cos(t * 1.5)
        
        return features
    
    def _draw_hand_contour(self):
        if not self.camera_active:
            return
        
        try:
            self.camera_container.canvas.after.clear()
            
            with self.camera_container.canvas.after:
                Color(0, 1, 0, 0.8)
                
                container_w = self.camera_container.width
                container_h = self.camera_container.height
                
                cx = container_w * self.hand_center[0]
                cy = container_h * self.hand_center[1]
                radius = min(container_w, container_h) * self.hand_radius
                
                contour_points = generate_hand_contour(cx, cy, radius, self.finger_count)
                
                if len(contour_points) >= 4:
                    Line(points=contour_points, width=2, close=True)
                
                Color(1, 0, 0, 1)
                Line(circle=(cx, cy, radius * 0.1), width=2)
                
        except Exception as e:
            Logger.error(f'HandGestureWidget: Draw contour error - {e}')
    
    def _update_ascii_display(self, frame_data=None):
        if frame_data is None:
            frame_data = {'brightness': 128}
        
        gesture = self.gesture_label.text.replace('Gesture: ', '')
        ascii_art = generate_ascii_art(frame_data, gesture)
        
        if self.image_widget:
            self.camera_container.remove_widget(self.image_widget)
        
        self.image_widget = Label(
            text=ascii_art,
            font_size=sp(10),
            markup=False
        )
        self.camera_container.clear_widgets()
        self.camera_container.add_widget(self.image_widget)


class HandGestureApp(App):
    def build(self):
        Logger.info('HandGestureApp: Building app...')
        Window.bind(on_keyboard=self._on_keyboard)
        return HandGestureWidget()
    
    def _on_keyboard(self, window, key, *args):
        if key == 27:
            return True
        return False
    
    def on_stop(self):
        Logger.info('HandGestureApp: Stopping...')


if __name__ == '__main__':
    HandGestureApp().run()
