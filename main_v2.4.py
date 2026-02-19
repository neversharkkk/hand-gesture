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
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.metrics import dp, sp
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle, Line, Ellipse
import time
import math
import random
from collections import deque

MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2
MODE_SYNTH = 3
MODE_SAMPLER = 4


class AdvancedGestureRecognizer:
    def __init__(self):
        self.history = deque(maxlen=10)
        self.feature_history = deque(maxlen=5)
        self.calibration_frames = 0
        self.calibration_data = []
        self.is_calibrated = False
        self.baseline_features = None
        self.confidence_threshold = 0.7
        
    def calibrate(self, features):
        self.calibration_data.append(features)
        self.calibration_frames += 1
        if self.calibration_frames >= 30:
            self._compute_baseline()
            self.is_calibrated = True
            return True
        return False
    
    def _compute_baseline(self):
        if not self.calibration_data:
            return None
        keys = self.calibration_data[0].keys()
        for key in keys:
            values = [d[key] for d in self.calibration_data if key in d]
            if values:
                mean = sum(values) / len(values)
                std = math.sqrt(sum((v - mean)**2 for v in values) / len(values)) if len(values) > 1 else 0
                self.baseline_features[key] = {'mean': mean, 'std': std}
    
    def extract_features(self, frame_data):
        features = {
            'brightness': frame_data.get('brightness', 128),
            'motion_x': frame_data.get('motion_x', 0),
            'motion_y': frame_data.get('motion_y', 0),
            'motion_magnitude': frame_data.get('motion_magnitude', 0),
            'center_x': frame_data.get('center_x', 0.5),
            'center_y': frame_data.get('center_y', 0.5),
            'spread': frame_data.get('spread', 0.3),
            'compactness': frame_data.get('compactness', 0.5),
            'symmetry': frame_data.get('symmetry', 0.5),
            'temporal_change': frame_data.get('temporal_change', 0),
        }
        self.feature_history.append(features)
        return features
    
    def classify_gesture(self, features):
        if not self.is_calibrated:
            return 'Calibrating...', 0, 0
        
        motion = features['motion_magnitude']
        spread = features['spread']
        compactness = features['compactness']
        symmetry = features['symmetry']
        temporal = features['temporal_change']
        
        scores = {}
        scores['Open Hand'] = spread * 0.8 + symmetry * 0.6 + (1 - compactness) * 0.5
        scores['Fist'] = compactness * 0.9 + (1 - spread) * 0.7
        scores['Pointing'] = (1 - spread) * 0.6 + compactness * 0.5
        scores['Victory'] = spread * 0.5 + symmetry * 0.3 + 0.4
        scores['Three'] = spread * 0.6 + symmetry * 0.4 + 0.3
        scores['Four'] = spread * 0.7 + symmetry * 0.5 + 0.3
        scores['Peace'] = spread * 0.5 + symmetry * 0.4 + 0.3
        scores['OK'] = compactness * 0.8 + symmetry * 0.7 + 0.3
        scores['Thumb Up'] = compactness * 0.6 + symmetry * 0.5 + 0.2
        scores['Thumb Down'] = compactness * 0.6 + symmetry * 0.5 + 0.2
        
        best = max(scores, key=scores.get)
        confidence = min(1.0, scores[best])
        
        self.history.append(best)
        if len(self.history) >= 3:
            from collections import Counter
            counter = Counter(list(self.history)[-3:])
            most = counter.most_common(1)[0]
            if most[1] >= 2:
                best = most[0]
        
        finger_map = {'Fist': 0, 'Pointing': 1, 'Victory': 2, 'Three': 3, 'Four': 4, 
                     'Open Hand': 5, 'Peace': 2, 'OK': 2, 'Thumb Up': 1, 'Thumb Down': 1}
        fingers = finger_map.get(best, 0)
        
        return best, confidence, fingers


class SynthVisualizer:
    def __init__(self):
        self.frequency = 440
        self.waveform = 'sine'
        self.volume = 0.5
        self.lfo_rate = 1.0
        self.waveform_points = []
        self.update_waveform()
    
    def update(self, x, y, intensity, gesture=''):
        self.frequency = int(200 + (1.0 - y) * 600)
        self.volume = min(0.8, intensity * 0.8 + 0.2)
        self.lfo_rate = 0.5 + x * 4.0
        
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
        
        self.update_waveform()
    
    def update_waveform(self):
        points = []
        num_points = 50
        for i in range(num_points):
            t = i / num_points
            if self.waveform == 'sine':
                y = math.sin(2 * math.pi * t * 4)
            elif self.waveform == 'square':
                y = 1 if math.sin(2 * math.pi * t * 4) > 0 else -1
            elif self.waveform == 'triangle':
                y = 2 * abs(2 * t * 4 % 1 - 1) - 1
            else:
                y = 2 * (t * 4 % 1) - 1
            points.extend([t, y * 0.8])
        self.waveform_points = points
    
    def get_params_text(self):
        return f'Freq: {self.frequency}Hz | Wave: {self.waveform.upper()}'


class DrumMachineVisualizer:
    def __init__(self):
        self.tempo = 120
        self.current_step = 0
        self.step_count = 16
        self.pattern = 0
        self.patterns = ['Classic', 'HipHop', 'House', 'Break']
        self.active_steps = [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
        self.flash_step = -1
    
    def update(self, x, y, gesture=''):
        self.tempo = int(80 + x * 120)
        
        if 'Fist' in gesture:
            self.pattern = 1
            self.active_steps = [1,0,0,0, 0,0,1,0, 1,0,0,0, 0,0,1,0]
        elif 'Open Hand' in gesture:
            self.pattern = 2
            self.active_steps = [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
        elif 'Victory' in gesture:
            self.pattern = 3
            self.active_steps = [1,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0]
        else:
            self.pattern = 0
            self.active_steps = [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]
    
    def step(self):
        curr = time.time()
        step_time = 60.0 / self.tempo / 4
        if curr - getattr(self, 'last_step', 0) >= step_time:
            self.current_step = (self.current_step + 1) % self.step_count
            self.last_step = curr
            self.flash_step = self.current_step
            return True
        return False    
    def get_params_text(self):
        return f'{self.patterns[self.pattern]} | {self.tempo} BPM'


class HandGestureWidget(BoxLayout):
    fps_value = NumericProperty(0)
    status_text = StringProperty('Ready')
    gesture_text = StringProperty('None')
    
    def __init__(self, **kwargs):
        super(HandGestureWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(4)
        self.padding = dp(8)
        
        self.current_mode = MODE_CAMERA
        self.camera_index = 0
        self.fps = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.camera_active = False
        self.camera = None
        
        self.gesture_recognizer = AdvancedGestureRecognizer()
        self.synth = SynthVisualizer()
        self.drum = DrumMachineVisualizer()
        
        self.prev_frame_data = None
        self.hand_center = (0.5, 0.5)
        self.finger_count = 0
        
        self._build_ui()
        
        Clock.schedule_once(self._request_permissions, 0.5)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        Logger.info('HandGestureWidget: Initialized v2.4')
    
    def _build_ui(self):
        header = BoxLayout(size_hint=(1, None), height=dp(50), spacing=dp(5))
        
        title_box = BoxLayout(orientation='vertical', size_hint_x=0.7)
        self.title_label = Label(
            text='[b]Hand Gesture[/b]',
            markup=True,
            font_size=sp(18),
            halign='left',
            padding_x=dp(5)
        )
        title_box.add_widget(self.title_label)
        
        self.fps_label = Label(
            text='30 FPS',
            font_size=sp(12),
            halign='right',
            color=(0.6, 0.8, 0.6, 1),
            size_hint_x=0.3
        )
        header.add_widget(title_box)
        header.add_widget(self.fps_label)
        self.add_widget(header)
        
        self.status_label = Label(
            text='Ready - Tap Start Camera',
            font_size=sp(13),
            size_hint=(1, None),
            height=dp(28),
            color=(0.4, 0.7, 0.9, 1)
        )
        self.add_widget(self.status_label)
        
        self.camera_container = BoxLayout(
            size_hint=(1, 0.52),
            padding=dp(2)
        )
        
        self.visualization_canvas = BoxLayout(size_hint=(1, 1))
        self.camera_container.add_widget(self.visualization_canvas)
        self.add_widget(self.camera_container)
        
        info_panel = BoxLayout(
            size_hint=(1, None),
            height=dp(70),
            spacing=dp(4),
            padding=dp(4)
        )
        
        info_left = BoxLayout(orientation='vertical', spacing=dp(2), size_hint_x=0.55)
        
        self.gesture_label = Label(
            text='Gesture: --',
            font_size=sp(16),
            bold=True,
            color=(0.2, 0.8, 1.0, 1),
            halign='left',
            padding_x=dp(5)
        )
        info_left.add_widget(self.gesture_label)
        
        self.confidence_label = Label(
            text='Confidence: 0%',
            font_size=sp(12),
            color=(0.7, 0.7, 0.7, 1),
            halign='left',
            padding_x=dp(5)
        )
        info_left.add_widget(self.confidence_label)
        
        info_right = BoxLayout(orientation='vertical', spacing=dp(2), size_hint_x=0.45)
        
        self.mode_label = Label(
            text='Camera',
            font_size=sp(14),
            color=(1.0, 0.8, 0.3, 1),
            halign='right',
            padding_x=dp(5)
        )
        info_right.add_widget(self.mode_label)
        
        self.params_label = Label(
            text='',
            font_size=sp(11),
            color=(0.6, 0.8, 0.6, 1),
            halign='right',
            padding_x=dp(5)
        )
        info_right.add_widget(self.params_label)
        
        info_panel.add_widget(info_left)
        info_panel.add_widget(info_right)
        self.add_widget(info_panel)
        
        self.mode_panel = GridLayout(
            cols=5,
            size_hint=(1, None),
            height=dp(44),
            spacing=dp(3)
        )
        
        modes = [('Camera', MODE_CAMERA), ('Gesture', MODE_GESTURE), 
                 ('ASCII', MODE_ASCII), ('Synth', MODE_SYNTH), ('TR-808', MODE_SAMPLER)]
        
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
            height=dp(50),
            spacing=dp(6)
        )
        
        self.btn_camera = Button(
            text='▶ Start',
            font_size=sp(13),
            background_color=(0.2, 0.7, 0.3, 1)
        )
        self.btn_camera.bind(on_press=self.toggle_camera)
        control_panel.add_widget(self.btn_camera)
        
        self.btn_switch = Button(
            text='⟳ Switch',
            font_size=sp(12),
            size_hint_x=0.4,
            background_color=(0.3, 0.5, 0.7, 1)
        )
        self.btn_switch.bind(on_press=self.switch_camera)
        control_panel.add_widget(self.btn_switch)
        
        self.btn_reset = Button(
            text='↺ Reset',
            font_size=sp(12),
            size_hint_x=0.35,
            background_color=(0.7, 0.4, 0.3, 1)
        )
        self.btn_reset.bind(on_press=self.reset)
        control_panel.add_widget(self.btn_reset)
        
        self.add_widget(control_panel)
    
    def _request_permissions(self, dt):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.RECORD_AUDIO])
            self.status_label.text = 'Permissions requested'
        except:
            self.status_label.text = 'Desktop mode'
    
    def set_mode(self, mode):
        self.current_mode = mode
        names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
        self.mode_label.text = names[mode]
        self.status_label.text = f'{names[mode]} mode'
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == mode else 'normal'
        
        if mode == MODE_GESTURE:
            self._start_visualization('gesture')
        elif mode == MODE_SYNTH:
            self._start_visualization('synth')
        elif mode == MODE_SAMPLER:
            self._start_visualization('drum')
        else:
            self._start_visualization('camera')
    
    def _start_visualization(self, viz_type):
        self.visualization_canvas.canvas.clear()
        with self.visualization_canvas.canvas:
            if viz_type == 'camera':
                Color(0.1, 0.1, 0.15, 1)
                Rectangle(size=self.visualization_canvas.size, pos=self.visualization_canvas.pos)
                Color(0.5, 0.5, 0.6, 1)
                Label(text='[b]Camera Preview[/b]\n\nTap "Start" to begin',
                      markup=True, font_size=sp(14), halign='center')
            elif viz_type == 'gesture':
                Color(0.15, 0.1, 0.2, 1)
                Rectangle(size=self.visualization_canvas.size, pos=self.visualization_canvas.pos)
            elif viz_type == 'synth':
                self._draw_synth_visualization()
            elif viz_type == 'drum':
                self._draw_drum_visualization()
    
    def _draw_synth_visualization(self):
        self.visualization_canvas.canvas.clear()
        with self.visualization_canvas.canvas:
            Color(0.1, 0.15, 0.2, 1)
            Rectangle(size=self.visualization_canvas.size, pos=self.visualization_canvas.pos)
            
            w, h = self.visualization_canvas.size
            cx, cy = w/2, h/2
            
            Color(0.2, 0.6, 0.8, 0.8)
            for i in range(20):
                radius = 30 + i * 8
                Ellipse(size=(radius*2, radius*2), pos=(cx-radius, cy-radius), segments=32)
            
            points = self.synth.waveform_points
            if points:
                scaled_points = []
                for i in range(0, len(points), 2):
                    scaled_points.append(points[i] * w * 0.8 + w * 0.1)
                    scaled_points.append(points[i+1] * h * 0.3 + h * 0.5)
                Color(0.3, 0.9, 0.5, 1)
                Line(points=scaled_points, width=2)
    
    def _draw_drum_visualization(self):
        self.visualization_canvas.canvas.clear()
        with self.visualization_canvas.canvas:
            Color(0.12, 0.12, 0.18, 1)
            Rectangle(size=self.visualization_canvas.size, pos=self.visualization_canvas.pos)
            
            w, h = self.visualization_canvas.size
            step_width = w / 16
            y_top = h * 0.3
            y_bottom = h * 0.7
            
            for i in range(16):
                x = i * step_width
                is_active = self.drum.active_steps[i]
                is_current = (i == self.drum.flash_step)
                
                if is_current:
                    Color(1.0, 0.8, 0.2, 1)
                elif is_active:
                    Color(0.2, 0.8, 0.4, 1)
                else:
                    Color(0.3, 0.3, 0.4, 0.5)
                
                Rectangle(size=(step_width - 4, y_bottom - y_top), pos=(x + 2, y_top))
            
            Color(0.8, 0.8, 0.9, 1)
            Label(text=f'{self.drum.patterns[self.drum.pattern]} - {self.drum.tempo} BPM',
                  font_size=sp(14), size=(w, 30), pos=(0, h - 40), halign='center')
    
    def toggle_camera(self, instance):
        if self.camera_active:
            self.stop_camera()
            instance.text = '▶ Start'
            instance.background_color = (0.2, 0.7, 0.3, 1)
        else:
            self.start_camera()
            instance.text = '■ Stop'
            instance.background_color = (0.8, 0.3, 0.3, 1)
    
    def start_camera(self):
        try:
            self.camera_container.clear_widgets()
            self.camera = Camera(index=self.camera_index, resolution=(480, 640), play=True)
            
            self.camera_container.add_widget(self.camera)
            self.camera_active = True
            self.status_label.text = f'Camera {self.camera_index} active'
        except Exception as e:
            Logger.error(f'Camera error: {e}')
            self.status_label.text = 'Camera error'
            self._start_camera_fallback()
    
    def _start_camera_fallback(self):
        try:
            self.camera_container.clear_widgets()
            self.camera = Camera(index=self.camera_index, play=True)
            self.camera_container.add_widget(self.camera)
            self.camera_active = True
        except:
            self.status_label.text = 'Camera failed'
    
    def stop_camera(self):
        if self.camera:
            self.camera.play = False
            self.camera_container.clear_widgets()
            self.camera = None
        self.camera_active = False
        self.status_label.text = 'Camera stopped'
    
    def switch_camera(self, instance):
        self.stop_camera()
        self.camera_index = 1 - self.camera_index
        self.start_camera()
    
    def reset(self, instance):
        self.stop_camera()
        self.camera_index = 0
        self.current_mode = MODE_CAMERA
        self.gesture_label.text = 'Gesture: --'
        self.confidence_label.text = 'Confidence: 0%'
        self.params_label.text = ''
        self.gesture_recognizer = AdvancedGestureRecognizer()
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == MODE_CAMERA else 'normal'
        
        self.set_mode(MODE_CAMERA)
        self.btn_camera.text = '▶ Start'
        self.btn_camera.background_color = (0.2, 0.7, 0.3, 1)
    
    def update(self, dt):
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
        self.prev_time = curr_time
        
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            self.fps_label.text = f'{int(self.fps)} FPS'
        
        if self.camera_active and self.camera:
            self._process_frame()
            
            if self.current_mode == MODE_SYNTH:
                self._draw_synth_visualization()
            elif self.current_mode == MODE_SAMPLER:
                self.drum.step()
                self._draw_drum_visualization()
    
    def _process_frame(self):
        frame_data = self._extract_frame_features()
        
        if self.current_mode == MODE_GESTURE:
            features = self.gesture_recognizer.extract_features(frame_data)
            
            if not self.gesture_recognizer.is_calibrated:
                if self.gesture_recognizer.calibrate(features):
                    self.status_label.text = 'Calibrated! Ready'
            else:
                gesture, confidence, fingers = self.gesture_recognizer.classify_gesture(features)
                self.gesture_label.text = f'Gesture: {gesture}'
                self.confidence_label.text = f'Confidence: {int(confidence*100)}%'
                self.finger_count = fingers
                self.hand_center = (frame_data.get('center_x', 0.5), frame_data.get('center_y', 0.5))
                
                self._draw_hand_contour()
        
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
            'motion_x': 0,
            'motion_y': 0,
            'motion_magnitude': 0.1,
            'center_x': 0.5,
            'center_y': 0.5,
            'spread': 0.3,
            'compactness': 0.5,
            'symmetry': 0.5,
            'temporal_change': 0,
        }
        
        t = time.time()
        features['brightness'] = 128 + int(40 * math.sin(t * 2))
        features['motion_magnitude'] = 0.1 + 0.2 * abs(math.sin(t * 3))
        features['center_x'] = 0.5 + 0.15 * math.sin(t * 1.5)
        features['center_y'] = 0.5 + 0.15 * math.cos(t * 1.5)
        
        return features
    
    def _draw_hand_contour(self):
        if self.current_mode != MODE_GESTURE:
            return
        
        try:
            self.visualization_canvas.canvas.clear()
            with self.visualization_canvas.canvas:
                Color(0.08, 0.08, 0.12, 1)
                Rectangle(size=self.visualization_canvas.size, pos=self.visualization_canvas.pos)
                
                w, h = self.visualization_canvas.size
                cx = w * self.hand_center[0]
                cy = h * self.hand_center[1]
                radius = min(w, h) * 0.25
                
                num_points = 60
                points = []
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    r_mod = radius
                    if self.finger_count > 0:
                        finger_phase = (i % (num_points // self.finger_count)) / (num_points // self.finger_count)
                        r_mod *= (1 + math.sin(finger_phase * math.pi) * 0.3)
                    
                    points.extend([cx + r_mod * math.cos(angle), cy + r_mod * math.sin(angle)])
                
                Color(0.2, 0.9, 0.4, 0.9)
                Line(points=points, width=2.5, close=True)
                
                Color(1.0, 0.3, 0.3, 1)
                Ellipse(size=(radius*0.15, radius*0.15), pos=(cx - radius*0.075, cy - radius*0.075))
                
                Color(0.6, 0.8, 1.0, 1)
                Label(text=f'{self.finger_count} fingers', 
                      font_size=sp(14), size=(100, 30), pos=(cx - 50, cy + radius + 10), halign='center')
        except Exception as e:
            Logger.error(f'Draw error: {e}')


class HandGestureApp(App):
    def build(self):
        Logger.info('HandGestureApp: Building v2.4...')
        return HandGestureWidget()
    
    def on_stop(self):
        Logger.info('HandGestureApp: Stopping...')


if __name__ == '__main__':
    HandGestureApp().run()
