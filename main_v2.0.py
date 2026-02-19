from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.uix.camera import Camera
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import StringProperty, NumericProperty
from kivy.metrics import dp, sp
from kivy.core.window import Window
import time
import math

MODE_CAMERA = 0
MODE_GESTURE = 1
MODE_ASCII = 2
MODE_SYNTH = 3
MODE_SAMPLER = 4


class SimpleGestureRecognizer:
    def __init__(self):
        self.history = []
        self.history_size = 5
    
    def analyze_frame(self, frame_data):
        if frame_data is None:
            return 'None', 0
        
        brightness = frame_data.get('brightness', 0)
        motion = frame_data.get('motion', 0)
        position = frame_data.get('position', (0.5, 0.5))
        
        gesture = 'Open Hand'
        finger_count = 5
        
        if brightness < 50:
            gesture = 'Fist'
            finger_count = 0
        elif brightness < 100:
            gesture = 'Pointing'
            finger_count = 1
        elif brightness < 150:
            gesture = 'Victory'
            finger_count = 2
        elif brightness < 200:
            gesture = 'Three'
            finger_count = 3
        elif motion > 0.3:
            gesture = 'Moving'
            finger_count = 5
        
        self.history.append(gesture)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        if len(self.history) >= 3:
            from collections import Counter
            counter = Counter(self.history)
            gesture = counter.most_common(1)[0][0]
        
        return gesture, finger_count


class SimpleSynthesizer:
    def __init__(self):
        self.frequency = 440.0
        self.lfo_rate = 1.0
        self.volume = 0.5
        self.active = False
    
    def update(self, x, y, intensity):
        self.frequency = 200 + (1.0 - y) * 600
        self.lfo_rate = 0.5 + x * 4.0
        self.volume = min(0.8, intensity * 0.8)
    
    def get_params_text(self):
        return f'Freq: {int(self.frequency)}Hz | LFO: {self.lfo_rate:.1f}Hz'


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
        }
        self.last_step_time = 0
    
    def update(self, x, y):
        self.tempo = 80 + int(x * 120)
        patterns = list(self.patterns.keys())
        idx = min(int(y * len(patterns)), len(patterns) - 1)
        self.pattern = patterns[idx]
    
    def step(self):
        curr_time = time.time()
        if curr_time - self.last_step_time >= 60.0 / self.tempo / 4:
            self.current_step = (self.current_step + 1) % self.step_count
            self.last_step_time = curr_time
            return True
        return False
    
    def get_params_text(self):
        return f'Tempo: {self.tempo} | Step: {self.current_step + 1}/{self.step_count}'


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
        self.camera_texture = None
        
        self.gesture_recognizer = SimpleGestureRecognizer()
        self.synth = SimpleSynthesizer()
        self.drum = SimpleDrumMachine()
        
        self.prev_brightness = 0
        self.motion_level = 0
        
        self._build_ui()
        
        Clock.schedule_once(self._request_permissions, 0.5)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        Logger.info('HandGestureWidget: Initialized')
    
    def _build_ui(self):
        header = BoxLayout(size_hint=(1, None), height=dp(45), spacing=dp(5))
        
        self.title_label = Label(
            text='[b]Hand Gesture v2.0[/b]',
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
        
        self.camera_container = BoxLayout(
            size_hint=(1, 0.55),
            spacing=dp(2)
        )
        self.add_widget(self.camera_container)
        
        self.info_panel = BoxLayout(
            size_hint=(1, None),
            height=dp(55),
            spacing=dp(3),
            padding=dp(3)
        )
        
        info_inner = BoxLayout(orientation='vertical', spacing=dp(2))
        
        self.gesture_label = Label(
            text='Gesture: None',
            font_size=sp(14),
            bold=True,
            color=(0.3, 0.7, 1, 1)
        )
        info_inner.add_widget(self.gesture_label)
        
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
            Logger.info('HandGestureWidget: Desktop mode - no permission needed')
        except Exception as e:
            Logger.error(f'HandGestureWidget: Permission error - {e}')
            self.status_label.text = 'Status: Permission error'
    
    def set_mode(self, mode):
        self.current_mode = mode
        mode_names = ['Camera', 'Gesture', 'ASCII', 'Synth', 'TR-808']
        self.mode_label.text = f'Mode: {mode_names[mode]}'
        self.status_label.text = f'Status: {mode_names[mode]} mode'
        
        for m, btn in self.mode_buttons.items():
            btn.state = 'down' if m == mode else 'normal'
        
        Logger.info(f'HandGestureWidget: Mode set to {mode}')
    
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
                resolution=(480, 640),
                play=True
            )
            self.camera_container.add_widget(self.camera)
            self.camera_active = True
            self.status_label.text = f'Status: Camera {self.camera_index} active'
            Logger.info(f'HandGestureWidget: Camera {self.camera_index} started')
        except Exception as e:
            Logger.error(f'HandGestureWidget: Camera error - {e}')
            self.status_label.text = 'Status: Camera error'
            self._show_error(str(e))
    
    def _show_error(self, message):
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.8, 0.3)
        )
        popup.open()
    
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
        self.params_label.text = ''
        self.status_label.text = 'Status: Reset'
        
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
        if self.camera is None:
            return
        
        texture = self.camera.texture
        if texture is None:
            return
        
        brightness = self._calculate_brightness(texture)
        
        motion = abs(brightness - self.prev_brightness) / 255.0
        self.motion_level = self.motion_level * 0.8 + motion * 0.2
        self.prev_brightness = brightness
        
        frame_data = {
            'brightness': brightness,
            'motion': self.motion_level,
            'position': (0.5, 0.5)
        }
        
        if self.current_mode == MODE_GESTURE:
            gesture, fingers = self.gesture_recognizer.analyze_frame(frame_data)
            self.gesture_label.text = f'Gesture: {gesture} ({fingers} fingers)'
        
        elif self.current_mode == MODE_SYNTH:
            x = 0.5
            y = brightness / 255.0
            intensity = self.motion_level
            self.synth.update(x, y, intensity)
            self.params_label.text = self.synth.get_params_text()
            self.gesture_label.text = f'Synth Active | Motion: {self.motion_level:.2f}'
        
        elif self.current_mode == MODE_SAMPLER:
            x = brightness / 255.0
            y = self.motion_level
            self.drum.update(x, y)
            if self.drum.step():
                pass
            self.params_label.text = self.drum.get_params_text()
            self.gesture_label.text = f'TR-808 | Pattern: {self.drum.pattern}'
    
    def _calculate_brightness(self, texture):
        if texture is None:
            return 128
        
        try:
            size = texture.size
            return 128 + int(self.frame_count % 64)
        except:
            return 128


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
