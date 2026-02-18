import numpy as np
import time
import threading
from collections import deque
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.core.camera import Camera

try:
    from android_camera import get_camera_instance
except ImportError:
    get_camera_instance = None

MODE_GESTURE = 0
MODE_SYNTH = 1

ASCII_CHARS = " .',:;ilI1|\\/()[]{}?_-+~=<>!@#$%&*#"

class AndroidAudioSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        self.frequency = 440.0
        self.frequency2 = 550.0
        self.detune = 0.0
        self.lfo_rate = 1.0
        self.lfo_depth = 0.3
        self.lfo_type = 0
        self.envelope = 0.0
        self.envelope_target = 0.0
        self.phase = 0.0
        self.phase2 = 0.0
        self.lfo_phase = 0.0
        self.color_hue = 0.0
        self.color_sat = 0.0
        self.volume = 0.3
        self.delay_buffer = np.zeros(int(sample_rate * 0.5))
        self.delay_index = 0
        self.audio_track = None
        self.buffer_size = 2048
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
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
            
            self.buffer_size = max(2048, AudioTrack.getMinBufferSize(
                self.sample_rate, channel_config, audio_format))
            
            self.audio_track = AudioTrack(
                AudioManager.STREAM_MUSIC,
                self.sample_rate,
                channel_config,
                audio_format,
                self.buffer_size,
                AudioTrack.MODE_STREAM
            )
            
            self.audio_track.play()
            Logger.info("AndroidAudioSynthesizer: AudioTrack started")
            
            while self.running:
                frames = self.buffer_size // 2
                wave = self._generate_audio(frames)
                audio_data = (np.clip(wave, -1, 1) * 32767).astype(np.int16).tobytes()
                self.audio_track.write(audio_data, 0, len(audio_data))
                
        except Exception as e:
            Logger.error(f"AndroidAudioSynthesizer: Audio error: {e}")
            self.running = False
    
    def _generate_audio(self, frames):
        t = np.arange(frames) / self.sample_rate
        
        if self.lfo_type == 0:
            lfo = np.sin(2 * np.pi * self.lfo_rate * (self.lfo_phase + t))
        elif self.lfo_type == 1:
            lfo = np.sign(np.sin(2 * np.pi * self.lfo_rate * (self.lfo_phase + t)))
        else:
            lfo = 2 * np.abs(2 * (self.lfo_phase + t) * self.lfo_rate % 1 - 0.5) - 1
        
        lfo_mod = 1.0 + lfo * self.lfo_depth
        freq1 = self.frequency * lfo_mod * (1.0 + self.color_hue * 0.1)
        phase_inc1 = 2 * np.pi * freq1 / self.sample_rate
        self.phase = self.phase + np.cumsum(phase_inc1)
        osc1 = np.sin(self.phase)
        
        freq2 = self.frequency2 * (1.0 + self.detune * 0.1) * lfo_mod
        phase_inc2 = 2 * np.pi * freq2 / self.sample_rate
        self.phase2 = self.phase2 + np.cumsum(phase_inc2)
        osc2 = np.sin(self.phase2)
        
        wave = osc1 * 0.6 + osc2 * 0.4
        envelope_diff = self.envelope_target - self.envelope
        if abs(envelope_diff) > 0.001:
            rate = 10.0 if envelope_diff > 0 else 5.0
            self.envelope += np.sign(envelope_diff) * min(abs(envelope_diff), rate / self.sample_rate * frames)
        
        wave = wave * self.envelope * self.volume
        
        for i in range(frames):
            delayed = self.delay_buffer[self.delay_index] * 0.3
            wave[i] = wave[i] + delayed
            self.delay_buffer[self.delay_index] = wave[i]
            self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
        
        self.phase = self.phase[-1] % (2 * np.pi * 1000)
        self.phase2 = self.phase2[-1] % (2 * np.pi * 1000)
        self.lfo_phase = (self.lfo_phase + self.lfo_rate * frames / self.sample_rate) % 1.0
        
        return wave
    
    def update_from_motion(self, motion_x, motion_y, motion_amount):
        self.frequency = 80 + (1.0 - motion_y) * 700
        self.frequency2 = self.frequency * 1.5
        self.lfo_depth = min(0.6, motion_amount * 0.8)
        self.envelope_target = min(1.0, motion_amount * 1.5)
        self.lfo_rate = 0.3 + motion_x * 4.0
        self.detune = motion_amount * 5.0


class HandGestureApp(App):
    def build(self):
        self.current_mode = MODE_GESTURE
        self.synth = None
        self.fps_smooth = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.last_frame = None
        self.motion_x = 0.5
        self.motion_y = 0.5
        self.motion_amount = 0.0
        
        layout = BoxLayout(orientation='vertical')
        
        self.image_widget = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.image_widget, size_hint=(1, 0.85))
        
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=5, padding=5)
        
        self.btn_gesture = ToggleButton(text='Camera', group='mode', state='down')
        self.btn_gesture.bind(on_press=lambda x: self.set_mode(MODE_GESTURE))
        btn_layout.add_widget(self.btn_gesture)
        
        self.btn_synth = ToggleButton(text='Synth', group='mode')
        self.btn_synth.bind(on_press=lambda x: self.set_mode(MODE_SYNTH))
        btn_layout.add_widget(self.btn_synth)
        
        btn_reset = Button(text='Reset')
        btn_reset.bind(on_press=self.reset_motion)
        btn_layout.add_widget(btn_reset)
        
        layout.add_widget(btn_layout)
        
        if get_camera_instance:
            self.camera = get_camera_instance(camera_index=0, width=640, height=480)
            if not self.camera.open():
                Logger.error("HandGestureApp: Failed to open camera")
        else:
            Logger.warn("No camera module available")
            self.camera = None
        
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        
        return layout
    
    def set_mode(self, mode):
        if mode == MODE_SYNTH and self.synth is None:
            self.synth = AndroidAudioSynthesizer()
            self.synth.start()
        elif mode != MODE_SYNTH and self.synth is not None:
            self.synth.stop()
            self.synth = None
        
        self.current_mode = mode
    
    def reset_motion(self, instance):
        self.motion_x = 0.5
        self.motion_y = 0.5
        self.motion_amount = 0.0
        Logger.info("Motion reset")
    
    def process_motion(self, frame):
        if frame is None:
            return
        
        h, w = frame.shape[:2]
        
        if self.last_frame is not None:
            gray = np.zeros((h, w), dtype=np.uint8)
            for y in range(0, h, 4):
                for x in range(0, w, 4):
                    diff = abs(int(frame[y, x, 0]) - int(self.last_frame[y, x, 0]))
                    diff += abs(int(frame[y, x, 1]) - int(self.last_frame[y, x, 1]))
                    diff += abs(int(frame[y, x, 2]) - int(self.last_frame[y, x, 2]))
                    gray[y, x] = min(255, diff)
            
            total_motion = np.sum(gray) / 255
            max_motion = (h // 4) * (w // 4)
            self.motion_amount = min(1.0, total_motion / (max_motion * 0.1))
            
            if total_motion > 100:
                y_coords, x_coords = np.where(gray > 50)
                if len(x_coords) > 0:
                    self.motion_x = np.mean(x_coords) / w
                    self.motion_y = np.mean(y_coords) / h
        
        self.last_frame = frame.copy()
    
    def update_frame(self, dt):
        if self.camera is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText = lambda img, text, pos, font, scale, color, thickness: None
            import sys
            sys.modules['cv2'] = type(sys)('cv2')
            sys.modules['cv2'].putText = lambda img, text, pos, font, scale, color, thickness: None
            display_image = placeholder
        else:
            ret, frame = self.camera.read()
            if not ret:
                return
            
            display_image = frame.copy()
            display_image = display_image[:, ::-1, :]
        
        self.process_motion(display_image)
        
        if self.current_mode == MODE_SYNTH and self.synth:
            self.synth.update_from_motion(self.motion_x, self.motion_y, self.motion_amount)
        
        curr_time = time.time()
        frame_time = curr_time - self.prev_time
        if frame_time > 0:
            fps = 1 / frame_time
            self.fps_smooth = self.fps_smooth * 0.9 + fps * 0.1
        self.prev_time = curr_time
        
        try:
            import cv2
            cv2.putText(display_image, f'FPS:{int(self.fps_smooth)}', (10, 30), 
                       cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            
            mode_text = {MODE_GESTURE: 'Camera', MODE_SYNTH: 'Synth'}
            cv2.putText(display_image, mode_text.get(self.current_mode, ''), (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if self.current_mode == MODE_SYNTH and self.synth:
                cv2.putText(display_image, f'Freq:{int(self.synth.frequency)}Hz LFO:{self.synth.lfo_rate:.1f}Hz', 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        except:
            pass
        
        buf = display_image.tobytes()
        texture = Texture.create(size=(display_image.shape[1], display_image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture
    
    def on_stop(self):
        if self.synth:
            self.synth.stop()
        if self.camera:
            self.camera.release()


if __name__ == '__main__':
    HandGestureApp().run()
