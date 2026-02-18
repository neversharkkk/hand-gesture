import numpy as np
import time
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.properties import StringProperty

MODE_CAMERA = 0
MODE_SYNTH = 1


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


class HandGestureApp(App):
    fps_text = StringProperty('FPS: 0')
    mode_text = StringProperty('Camera')
    synth_text = StringProperty('Freq: 440Hz')
    
    def build(self):
        self.current_mode = MODE_CAMERA
        self.synth = None
        self.fps = 30
        self.prev_time = time.time()
        self.motion_x = 0.5
        self.motion_y = 0.5
        self.motion_amount = 0.0
        self.frame_count = 0
        
        layout = BoxLayout(orientation='vertical')
        
        self.display = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.display, size_hint=(1, 0.85))
        
        btn_layout = BoxLayout(size_hint=(1, 0.15), spacing=5, padding=5)
        
        self.btn_camera = ToggleButton(text='Camera', group='mode', state='down')
        self.btn_camera.bind(on_press=lambda x: self.set_mode(MODE_CAMERA))
        btn_layout.add_widget(self.btn_camera)
        
        self.btn_synth = ToggleButton(text='Synth', group='mode')
        self.btn_synth.bind(on_press=lambda x: self.set_mode(MODE_SYNTH))
        btn_layout.add_widget(self.btn_synth)
        
        btn_reset = Button(text='Reset')
        btn_reset.bind(on_press=self.reset_motion)
        btn_layout.add_widget(btn_reset)
        
        layout.add_widget(btn_layout)
        
        self._update_display()
        
        Clock.schedule_interval(self.update_loop, 1.0 / 10.0)
        
        return layout
    
    def set_mode(self, mode):
        if mode == MODE_SYNTH:
            if self.synth is None:
                try:
                    self.synth = AndroidAudioSynthesizer()
                    self.synth.start()
                    self.mode_text = 'Synth'
                except Exception as e:
                    Logger.error(f"Synth error: {e}")
        else:
            if self.synth:
                try:
                    self.synth.stop()
                except:
                    pass
                self.synth = None
            self.mode_text = 'Camera'
        self.current_mode = mode
    
    def reset_motion(self, instance):
        self.motion_x = 0.5
        self.motion_y = 0.5
        self.motion_amount = 0.0
    
    def _update_display(self):
        try:
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            
            h, w = display.shape[:2]
            
            if self.motion_amount > 0.05:
                mx = int(self.motion_x * w)
                my = int(self.motion_y * h)
                radius = int(20 + self.motion_amount * 60)
                
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx * dx + dy * dy <= radius * radius:
                            px, py = mx + dx, my + dy
                            if 0 <= px < w and 0 <= py < h:
                                display[py, px] = [0, 255, 0]
            
            cv2 = self._get_cv2()
            if cv2:
                cv2.putText(display, self.fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
                cv2.putText(display, self.mode_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                if self.current_mode == MODE_SYNTH and self.synth:
                    cv2.putText(display, f'Freq: {int(self.synth.frequency)}Hz', 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            
            buf = display.tobytes()
            texture = Texture.create(size=(640, 480), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.display.texture = texture
            
        except Exception as e:
            Logger.error(f"Display error: {e}")
    
    def _get_cv2(self):
        try:
            import cv2
            return cv2
        except:
            return None
    
    def update_loop(self, dt):
        try:
            curr_time = time.time()
            frame_time = curr_time - self.prev_time
            if frame_time > 0:
                self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
            self.prev_time = curr_time
            
            self.frame_count += 1
            if self.frame_count % 3 == 0:
                self.fps_text = f'FPS: {int(self.fps)}'
                
                self.motion_x += (np.random.rand() - 0.5) * 0.1
                self.motion_y += (np.random.rand() - 0.5) * 0.1
                self.motion_amount = 0.3 + 0.3 * np.random.rand()
                
                self.motion_x = max(0.1, min(0.9, self.motion_x))
                self.motion_y = max(0.1, min(0.9, self.motion_y))
            
            if self.current_mode == MODE_SYNTH and self.synth:
                self.synth.update(self.motion_x, self.motion_y, self.motion_amount)
                self.synth_text = f'Freq: {int(self.synth.frequency)}Hz'
            
            self._update_display()
            
        except Exception as e:
            Logger.error(f"Update error: {e}")
    
    def on_stop(self):
        if self.synth:
            try:
                self.synth.stop()
            except:
                pass


if __name__ == '__main__':
    HandGestureApp().run()
