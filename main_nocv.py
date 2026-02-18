from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import StringProperty
from kivy.graphics.texture import Texture
import time

class MinimalCameraApp(App):
    fps_text = StringProperty('FPS: 0')
    status_text = StringProperty('Starting...')
    
    def build(self):
        Logger.info("Building app...")
        self.fps = 30
        self.prev_time = time.time()
        self.frame_count = 0
        self.camera = None
        self.camera_index = 0
        
        layout = BoxLayout(orientation='vertical')
        
        self.display = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.display, size_hint=(1, 0.85))
        
        btn_layout = GridLayout(cols=3, size_hint=(1, 0.15), spacing=5, padding=5)
        
        btn_camera = Button(text='Start Camera')
        btn_camera.bind(on_press=self.toggle_camera)
        btn_layout.add_widget(btn_camera)
        
        btn_switch = Button(text='Switch Camera')
        btn_switch.bind(on_press=self.switch_camera)
        btn_layout.add_widget(btn_switch)
        
        btn_reset = Button(text='Reset')
        btn_reset.bind(on_press=self.reset)
        btn_layout.add_widget(btn_reset)
        
        layout.add_widget(btn_layout)
        
        Clock.schedule_interval(self.update_loop, 1.0 / 30.0)
        
        Logger.info("App built successfully")
        return layout
    
    def toggle_camera(self, instance):
        if self.camera is None:
            self.start_camera()
            instance.text = 'Stop Camera'
        else:
            self.stop_camera()
            instance.text = 'Start Camera'
    
    def start_camera(self):
        try:
            from kivy.core.camera import Camera
            Logger.info(f"Trying to create camera with index {self.camera_index}")
            self.camera = Camera(index=self.camera_index, resolution=(640, 480), play=True)
            self.status_text = f'Camera {self.camera_index} started'
            Logger.info("Camera started successfully")
        except Exception as e:
            Logger.error(f"Camera error: {e}")
            self.status_text = f'Camera error: {e}'
            self.camera = None
    
    def stop_camera(self):
        if self.camera:
            try:
                self.camera.play = False
            except:
                pass
            self.camera = None
        self.status_text = 'Camera stopped'
    
    def switch_camera(self, instance):
        self.stop_camera()
        self.camera_index = 1 - self.camera_index
        self.start_camera()
    
    def reset(self, instance):
        self.stop_camera()
        self.camera_index = 0
        self.status_text = 'Reset'
    
    def update_loop(self, dt):
        try:
            curr_time = time.time()
            frame_time = curr_time - self.prev_time
            if frame_time > 0:
                self.fps = self.fps * 0.9 + (1.0 / frame_time) * 0.1
            self.prev_time = curr_time
            
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.fps_text = f'FPS: {int(self.fps)}'
            
            if self.camera is None:
                import numpy as np
                display = np.zeros((480, 640, 3), dtype=np.uint8)
                self._draw_text(display, self.status_text, (320, 240), 1.0, (255, 255, 255))
                self._update_texture(display)
            else:
                try:
                    texture = self.camera.texture
                    if texture:
                        self.display.texture = texture
                    else:
                        import numpy as np
                        display = np.zeros((480, 640, 3), dtype=np.uint8)
                        self._draw_text(display, 'No frame', (320, 240), 1.0, (255, 255, 255))
                        self._update_texture(display)
                except Exception as e:
                    Logger.error(f"Texture error: {e}")
                    
        except Exception as e:
            Logger.error(f"Update error: {e}")
    
    def _draw_text(self, img, text, center, scale, color):
        h, w = img.shape[:2]
        text_w = len(text) * 12
        text_h = 20
        x = int(center[0] - text_w / 2)
        y = int(center[1] + text_h / 2)
        for i, char in enumerate(text):
            for dy in range(-8, 8):
                for dx in range(-6, 6):
                    px, py = x + i * 12 + dx, y + dy
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] = color
    
    def _update_texture(self, img):
        from kivy.graphics.texture import Texture
        buf = img.tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.display.texture = texture
    
    def on_stop(self):
        Logger.info("App stopping")
        self.stop_camera()


if __name__ == '__main__':
    try:
        Logger.info("Starting MinimalCameraApp...")
        MinimalCameraApp().run()
    except Exception as e:
        Logger.error(f"App crash: {e}")
        import traceback
        Logger.error(traceback.format_exc())
