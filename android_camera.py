import cv2
import numpy as np
from kivy.logger import Logger

try:
    from android.permissions import request_permissions, Permission, check_permission
    from jnius import autoclass
    ANDROID_AVAILABLE = True
    PythonActivity = autoclass('org.kivy.android.PythonActivity')
except ImportError:
    ANDROID_AVAILABLE = False
    Logger.warning("AndroidCamera: Android modules not available, running in desktop mode")


class AndroidCamera:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.permission_granted = False
        
        if ANDROID_AVAILABLE:
            self._request_android_permissions()
        else:
            self.permission_granted = True
    
    def _request_android_permissions(self):
        if not ANDROID_AVAILABLE:
            return
        
        try:
            permissions = [
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE,
            ]
            
            try:
                permissions.append(Permission.READ_MEDIA_IMAGES)
                permissions.append(Permission.READ_MEDIA_VIDEO)
            except AttributeError:
                pass
            
            request_permissions(permissions)
            
            self.permission_granted = True
            Logger.info("AndroidCamera: Permissions requested")
        except Exception as e:
            Logger.error(f"AndroidCamera: Permission request failed: {e}")
            self.permission_granted = True
    
    def open(self):
        if not self.permission_granted and ANDROID_AVAILABLE:
            Logger.warning("AndroidCamera: Camera permission not granted, trying anyway")
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANDROID)
            
            if not self.cap.isOpened():
                for i in range(4):
                    self.cap = cv2.VideoCapture(i, cv2.CAP_ANDROID)
                    if self.cap.isOpened():
                        Logger.info(f"AndroidCamera: Opened camera {i}")
                        break
            
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
                for i in range(4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        Logger.info(f"AndroidCamera: Opened camera {i} (fallback)")
                        break
            
            if self.cap is None or not self.cap.isOpened():
                Logger.error("AndroidCamera: Could not open any camera")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            Logger.info(f"AndroidCamera: Resolution {actual_w}x{actual_h}")
            
            return True
        except Exception as e:
            Logger.error(f"AndroidCamera: Error opening camera: {e}")
            return False
    
    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            Logger.info("AndroidCamera: Camera released")
    
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()
    
    def get(self, prop_id):
        if self.cap is not None:
            return self.cap.get(prop_id)
        return 0
    
    def set(self, prop_id, value):
        if self.cap is not None:
            return self.cap.set(prop_id, value)
        return False


def get_camera_instance(camera_index=0, width=640, height=480):
    return AndroidCamera(camera_index, width, height)
