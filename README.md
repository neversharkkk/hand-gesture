# V - 手势识别应用

## 概述

V 是一个基于 Kivy 的手势识别应用，通过摄像头实时识别手势（张开手、握拳、指向、比耶、比 V 等），支持桌面和 Android 平台。

## 功能特性

### 模式列表

| 模式 | 功能 |
|------|------|
| Camera | 摄像头预览 |
| Gesture | 手势识别（肤色+轮廓），显示手势与手指数量 |
| ASCII | 手部区域 ASCII 艺术风格叠加 |
| Synth | 手势控制合成器（Android 端有实际音频） |
| TR-808 | 手势控制 TR-808 鼓机（Android 端有实际音频） |

**说明**：桌面端使用 Kivy Camera；Android 端使用 OpenCV 摄像头，具备完整手势识别与音频功能。

### 手势识别

- **SimpleGestureRecognizer**：基于亮度和运动检测的轻量级手势分类
- 支持手势：Open Hand、Fist、Pointing、Victory、Peace
- 手指数量映射：Fist=0, Pointing=1, Victory/Peace=2, Open Hand=5
- 校准机制：约 30 帧后完成校准

### 界面控制

| 按钮 | 功能 |
|------|------|
| ▶ Start / ■ Stop | 启动/停止摄像头 |
| ⟳ Switch | 切换前后摄像头 |
| ↺ Reset | 重置状态 |

## 运行方式

```bash
python main.py
```

或使用入口脚本：

```bash
python main_entry.py
```

### 依赖

```bash
pip install -r requirements.txt
```

## 依赖

- **桌面**：Python 3.x, Kivy
- **Android 构建**：buildozer 自动包含 kivy, opencv, numpy, android

## 构建 Android APK

使用 Buildozer 构建：

```bash
buildozer android debug
```

构建产物位于 `bin/` 目录。详见 [BUILD_ANDROID.md](BUILD_ANDROID.md)。

## 版本

当前版本：2.6.0
