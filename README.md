# Hand Gesture Recognition

实时手势识别应用，支持PC端和Android平台。

## 功能特性

### 手势识别模式
- 实时检测并识别多种手势：Fist（握拳）、Paper（手掌）、Victory（剪刀）、One（一）、Three（三）、Point（指向）
- 显示手部轮廓和手势名称
- 支持手势平滑处理，减少抖动

### ASCII艺术模式
- 将手部区域实时转换为ASCII艺术效果
- 复古绿色渐变风格
- 根据亮度动态选择字符和颜色

### 色彩调色板模式
- 提取画面中8种主色调，创建艺术化效果
- 实时色彩映射和饱和度增强
- 平滑过渡处理

### 音乐合成器模式（仅PC）
- 手势控制音乐参数：频率、LFO、滤波器
- 颜色影响音色：色相、饱和度、明度
- ASCII艺术随音乐动态变化
- 支持多种LFO波形（正弦、方波、三角波）

### 鼠标控制模式（仅PC）
- 通过手部位置控制鼠标移动
- 支持pyautogui进行光标控制

## 快速开始

### PC端运行

```bash
pip install -r requirements.txt
python main.py
```

### Android APK构建

本项目支持通过GitHub Actions自动构建Android APK：

1. Fork或推送代码到GitHub仓库
2. 进入Actions页面查看构建进度
3. 构建完成后下载APK文件

详细构建说明请参考 [BUILD_ANDROID.md](BUILD_ANDROID.md)

## 操作说明

### PC端快捷键

| 按键 | 功能 |
|------|------|
| `1` | 鼠标控制模式 |
| `2` | 手势识别模式 |
| `3` | ASCII艺术模式 |
| `4` | 8色调色板模式 |
| `5` | 音乐合成器模式 |
| `r` | 重置历史记录 |
| `q` | 退出程序 |

### Android端

点击底部按钮切换模式：
- **手势** - 手势识别
- **ASCII** - ASCII艺术
- **调色板** - 色彩调色板
- **合成器** - 视觉合成器（无音频）
- **重置** - 重置历史记录

## 技术栈

- **OpenCV** - 图像处理、轮廓检测、色彩空间转换
- **NumPy** - 数值计算、K-means聚类
- **Kivy** - Android跨平台UI框架
- **Buildozer** - Android打包工具
- **PyAutoGUI** - PC端鼠标控制
- **SoundDevice** - PC端音频输出

## 项目结构

```
├── main.py              # PC端主程序
├── main_android.py      # Android端主程序
├── android_camera.py    # Android摄像头适配模块
├── buildozer.spec       # Buildozer打包配置
├── requirements.txt     # PC端依赖
├── requirements_android.txt  # Android端依赖
├── .github/
│   └── workflows/
│       └── build.yml    # GitHub Actions工作流
└── BUILD_ANDROID.md     # Android构建说明
```

## 系统要求

### PC端
- Python 3.7+
- 摄像头
- Windows / macOS / Linux
- 音频输出设备（合成器模式）

### Android端
- Android 5.0 (API 21) 或更高版本
- 摄像头权限

## 手势识别原理

1. **皮肤检测**：使用HSV、YCrCb、LAB三个颜色空间融合检测皮肤区域
2. **背景减除**：使用MOG2背景减除器分离前景
3. **轮廓筛选**：基于面积、圆形度、实度、长宽比评分筛选手部轮廓
4. **凸包分析**：计算凸包和凸缺陷，识别手指数量
5. **手势判断**：根据实度和手指数量判断具体手势

## 合成器模式原理

手势和颜色实时控制音乐参数：

| 参数 | 控制方式 |
|------|----------|
| 频率 | 手的Y位置（高=高频，低=低频） |
| LFO速率 | 手的X位置 |
| LFO深度 | 手的面积 |
| 失谐 | 手指数量 |
| LFO波形 | 手指数量 % 3 |
| 频率微调 | 主色调色相 |
| 泛音丰富度 | 主色调饱和度 |
| 滤波器 | 主色调明度 |

## 许可证

MIT License
