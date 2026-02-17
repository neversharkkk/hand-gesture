# Android APK 打包说明

## 环境要求

打包Android APK需要在Linux环境下进行（推荐Ubuntu 22.04或更高版本）。Windows用户建议使用WSL2或虚拟机。

## 兼容性

- **最低Android版本**：Android 7.0 (API 24)
- **目标Android版本**：Android 14 (API 34)
- **支持架构**：arm64-v8a（64位设备）

## 方法一：使用GitHub Actions自动打包（推荐）

已创建 `.github/workflows/build.yml` 工作流文件，推送到GitHub后会自动构建APK。

### 使用步骤

1. **创建GitHub仓库**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/你的用户名/hand-gesture.git
   git push -u origin main
   ```

2. **查看构建进度**
   - 进入GitHub仓库页面
   - 点击 "Actions" 标签
   - 查看构建状态

3. **下载APK**
   - 构建完成后，在 Actions 页面点击对应的工作流
   - 在 "Artifacts" 区域下载 `hand-gesture-apk`

4. **发布版本**
   - 创建tag会自动发布Release：
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```
   - 发布后可在 Releases 页面下载APK

### 工作流触发条件

- 推送到 `main` 或 `master` 分支
- 创建 `v*` 格式的tag
- 手动触发（workflow_dispatch）
- Pull Request

## 方法二：使用Buildozer本地打包

### 1. 安装依赖

```bash
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo-dev cmake libffi-dev libssl-dev automake python3-pip
```

### 2. 安装Buildozer

```bash
pip3 install buildozer cython==0.29.36
```

### 3. 初始化并打包

将以下文件复制到Linux环境的项目目录：
- `main_android.py` 重命名为 `main.py`
- `android_camera.py`
- `buildozer.spec`
- `requirements_android.txt`

然后执行：

```bash
buildozer init  # 如果没有buildozer.spec
buildozer android debug
```

首次打包会自动下载Android SDK、NDK等，可能需要较长时间。

### 4. 输出位置

打包完成后，APK文件位于 `bin/` 目录下。

## 方法三：使用Docker

```bash
docker pull kivy/buildozer
docker run --rm -v "D:\Documents\V":/app kivy/buildozer android debug
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `main_android.py` | Kivy版本的Android应用主程序 |
| `android_camera.py` | Android摄像头适配模块 |
| `buildozer.spec` | Buildozer配置文件 |
| `requirements_android.txt` | Python依赖列表 |

## 权限配置

应用需要以下权限：
- `CAMERA` - 摄像头访问
- `WRITE_EXTERNAL_STORAGE` - 存储写入（Android 10及以下）
- `READ_EXTERNAL_STORAGE` - 存储读取（Android 10及以下）
- `READ_MEDIA_IMAGES` - 读取图片（Android 13+）
- `READ_MEDIA_VIDEO` - 读取视频（Android 13+）
- `READ_MEDIA_AUDIO` - 读取音频（Android 13+，合成器模式）

## Android 14+ 兼容性

针对Android 14及以上版本的兼容性处理：

1. **权限模型更新**：自动适配Android 13+的新媒体权限
2. **摄像头API**：使用`cv2.CAP_ANDROID`后端优先
3. **缓冲区优化**：设置`BUFFERSIZE=1`减少延迟
4. **音频输出**：合成器模式使用AudioTrack API实时音频输出

## 调试模式安装

```bash
# 通过ADB安装到设备
adb install bin/handgesture-1.1.0-arm64-v8a-debug.apk
```

## 发布版本打包

```bash
buildozer android release
```

发布版本需要签名，请参考Android官方文档。

## 常见问题

### 应用闪退
1. 确保已授予摄像头权限
2. 检查设备是否为arm64架构
3. 查看logcat日志：`adb logcat | grep python`

### 摄像头无法打开
1. 确认摄像头权限已授予
2. 尝试重启应用
3. 检查是否有其他应用占用摄像头

### 画面卡顿
1. 降低分辨率（修改`width`和`height`参数）
2. 关闭其他后台应用
