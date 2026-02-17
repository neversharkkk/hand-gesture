# Android APK 打包说明

## 环境要求

打包Android APK需要在Linux环境下进行（推荐Ubuntu 20.04或更高版本）。Windows用户建议使用WSL2或虚拟机。

## 方法一：使用Buildozer（推荐）

### 1. 安装依赖

```bash
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev automake python3-pip
```

### 2. 安装Buildozer

```bash
pip3 install buildozer cython==0.29.33
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

## 方法二：使用Docker（更简单）

### 1. 创建Docker容器

```bash
docker pull kivy/buildozer
```

### 2. 运行打包

```bash
docker run --rm -v "D:\Documents\V":/app kivy/buildozer android debug
```

## 方法三：使用GitHub Actions自动打包（推荐）

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
   git tag v1.0.0
   git push origin v1.0.0
   ```
   - 发布后可在 Releases 页面下载APK

### 工作流触发条件

- 推送到 `main` 或 `master` 分支
- 创建 `v*` 格式的tag
- 手动触发（workflow_dispatch）
- Pull Request

## 文件说明

| 文件 | 说明 |
|------|------|
| `main_android.py` | Kivy版本的Android应用主程序 |
| `android_camera.py` | Android摄像头适配模块 |
| `buildozer.spec` | Buildozer配置文件 |
| `requirements_android.txt` | Python依赖列表 |

## 注意事项

1. **原版main.py使用pyautogui进行鼠标控制，这在Android上不可用，已移除该功能**

2. **手势识别功能保留**：
   - 手势模式：检测并显示手势名称
   - ASCII艺术模式：将手部转换为ASCII艺术
   - 调色板模式：提取并应用主色调

3. **权限配置**：已在buildozer.spec中配置摄像头和存储权限

4. **架构支持**：支持arm64-v8a和armeabi-v7a架构

## 调试模式安装

```bash
# 通过ADB安装到设备
adb install bin/handgesture-1.0.0-arm64-v8a-debug.apk
```

## 发布版本打包

```bash
buildozer android release
```

发布版本需要签名，请参考Android官方文档。
