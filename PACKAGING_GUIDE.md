# Android APK 本地打包指南

## 前置准备

由于Windows下Buildozer打包有一定难度，推荐以下方式：

## 方式一：GitHub Actions 自动打包（推荐）

这是最简单的方法，无需本地环境配置。

### 步骤：

1. **创建GitHub仓库**
   ```bash
   cd d:\Documents\V
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

## 方式二：使用WSL2 + Buildozer

如果您有Windows Subsystem for Linux (WSL2)，可以按以下步骤：

### 1. 启动WSL2
```bash
wsl
```

### 2. 安装依赖
```bash
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo-dev cmake libffi-dev libssl-dev automake python3-pip
```

### 3. 安装Buildozer
```bash
pip3 install buildozer cython==0.29.36
```

### 4. 准备项目文件
```bash
cd /mnt/d/Documents/V
cp main_buildozer.py main.py
```

### 5. 开始构建
```bash
buildozer android debug
```

## 方式三：使用虚拟机

安装Ubuntu 22.04虚拟机，然后按照方式二的步骤操作。

## 当前文件说明

| 文件 | 用途 |
|------|------|
| `main.py` | PC端主程序 |
| `main_android.py` | Android版本源码 |
| `main_buildozer.py` | Buildozer打包用主文件 |
| `buildozer.spec` | Buildozer配置文件 |
| `requirements_android.txt` | Android依赖列表 |

## 打包前检查清单

- [ ] 已将 `main_buildozer.py` 重命名为 `main.py`（如果需要）
- [ ] buildozer.spec中的requirements包含所有需要的库
- [ ] android.permissions包含所需权限
- [ ] 有足够的磁盘空间（至少10GB）
- [ ] 稳定的网络连接（首次打包需要下载SDK/NDK）

## 常见问题

### Q: 首次打包时间太长？
A: 首次打包需要下载Android SDK、NDK等组件，通常需要30-60分钟，后续打包会快很多。

### Q: 构建失败？
A: 
1. 检查磁盘空间是否足够
2. 检查网络连接
3. 查看 `.buildozer/android/platform/build-*/logs/` 目录下的日志

### Q: 如何安装到手机？
A: 
```bash
adb install bin/handgesture-1.1.0-arm64-v8a-debug.apk
```
或者直接将APK文件复制到手机安装。
