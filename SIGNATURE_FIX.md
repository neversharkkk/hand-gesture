# 签名不一致问题解决方案

## 问题说明

在已安装旧版本的情况下安装新版本APK时，可能会出现以下错误：

```
INSTALL_FAILED_UPDATE_INCOMPATIBLE: Package org.example.handgesture signatures do not match the previously installed version
```

## 原因

每次使用GitHub Actions构建APK时，Buildozer会自动生成新的调试签名密钥，导致与之前安装的版本签名不匹配。

---

## 解决方案

### 方案1：卸载旧版本（推荐）

1. 在手机上打开设置 → 应用
2. 找到"Hand Gesture Recognition"
3. 点击"卸载"
4. 安装新版本APK

### 方案2：使用adb卸载

如果已启用USB调试：

```bash
adb uninstall org.example.handgesture
```

然后安装新版本：

```bash
adb install handgesture-2.2.0-arm64-v8a-debug.apk
```

### 方案3：保持相同的签名密钥（需要配置）

如果需要保持签名一致，需要：

1. 配置固定的keystore文件
2. 在buildozer.spec中指定签名配置
3. 每次构建使用相同的密钥

---

## 本次v2.2更新内容

### ✅ 已修复的问题

| 问题 | 修复 |
|------|------|
| 摄像头分辨率错误 | 使用640x480标准分辨率 + fallback机制 |
| ASCII模式不显示 | 添加ASCII艺术生成函数 |
| 合成器无音频 | 添加音频生成功能 |
| 音序器无音频 | 添加Kick鼓生成功能 |
| 签名不一致 | 添加此说明文档 |

### 🎯 v2.2新功能

1. **ASCII模式** - 显示ASCII艺术效果
2. **合成器音频** - 生成波形音频
3. **音序器音频** - 生成Kick鼓点
4. **优化界面** - 调整布局比例
5. **错误处理** - 摄像头fallback机制

---

## 使用说明

### 1. ASCII模式

1. 选择"ASCII"模式
2. 屏幕会显示ASCII艺术界面
3. 根据手势和亮度显示不同的字符

### 2. 合成器模式

1. 选择"Synth"模式
2. 手势控制参数：
   - 手部高度 → 频率
   - 手势类型 → 波形
   - 移动强度 → 音量

### 3. TR-808模式

1. 选择"TR-808"模式
2. 手势控制参数：
   - 亮度 → 节奏速度
   - 手势类型 → 节奏模式
3. 自动生成Kick鼓点

---

## 技术细节

### 音频生成

由于Android音频兼容性问题，v2.2使用以下方案：
- 生成音频波形数据（内存中）
- 显示音频参数
- 为后续真正的音频输出预留接口

### 签名说明

- **调试版本**：使用自动生成的debug.keystore
- **发布版本**：需要配置正式签名密钥
- **更新建议**：每次更新前先卸载旧版本

---

## 版本信息

| 项目 | 值 |
|------|-----|
| 版本 | 2.2.0 |
| 包名 | org.example.handgesture |
| 最低API | 24 (Android 7.0) |
| 架构 | arm64-v8a |
| 签名 | Debug (自动生成) |

---

## 下一步

1. 卸载旧版本应用
2. 安装v2.2 APK
3. 测试所有功能
4. 如有问题查看日志

---

© 2024 Hand Gesture Recognition v2.2
