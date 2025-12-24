# Backend 测试脚本使用说明

本目录包含两个测试脚本，用于验证RK3568开发板硬件功能。

---

## 脚本1：check_led.sh - LED检查工具

### 功能
自动检测板载LED并提供交互式测试功能。

### 使用方法

**1. 上传到开发板**：
```bash
scp backend/scripts/check_led.sh root@<开发板IP>:/tmp/
```

**2. 运行脚本**：
```bash
ssh root@<开发板IP>
chmod +x /tmp/check_led.sh
/tmp/check_led.sh
```

### 输出说明

**[1] 可用的 LED 列表**
显示所有 `/sys/class/leds/` 下的LED设备。

**[2] LED 详细信息**
每个LED的：
- 当前亮度（0=关闭，>0=打开）
- 最大亮度（通常为1或255）
- 触发模式（none/timer/heartbeat等）
- 完整路径

**[3] 推荐配置**
优先级：`sys_led` > `user` > `work`，显示推荐使用的LED路径。

**[4] 交互式测试**
输入`y`进入测试模式：
- 输入LED名称（如`sys_led`）
- 自动执行：关闭 → 打开 → 闪烁5次
- 恢复原始状态

### 使用示例

```bash
$ ./check_led.sh

[1] 可用的 LED 列表：
sys_led  user  work

[2] LED 详细信息：
--- sys_led ---
  当前亮度: 0
  最大亮度: 1
  触发模式: [none]
  路径: /sys/class/leds/sys_led/

[3] 推荐配置：
✓ 推荐使用: sys_led
  配置路径: /sys/class/leds/sys_led/brightness

[4] 是否测试 LED 控制？(y/n)
y
请输入要测试的 LED 名称（如 sys_led）：
sys_led
开始测试 sys_led ...
  原始亮度: 0
  [1/3] 关闭 LED...
  [2/3] 打开 LED...
  [3/3] 闪烁测试（5次）...
  恢复原始状态: 0
✓ 测试完成！
```

### 配置到代码

测试成功后，将推荐的LED路径配置到`backend/config.py`：
```python
LED_PATH = "/sys/class/leds/sys_led/brightness"
```

---

## 脚本2：test_camera.py - 摄像头测试工具

### 功能
全面测试OV5695摄像头的连接、模式和性能。

### 使用方法

**1. 上传到开发板**：
```bash
scp -r backend/scripts root@<开发板IP>:/tmp/
```

**2. 基础测试（推荐）**：
```bash
ssh root@<开发板IP>
cd /tmp/scripts
python3 test_camera.py
```

**3. 完整测试**：
```bash
python3 test_camera.py --width 640 --height 480
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--index` | 摄像头索引 | 自动检测 |
| `--width` | 图像宽度 | 640 |
| `--height` | 图像高度 | 480 |
| `--output` | 输出目录 | camera_test_output |
| `--skip-gstreamer` | 跳过GStreamer测试 | False |
| `--skip-performance` | 跳过性能测试 | False |

### 测试流程

**步骤1：检查Video设备**
扫描`/dev/video*`，显示所有可用设备。

**步骤2：OpenCV模式测试**
- 使用标准V4L2接口
- 设置分辨率并读取帧
- 验证图像质量（亮度检查）
- 保存测试图片：`opencv_test_<时间戳>_<分辨率>.jpg`

**步骤3：GStreamer模式测试**（仅Linux）
- 使用RK3568硬件加速管道
- 测试rkisp插件
- 保存测试图片：`gstreamer_test_<时间戳>_<分辨率>.jpg`

**步骤4：性能测试**
- 连续捕获5秒
- 计算实际FPS
- 保存最后一帧：`continuous_test_<时间戳>_<分辨率>.jpg`

### 输出解读

**成功示例**：
```
============================================================
  1. 检查 Video 设备
============================================================
找到 3 个 video 设备：
  /dev/video0: Card type: rkisp
  /dev/video1: Card type: ov5695
[✓ PASS] Video 设备检测
      找到 3 个设备

============================================================
  2. 测试 OpenCV 模式 (索引 0)
============================================================
正在打开摄像头 (索引 0)...
[✓ PASS] OpenCV 打开摄像头
请求分辨率: 640x480
实际分辨率: 640x480
帧率: 30.0 FPS
[✓ PASS] 捕获图像
      图像尺寸: 640x480
图像平均亮度: 128.45
[✓ PASS] 图像质量检查
      亮度正常
[✓ PASS] 保存图像
      文件: opencv_test_20251224_143025_640x480.jpg

============================================================
  测试总结
============================================================
Video 设备检测:   ✓
OpenCV 模式:      ✓
GStreamer 模式:   ✓
性能测试:         ✓

保存的测试图像 (3 张):
  continuous_test_20251224_143032_640x480.jpg (45.2 KB)
  gstreamer_test_20251224_143028_640x480.jpg (42.8 KB)
  opencv_test_20251224_143025_640x480.jpg (43.5 KB)

============================================================
  建议
============================================================
✓ GStreamer 模式工作正常，性能更好（推荐）
  配置: CAMERA_MODE = 'gstreamer'
```

**失败处理**：

如果OpenCV失败：
```
[✗ FAIL] OpenCV 打开摄像头
      无法打开摄像头
```
→ 检查排线连接，运行`ls /dev/video*`查看设备

如果GStreamer失败：
```
[✗ FAIL] GStreamer 打开摄像头
      无法打开（可能系统不支持 GStreamer 或 rkisp）
```
→ 正常现象，使用OpenCV模式即可

如果图像过暗/过亮：
```
[✗ FAIL] 图像质量检查
      图像过暗，可能是摄像头故障
```
→ 检查摄像头镜头，调整光照环境

### 下载测试图片

使用WinSCP或scp下载：
```bash
scp -r root@<开发板IP>:/tmp/scripts/camera_test_output ./
```

打开图片检查：
- ✓ 清晰、色彩正常 → 摄像头工作正常
- ✗ 偏绿 → 未使用ISP，尝试GStreamer模式
- ✗ 模糊/黑色 → 检查排线和驱动

### 配置到代码

根据测试结果修改`backend/config.py`：

**GStreamer测试成功（推荐）**：
```python
CAMERA_MODE = 'auto'  # 或 'gstreamer'
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

**仅OpenCV成功**：
```python
CAMERA_MODE = 'opencv'
CAMERA_INDEX = 1  # 根据测试结果调整
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

---

## 常见问题

### Q1: check_led.sh 提示 Permission denied？
```bash
chmod +x check_led.sh  # 添加执行权限
sudo ./check_led.sh    # 或使用sudo运行
```

### Q2: test_camera.py 找不到cv2模块？
```bash
# 安装OpenCV
sudo apt-get install python3-opencv
```

### Q3: 测试图片无法打开？
确认文件格式为.jpg，使用图片查看器或：
```bash
file opencv_test_*.jpg  # 查看文件类型
```

### Q4: 摄像头测试全部失败？
```bash
# 1. 检查设备节点
ls -l /dev/video*

# 2. 检查驱动
dmesg | grep ov5695

# 3. 重新加载驱动
sudo modprobe ov5695
```

### Q5: 想在PC上测试？
```bash
# 跳过GStreamer，只测OpenCV
python test_camera.py --skip-gstreamer --index 0
```

---

## 总结

**测试流程**：
1. 运行`check_led.sh` → 找到可用LED → 更新配置
2. 运行`test_camera.py` → 查看测试图片 → 选择模式 → 更新配置
3. 启动后端服务验证功能

**配置文件**：`backend/config.py`
**详细配置指南**：`docs/backend/RK3568开发板配置指南.md`
