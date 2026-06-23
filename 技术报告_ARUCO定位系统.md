# 基于ARUCO标记的智能车定位系统技术报告

## 1. 项目背景

### 1.1 赛事介绍
本项目面向第21届全国智能汽车大赛人工智能模型组别，设计并实现了一套基于ARUCO视觉标记的智能车实时定位系统。系统通过摄像头采集赛道信息，识别ARUCO标记码，计算智能车在虚拟AR环境中的位置，实现虚实融合的定位效果。

### 1.2 系统目标
- 实时识别赛道上的ARUCO标记码
- 通过多标记位姿估计计算摄像头/智能车位置
- 通过UDP协议将位置数据发送至上位机
- 实现虚拟AR摄像头与真实智能车的位置同步

## 2. 系统架构

### 2.1 硬件配置
| 设备 | 参数 |
|------|------|
| 摄像头 | 分辨率1280×720，FOV 100.196° |
| 安装高度 | 3米（俯视赛道） |
| 上位机 | RK3588香橙派 |
| 通信网络 | UDP局域网，IP 192.168.2.124:9005 |

### 2.2 软件架构
```
摄像头 → OpenCV ARUCO检测 → PnP位姿估计 → UDP发送 → RK3588接收 → AR融合显示
```

### 2.3 坐标系定义
```
        Z (远离摄像头方向)
         ↑
         │    X (横向)
         │   ↗
         │  /
         │ /
  ──────●────────→ X
  (0,0) │
        ↓ Y (高度，地面时Y=0)
```

赛道尺寸：4m × 5m (X: 0-4, Z: 0-5)

## 3. 技术实现

### 3.1 ARUCO标记设计
采用DICT_4X4_50字典，标记尺寸166mm×166mm。

**标记布局：**
| 标记ID | 位置 | 用途 |
|--------|------|------|
| 1 | (0, 0, 0) | 固定参考点-左下角 |
| 2 | (4, 0, 0) | 固定参考点-右下角 |
| 3 | (4, 0, 5) | 固定参考点-右上角 |
| 4 | (0, 0, 5) | 固定参考点-左上角 |
| 10 | 车上 | 智能车位置标记 |

### 3.2 位姿估计算法
使用Perspective-n-Point (PnP)算法结合OpenCV的SOLVEPNP_SOLVEP6方法。

**多标记定位流程：**
1. 检测画面中所有可见的ARUCO标记
2. 筛选已知世界坐标的固定标记（ID 1-4）
3. 构建3D-2D对应点对（标记角点）
4. 调用cv2.solvePnP估计摄像头位姿
5. 根据摄像头位姿计算智能车世界坐标

**数学模型：**
```
[s · u]   [K|0] [R|t] [X]
[ s v ] =           [Y]
[ 1  ]             [Z]
                     [1]

其中：
- K: 相机内参矩阵
- R, t: 相机外参（旋转矩阵和平移向量）
- [X,Y,Z]: 标记世界坐标
- [u,v]: 图像像素坐标
```

### 3.3 相机内参
使用默认FOV参数：
- fx = width / (2 × tan(FOV_H / 2))
- fy = height / (2 × tan(FOV_V / 2))
- cx = width / 2
- cy = height / 2

### 3.4 数据通信协议
采用JSON格式UDP数据包：

```json
{
  "type": "robot_position",
  "pos": [x, y, z],
  "euler": [rx, ry, rz],
  "seq": 序列号,
  "timestamp": 时间戳
}
```

**通信参数：**
- 目标IP: 192.168.2.124
- 端口: 9005
- 协议: UDP
- 发送频率: ~30Hz

## 4. 核心代码模块

### 4.1 标记检测器
```python
class SimpleTagTracker:
    def detect_tags(self, gray):
        corners, ids, rejected = self.detector.detectMarkers(gray)
        # 返回检测到的标记列表
```

### 4.2 单标记位姿估计
```python
def estimate_single_tag_pose(self, corners):
    objPoints = np.array([...])  # 标记4角3D坐标
    success, rvec, tvec = cv2.solvePnP(objPoints, imagePoints, K, D)
    return rvec, tvec
```

### 4.3 多标记联合定位
```python
def estimate_camera_pose(self, detections):
    # 收集所有固定标记的3D-2D对应点
    # 调用solvePnP获得相机位姿
    return rvec, tvec
```

### 4.4 UDP发送模块
```python
def send_to_unity(self, position):
    data = {
        "type": "robot_position",
        "pos": [position[0], position[1], position[2]],
        "euler": [0, 0, 0],
        "seq": self.seq,
        "timestamp": time.time()
    }
    self.udp_client.sendto(json.dumps(data).encode(), target_addr)
```

## 5. 实验结果

### 5.1 通信测试
- 使用mcu_simulator.py进行UDP通信测试
- 成功实现X: -2~+2, Y: -1~+1, Z: 1的连续变化数据发送
- AR融合窗口正确响应位置变化

### 5.2 标记识别
- 在标准测试环境下成功识别ID 1-4和ID 10标记
- 简化模式下可根据单标记位置进行坐标映射

### 5.3 存在的问题与改进
1. **摄像头参数匹配**：屏幕测试与实际赛道场景存在尺度差异，需现场标定
2. **遮挡处理**：多标记联合定位需至少2个固定标记可见
3. **实时性优化**：当前帧率受限于摄像头和PnP算法计算量

## 6. 结论与展望

### 6.1 完成工作
- 设计并实现了基于ARUCO标记的智能车定位系统
- 完成了UDP通信模块的开发与测试
- 验证了虚实位置同步的可行性

### 6.2 下一步计划
1. 完成赛道现场标定，确定精确的相机内参
2. 实现多标记融合算法，提高定位精度
3. 优化实时性能，满足30Hz以上的更新率要求
4. 与上位机控制系统集成，实现闭环定位控制

## 7. 参考文献

1. OpenCV Documentation: ArUco Marker Detection
2. OpenCV Documentation: Camera Calibration and 3D Reconstruction
3. F. Romero-Ramirez, et al. "Speeded up detection of squared fiducial markers", Image and Vision Computing, 2018

---

**项目时间**: 2026年4月

**开发环境**:
- Python 3.x
- OpenCV 4.x with opencv-contrib
- NumPy

**文件结构**:
```
SeeingTag/
├── test_tag_tracker.py    # 主程序
├── tag_locator.py         # 定位核心模块
├── test_udp_sender.py     # UDP测试工具
├── test_udp_listener.py   # UDP监听工具
└── tag_config.json        # 配置文件
```
