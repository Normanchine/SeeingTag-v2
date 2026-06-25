# SeeingTag-v2

面向智能车的 ArUco 视觉定位方案。SeeingTag-v2 使用四个固定 Tag 建立场地平面坐标系，实时追踪车载 Tag，并通过 UDP 将位置、朝向和追踪状态发送给 Unity。

## 定位流程

```text
4 个固定 Tag 同时可见
        ↓
固定 Tag 图像中心  →  场地中心坐标
        ↓
计算并缓存 Homography（像素 → 场地 X/Z）
        ↓
原始画面识别车载 Tag ──失败──→ 高分辨率鸟瞰图兜底识别
        ↓                                ↓
        └────────── 平滑、短暂丢失保持 ───┘
                         ↓
                    UDP → Unity
```

## v2 核心改进

- **固定 Tag 中心点建模**：只使用四个固定 Tag 的中心点计算 H；固定码可按现场识别效果横放、竖放或旋转，不需要统一朝向。
- **原图优先 + 鸟瞰图兜底**：原图丢失车载 Tag 时，自动在 `1600 × 1280` 鸟瞰图中重试。
- **H 缓存保护**：在 `interval` / `every_frame` 更新失败时保留上一份有效 H，不因固定码短暂离开画面而中断定位。
- **稳定输出**：视觉层与 UDP 输出层分别平滑；偶发丢帧时保留最近可信位置 `0.35` 秒。
- **可观测状态**：UDP 附带 `tracking_state`（`raw`、`bird_eye_fallback`、`hold`），便于 Unity 做状态提示。

## 快速开始

```powershell
git clone https://github.com/Normanchine/SeeingTag-v2.git
cd SeeingTag-v2

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 比赛模式：定位并发送 UDP
python main.py

# 校准模式：仅显示和调试，不发送 UDP
python main.py --calibrate
```

## 场地与标签

默认场地为 `5m × 4m`。固定 Tag（ID 1~4）配置在 `tag_config.json` 的 `tag_world_positions` 中；车载 Tag 默认 ID 为 `10`。

初始化与重新校准时，四个固定 Tag 都必须在画面内，并且平贴在同一场地平面。它们的摆放朝向可以不同，但配置的 `x/z` 必须是各 Tag 的实际中心位置。

## 主要配置

| 配置项 | 说明 |
|---|---|
| `field_width_m` / `field_height_m` | 场地宽高（米） |
| `unity_ip` / `unity_port` | Unity UDP 接收地址 |
| `homography_update_mode` | `once`、`interval` 或 `every_frame` |
| `filter_alpha` | 鸟瞰图和 HUD 的位置/朝向平滑系数 |
| `output_filter_alpha` | UDP 输出平滑系数 |
| `flip_x` / `flip_z` | 镜像坐标与朝向，用于校正 Unity 坐标方向 |
| `car_heading_offset_degrees` | 车头相对车载 Tag 默认方向的角度偏移（度）；会影响鸟瞰箭头和 Unity yaw |
| `debug_logging` | 输出高频调试日志，比赛时建议关闭 |

## 调整车头方向

车载 Tag 的默认方向由其印刷朝向决定；当 Tag 横放或以任意角度安装时，可以为真实车头设置一个固定偏移角。此偏移会同时作用于鸟瞰图箭头、HUD 和发送给 Unity 的 yaw，并永久写入 `tag_config.json`，下次启动自动生效。

在项目目录执行（角度单位为度）：

```powershell
# 例如：让车头相对当前 Tag 默认方向逆时针旋转 90°
python main.py --set-car-heading 90

# 反方向旋转 90°
python main.py --set-car-heading -90

# 恢复为 Tag 默认方向
python main.py --set-car-heading 0

# 查看当前已保存的偏移值
python main.py --show-car-heading
```

建议先以 `0°` 启动并观察鸟瞰箭头，再以 `±90°`、`180°` 逐步试验，直到箭头与真实车头一致。正值按系统世界坐标的 yaw 正方向叠加；角度会自动归一化到 `[-180°, 180°)`。

## 项目结构

```text
.
├── main.py                    # 视觉定位、鸟瞰图、UDP 主程序
├── tag_config.json            # 标签布局、场地、网络和滤波配置
├── generate_tag.py            # 生成固定/车载 ArUco Tag
├── calibrate_camera.py        # 棋盘格相机标定工具（可选）
├── test_*.py                  # 摄像头、IPM、UDP 调试工具
├── requirements.txt
├── 本次定位优化说明.md         # 完整技术与调参说明
└── 技术报告_*.md
```

## 文档

详细的定位流程、鸟瞰图兜底机制、UDP 数据状态和现场调参建议见 [本次定位优化说明.md](本次定位优化说明.md)。

## 依赖

- Python 3.10+
- `opencv-contrib-python >= 4.8.0`
- `numpy >= 1.24.0`
