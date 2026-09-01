# SeeingTag-v2

面向智能车的 ArUco 视觉定位程序：用场地四角固定 Tag 建立平面坐标系，追踪车载 Tag，并通过 UDP 输出位置与朝向给 Unity；同时提供鸟瞰视图、轨迹显示和 K/P/D 调参窗口。

## 特性

- 固定 Tag 按中心点建模，允许不同安装朝向；
- 车载 Tag 尺寸筛选使用四边像素长度中位数，降低旋转掉点；
- Homography 支持 `once`、`interval`、`every_frame`，失败时保留有效矩阵；
- 原图优先、鸟瞰图兜底，位置/yaw 平滑并输出 `raw`、`bird_eye_fallback`、`hold` 状态；
- 盲开保护受时间、速度和最大距离限制；
- `Track Map - 4x5` 显示固定 Tag、车辆轨迹并发送一次性 K/P/D 保存包；
- 支持可缩放原图窗口、校准模式和 headless 运行。

## 安装与运行

```powershell
git clone https://github.com/Normanchine/SeeingTag-v2.git
cd SeeingTag-v2
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python main.py --calibrate
```

依赖 Python 3.10+、`opencv-contrib-python >= 4.8.0`、`numpy >= 1.24.0`。默认场地为 `5m × 4m`，固定 Tag ID 1~4，车载 Tag ID 10。

## 配置与调参

在 [tag_config.json](tag_config.json) 中填写固定 Tag 实际中心坐标、Unity UDP 地址、Homography 更新、平滑系数、`blind_*` 盲开参数和 `headless`。示例 IP、摄像头索引及朝向偏移需按现场修改；初始化时四个固定 Tag 必须同时可见并处于同一平面。

```powershell
python main.py --set-car-heading 90
python main.py --show-car-heading
python track_calibrate.py
```

## 目录

```text
main.py / tag_config.json     # 定位、配置与 UDP
track_calibrate.py            # 赛道底图标定
generate_tag.py               # 生成 ArUco Tag
calibrate_camera.py           # 棋盘格相机标定
test_*.py                     # 摄像头、IPM、UDP 调试工具
track_map_*.png/json          # 赛道底图与标定数据
本次定位优化说明.md            # 定位与调参说明
```

盲开保护不是安全急停。请先在低速、可人工接管条件下验证坐标轴、车头偏移、Unity 接收端和摄像头编号。
