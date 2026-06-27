"""
SeeingTag — ARUCO 智能车定位系统（Homography 平面映射版）
核心逻辑：检测标签 → 单应矩阵映射 → UDP 发给 Unity
特点：只需平面坐标 (X, Z)，无需相机内参标定
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import socket
import time
import sys
import os
import argparse
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


def load_config(path: str) -> dict:
    """
    加载配置文件
    参数:
        path: 配置文件路径
    返回:
        配置字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    tag_positions: Dict[int, np.ndarray] = {}
    for tid, pos in cfg["tag_world_positions"].items():
        tag_positions[int(tid)] = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
    
    # 获取 H 矩阵更新模式配置，默认为 "interval"
    homography_mode = cfg.get("homography_update_mode", "interval")
    # 验证模式值是否合法
    valid_modes = ("once", "interval", "every_frame")
    if homography_mode not in valid_modes:
        print(f"[WARN] 无效的 homography_update_mode: {homography_mode}，使用默认值 'interval'")
        homography_mode = "interval"
    
    # 获取更新间隔，默认 30 帧
    homography_interval = cfg.get("homography_update_interval", 30)
    if homography_interval < 1:
        homography_interval = 30

    car_heading_offset = cfg.get("car_heading_offset_degrees", 0.0)
    if not isinstance(car_heading_offset, (int, float)) or not math.isfinite(car_heading_offset):
        print("[WARN] 无效的 car_heading_offset_degrees，使用 0°")
        car_heading_offset = 0.0

    motion_history_seconds = cfg.get("motion_history_seconds", 4.0)
    blind_drive_seconds = cfg.get("blind_drive_seconds", 1.2)
    position_hold_seconds = cfg.get("position_hold_seconds", 0.35)
    yaw_log_interval_seconds = cfg.get("yaw_log_interval_seconds", 0.5)
    blind_speed_window_seconds = cfg.get("blind_speed_window_seconds", 0.5)
    blind_yaw_rate_window_seconds = cfg.get("blind_yaw_rate_window_seconds", 0.5)
    blind_yaw_rate_scale = cfg.get("blind_yaw_rate_scale", 0.35)
    blind_max_yaw_rate_dps = cfg.get("blind_max_yaw_rate_dps", 25.0)
    blind_yaw_rate_decay_per_second = cfg.get("blind_yaw_rate_decay_per_second", 0.75)
    blind_max_speed_mps = cfg.get("blind_max_speed_mps", 1.2)
    blind_max_distance_m = cfg.get("blind_max_distance_m", 0.45)
    if motion_history_seconds <= 0.1:
        motion_history_seconds = 4.0
    if blind_drive_seconds < 0.0:
        blind_drive_seconds = 0.0
    if position_hold_seconds < 0.0:
        position_hold_seconds = 0.0
    if yaw_log_interval_seconds <= 0.0:
        yaw_log_interval_seconds = 0.5
    if blind_speed_window_seconds <= 0.05:
        blind_speed_window_seconds = 0.5
    if blind_yaw_rate_window_seconds <= 0.05:
        blind_yaw_rate_window_seconds = 0.5
    blind_yaw_rate_scale = float(np.clip(blind_yaw_rate_scale, 0.0, 1.0))
    if blind_max_yaw_rate_dps < 0.0:
        blind_max_yaw_rate_dps = 0.0
    blind_yaw_rate_decay_per_second = float(np.clip(blind_yaw_rate_decay_per_second, 0.0, 1.0))
    if blind_max_speed_mps <= 0.0:
        blind_max_speed_mps = 1.2
    if blind_max_distance_m <= 0.0:
        blind_max_distance_m = 0.45
    
    return {
        "tag_size_m": cfg["tag_size_m"],
        "min_tag_width": cfg.get("min_tag_width_pixels", 10),
        "field_width_m": cfg.get("field_width_m", 5.0),
        "field_height_m": cfg.get("field_height_m", 4.0),
        "filter_alpha": cfg.get("filter_alpha", 1.0),
        "output_filter_alpha": cfg.get("output_filter_alpha", 1.0),
        "flip_x": cfg.get("flip_x", False),
        "flip_z": cfg.get("flip_z", False),
        "debug_logging": cfg.get("debug_logging", False),
        "unity_ip": cfg["unity_ip"],
        "unity_port": cfg["unity_port"],
        "tag_positions": tag_positions,
        "car_tag_id": cfg["car_tag_id"],
        "car_heading_offset_degrees": normalize_angle(float(car_heading_offset)),
        "motion_history_seconds": float(motion_history_seconds),
        "blind_drive_seconds": float(blind_drive_seconds),
        "position_hold_seconds": float(position_hold_seconds),
        "yaw_log_interval_seconds": float(yaw_log_interval_seconds),
        "blind_speed_window_seconds": float(blind_speed_window_seconds),
        "blind_yaw_rate_window_seconds": float(blind_yaw_rate_window_seconds),
        "blind_yaw_rate_scale": float(blind_yaw_rate_scale),
        "blind_max_yaw_rate_dps": float(blind_max_yaw_rate_dps),
        "blind_yaw_rate_decay_per_second": float(blind_yaw_rate_decay_per_second),
        "blind_max_speed_mps": float(blind_max_speed_mps),
        "blind_max_distance_m": float(blind_max_distance_m),
        "homography_update_mode": homography_mode,
        "homography_update_interval": homography_interval,
        "bird_eye_only": cfg.get("bird_eye_only", False),
        "tuning_udp_enabled": cfg.get("tuning_udp_enabled", True),
        "tuning_ip": cfg.get("tuning_ip", "127.0.0.1"),
        "tuning_port": cfg.get("tuning_port", 9010),
        "tuning_defaults": cfg.get("tuning_defaults", {"k": 30.0, "p": 1.0, "d": 0.10}),
    }


def save_car_heading_offset(path: str, angle: float) -> float:
    """将车头相对 Tag 的角度偏移写入配置文件，并返回规范化后的角度。"""
    if not math.isfinite(angle):
        raise ValueError("角度必须是有限数字")

    normalized_angle = normalize_angle(angle)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["car_heading_offset_degrees"] = normalized_angle
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return normalized_angle


def truncate_to_2_decimals(value: float) -> float:
    """
    截断浮点数到两位小数（直接舍弃第三位及以后，不四舍五入）
    例如：2.357 → 2.35, -1.239 → -1.23
    """
    return int(value * 100) / 100.0


def normalize_angle(angle: float) -> float:
    """将角度约束到 [-180, 180) 范围。"""
    return (angle + 180.0) % 360.0 - 180.0


def smooth_angle(previous: float, current: float, alpha: float) -> float:
    """沿最短旋转方向平滑 yaw，避免 179° 到 -179° 时绕一整圈。"""
    delta = normalize_angle(current - previous)
    return normalize_angle(previous + alpha * delta)


def append_motion_sample(history: Deque[Tuple[float, float, float, float]],
                         timestamp: float, x: float, z: float, yaw: float,
                         window_seconds: float):
    """保存最近几秒的位置样本，用于车标签短时丢失时外推。"""
    history.append((timestamp, x, z, yaw))
    while history and timestamp - history[0][0] > window_seconds:
        history.popleft()


def predict_blind_position(history: Deque[Tuple[float, float, float, float]],
                           timestamp: float, max_blind_seconds: float,
                           blind_elapsed_seconds: float,
                           speed_window: float = 0.5,
                           yaw_rate_window: float = 0.5,
                           yaw_rate_scale: float = 0.35,
                           max_yaw_rate_dps: float = 25.0,
                           yaw_rate_decay_per_second: float = 0.75,
                           max_speed_mps: float = 1.2
                           ) -> Optional[Tuple[float, float, float]]:
    """受限 CTRV 模型：按近期弧线趋势外推，但限制转向和速度。

    速度用短窗口平均，yaw 角速度会缩放、限幅，并随盲开时间衰减；
    这样弯道出口丢失时会逐渐趋向直线，而不是一直按大圆弧拐。
    """
    if len(history) < 3:  # 至少 3 个点才能估计曲率
        return None

    curr_t, curr_x, curr_z, curr_yaw = history[-1]
    step_seconds = timestamp - curr_t
    if step_seconds < 0.0 or blind_elapsed_seconds > max_blind_seconds:
        return None

    if step_seconds < 0.001:
        return None

    # 1) 速度大小：最近 speed_window 秒的路径长度 / 时间，比只用最后两帧抗抖。
    speed_cutoff_t = curr_t - speed_window
    total_distance = 0.0
    speed_total_dt = 0.0
    for i in range(len(history) - 1, 0, -1):
        t1, x1, z1, _ = history[i - 1]
        t2, x2, z2, _ = history[i]
        if t2 <= speed_cutoff_t:
            break
        dt = t2 - t1
        if dt > 0.001:
            total_distance += math.hypot(x2 - x1, z2 - z1)
            speed_total_dt += dt
    if speed_total_dt <= 0.001:
        return None
    speed = min(total_distance / speed_total_dt, max_speed_mps)

    # 2) 角速度：最近 yaw_rate_window 秒平均，然后缩放、限幅、随时间衰减。
    cutoff_t = curr_t - yaw_rate_window
    total_dyaw = 0.0
    yaw_total_dt = 0.0
    for i in range(len(history) - 1, 0, -1):
        t1, _, _, y1 = history[i - 1]
        t2, _, _, y2 = history[i]
        if t2 <= cutoff_t:
            break
        dt = t2 - t1
        if dt > 0.001:
            total_dyaw += normalize_angle(y2 - y1)
            yaw_total_dt += dt
    omega = total_dyaw / yaw_total_dt if yaw_total_dt > 0.001 else 0.0  # deg/s
    omega *= yaw_rate_scale
    omega = float(np.clip(omega, -max_yaw_rate_dps, max_yaw_rate_dps))
    omega *= yaw_rate_decay_per_second ** max(0.0, blind_elapsed_seconds)

    # 3) CTRV 沿圆弧外推
    dyaw = omega * step_seconds
    yaw_rad = math.radians(curr_yaw)
    dyaw_rad = math.radians(dyaw)
    omega_rad = math.radians(omega) if abs(omega) > 0.001 else 0.0

    if abs(omega) < 0.5 or speed < 0.001:
        # 近似直线
        new_x = curr_x + speed * math.cos(yaw_rad) * step_seconds
        new_z = curr_z + speed * math.sin(yaw_rad) * step_seconds
        new_yaw = curr_yaw
    else:
        # 圆弧: R = v / ω
        R = speed / omega_rad
        new_x = curr_x + R * (math.sin(yaw_rad + dyaw_rad) - math.sin(yaw_rad))
        new_z = curr_z + R * (math.cos(yaw_rad) - math.cos(yaw_rad + dyaw_rad))
        new_yaw = normalize_angle(curr_yaw + dyaw)

    return (new_x, new_z, new_yaw)


def draw_blind_status_overlay(image: np.ndarray, armed: bool, active: bool,
                              message: str = "", message_visible: bool = False):
    """在画面左上角显示盲开保护状态，方便试车时确认 1/0 是否生效。"""
    if image is None:
        return

    if active:
        status_text = "BLIND PROTECT: DRIVING"
        status_color = (0, 165, 255)
    elif armed:
        status_text = "BLIND PROTECT: ARMED"
        status_color = (0, 255, 255)
    else:
        status_text = "BLIND PROTECT: OFF"
        status_color = (150, 150, 150)

    cv2.rectangle(image, (8, 8), (390, 78), (0, 0, 0), -1)
    cv2.rectangle(image, (8, 8), (390, 78), status_color, 2)
    cv2.putText(image, status_text, (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(image, "1=ONCE ARM   0=OFF", (20, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    if message_visible and message:
        cv2.rectangle(image, (8, 86), (520, 126), (0, 0, 0), -1)
        cv2.putText(image, message, (20, 113),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)


class UdpSender:
    def __init__(self, target_ip: str, target_port: int, output_filter_alpha: float = 1.0,
                 debug_logging: bool = False):
        self.target = (target_ip, target_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.output_filter_alpha = float(np.clip(output_filter_alpha, 0.0, 1.0))
        self.debug_logging = debug_logging
        self.filtered_output: Optional[Tuple[float, float, float]] = None

    def send(self, x: float, z: float, yaw: float = 0.0, tracking_state: str = "raw") -> bool:
        """
        通过UDP发送位置和偏航角数据
        参数:
            x: 世界坐标X（米）
            z: 世界坐标Z（米）
            yaw: 偏航角（度），范围 -180 ~ 180
        返回:
            发送成功返回True，失败返回False
        """
        try:
            # 单独平滑 Unity 输出；画面中的位置仍使用视觉追踪层的平滑结果。
            if self.filtered_output is None:
                self.filtered_output = (x, z, yaw)
            else:
                old_x, old_z, old_yaw = self.filtered_output
                alpha = self.output_filter_alpha
                self.filtered_output = (
                    old_x + alpha * (x - old_x),
                    old_z + alpha * (z - old_z),
                    smooth_angle(old_yaw, yaw, alpha),
                )
            x, z, yaw = self.filtered_output

            # 截断到两位小数
            x_truncated = truncate_to_2_decimals(x)
            z_truncated = truncate_to_2_decimals(z)
            yaw_truncated = truncate_to_2_decimals(yaw)
            
            data = {
                "type": "robot_position",
                "pos": [float(z_truncated), 0.10, float(x_truncated)],  # Unity坐标系：交换x和z的顺序
                "euler": [0.0, float(yaw_truncated), 0.0],  # yaw角（绕Y轴旋转）
                "tracking_state": tracking_state,
                "seq": self.seq,
                "timestamp": time.time()
            }
            self.seq += 1
            self.sock.sendto(json.dumps(data).encode('utf-8'), self.target)
            if self.debug_logging:
                print(f"[UDP] → {self.target[0]}:{self.target[1]} pos=({z_truncated:.2f}, 0.0, {x_truncated:.2f}) yaw={yaw_truncated:.1f}° seq={self.seq-1} state={tracking_state}")
            return True
        except Exception as e:
            print(f"[UDP] 发送失败: {e}")
            return False

    def close(self):
        self.sock.close()


class TuningUdpSender:
    """Send K/P/D tuning values for SmartCar to save and apply on next restart."""

    def __init__(self, target_ip: str, target_port: int, enabled: bool = True):
        self.target = (target_ip, int(target_port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enabled = bool(enabled)
        self.seq = 0

    def send_once(self, k: float, p: float, d: float) -> bool:
        if not self.enabled:
            return False

        values = (round(float(k), 3), round(float(p), 3), round(float(d), 3))
        now = time.time()
        data = {
            "type": "control_tuning",
            "version": 1,
            "action": "save_next_restart",
            "seq": self.seq,
            "timestamp": now,
            "params": {
                "pwm_k": values[0],
                "pid_p": values[1],
                "pid_d": values[2],
            },
        }
        try:
            self.sock.sendto(json.dumps(data, separators=(",", ":")).encode("utf-8"), self.target)
            self.seq += 1
            print(
                f"[TuneUDP] save_next_restart -> {self.target[0]}:{self.target[1]} "
                f"K={values[0]:.2f} P={values[1]:.2f} D={values[2]:.2f}"
            )
            return True
        except OSError as e:
            print(f"[TuneUDP] 发送失败: {e}")
            return False

    def close(self):
        self.sock.close()


class TagTracker:
    def __init__(self, cfg: dict):
        self.tag_size = cfg["tag_size_m"]
        self.min_tag_width = cfg["min_tag_width"]
        self.tag_positions = cfg["tag_positions"]
        self.car_tag_id = cfg["car_tag_id"]
        self.field_width = cfg["field_width_m"]
        self.field_height = cfg["field_height_m"]
        self.flip_x = cfg["flip_x"]
        self.flip_z = cfg["flip_z"]
        # 用于把 Tag 的默认方向校正为真实车头；配置会同时影响 HUD、鸟瞰箭头和 UDP yaw。
        self.car_heading_offset_degrees = cfg["car_heading_offset_degrees"]
        self.debug_logging = cfg["debug_logging"]
        
        # H 矩阵更新模式配置
        self.homography_update_mode = cfg.get("homography_update_mode", "interval")
        self.homography_update_interval = cfg.get("homography_update_interval", 30)

        # ArUco 检测器
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 5
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.perspectiveRemovePixelPerCell = 8
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

        # UDP 发送器
        self.udp: Optional[UdpSender] = UdpSender(
            cfg["unity_ip"], cfg["unity_port"], cfg["output_filter_alpha"], self.debug_logging
        )

        # 单应矩阵缓存
        self.H: Optional[np.ndarray] = None
        self.homography_valid = False
        # 标记 H 矩阵是否已初始化（用于 "once" 模式）
        self.homography_initialized = False

    def detect(self, gray: np.ndarray) -> List[dict]:
        """
        检测所有可见的 ARUCO 标签
        参数:
            gray: 灰度图像
        返回:
            检测结果列表，每个元素包含 id, corners, width_px
        """
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        results = []
        for i, tag_id in enumerate(ids.flatten()):
            c = corners[i][0] if corners[i].shape[0] == 1 else corners[i]
            w = abs(c[0][0] - c[1][0])
            if w < self.min_tag_width:
                continue
            results.append({"id": int(tag_id), "corners": c.copy(), "width_px": w})
        return results

    def detect_car_only(self, gray: np.ndarray) -> Optional[dict]:
        """
        只检测车标签（优化性能）
        参数:
            gray: 灰度图像
        返回:
            车标签检测结果，未检测到则返回 None
        """
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        
        for i, tag_id in enumerate(ids.flatten()):
            if tag_id == self.car_tag_id:
                c = corners[i][0] if corners[i].shape[0] == 1 else corners[i]
                w = abs(c[0][0] - c[1][0])
                if w >= self.min_tag_width:
                    return {"id": int(tag_id), "corners": c.copy(), "width_px": w}
        return None

    def compute_homography(self, detections: List[dict], keep_existing: bool = False) -> bool:
        """
        用固定标签的中心点计算单应矩阵。

        每个固定标签只贡献一对「图像中心 → 场地中心」对应点，因此标签
        可以按识别效果任意旋转；为计算 H 必须同时看见至少 4 个固定标签。
        """
        src_pts = []  # 像素坐标 (u, v)
        dst_pts = []  # 世界坐标 (x, z)

        for d in detections:
            if d["id"] not in self.tag_positions:
                continue
            src_pts.append(np.mean(d["corners"], axis=0).astype(np.float64))
            tag_center = self.tag_positions[d["id"]]
            dst_pts.append([tag_center[0], tag_center[2]])

        # 至少需要 4 个点才能计算单应矩阵
        if len(src_pts) < 4:
            if not keep_existing:
                self.homography_valid = False
                self.H = None
            return False

        src_pts = np.array(src_pts, dtype=np.float64)
        dst_pts = np.array(dst_pts, dtype=np.float64)

        # 使用 RANSAC 鲁棒估计单应矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

        if H is not None:
            self.H = H
            self.homography_valid = True
            inlier_count = int(np.sum(mask))
            if self.debug_logging:
                print(f"[Homography] 计算成功: {len(src_pts)}个点, {inlier_count}个内点")
            return True
        else:
            if not keep_existing:
                self.homography_valid = False
                self.H = None
            return False

    def compute_yaw_from_corners(self, corners: np.ndarray) -> Optional[float]:
        """
        通过标签角点计算偏航角
        原理：角点0→角点1的连线方向代表标签的"前方"，映射到世界坐标系后计算角度
        参数:
            corners: 标签4个角点的像素坐标，shape=(4, 2)
        返回:
            yaw角（度），失败返回None
        """
        if not self.homography_valid or self.H is None:
            return None
        
        # 获取角点0和角点1的像素坐标（角点顺序：左上、右上、右下、左下）
        # 角点0→角点1的方向代表标签的"前方"（X轴正方向）
        p0 = corners[0]  # 左上角
        p1 = corners[1]  # 右上角
        
        # 计算标签中心点
        center = np.mean(corners, axis=0)
        
        # 计算方向向量（从中心指向标签前方）
        # 使用角点0和角点1的中点作为参考
        front_mid = (p0 + p1) / 2.0
        direction_pixel = front_mid - center
        
        # 如果方向向量太小，使用角点0→角点1的连线方向
        if np.linalg.norm(direction_pixel) < 5:
            direction_pixel = p1 - p0
        
        # 归一化方向向量
        norm = np.linalg.norm(direction_pixel)
        if norm < 1e-6:
            return None
        direction_pixel = direction_pixel / norm
        
        # 通过单应矩阵映射方向向量到世界坐标系
        # 注意：单应矩阵是3x3，方向向量需要齐次坐标
        # 映射中心点和方向终点，然后计算世界坐标系中的方向
        center_h = np.array([[center[0], center[1]]], dtype=np.float64)
        end_point = center + direction_pixel * 100  # 延长方向向量
        end_h = np.array([[end_point[0], end_point[1]]], dtype=np.float64)
        
        # 映射到世界坐标
        center_world = cv2.perspectiveTransform(center_h.reshape(1, 1, 2), self.H)[0, 0]
        end_world = cv2.perspectiveTransform(end_h.reshape(1, 1, 2), self.H)[0, 0]
        
        # 计算世界坐标系中的方向向量
        direction_world = end_world - center_world
        direction_world = direction_world / (np.linalg.norm(direction_world) + 1e-9)
        
        # 计算yaw角（绕Y轴旋转角度）
        # 世界坐标系：X轴正方向为0度，逆时针为正
        # direction_world = [dx, dz]
        yaw_rad = np.arctan2(direction_world[1], direction_world[0])  # atan2(dz, dx)
        yaw_deg = np.degrees(yaw_rad)
        
        return float(yaw_deg)

    def locate(self, detections: List[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        定位车标签，返回世界坐标 (x, z) 和偏航角 yaw
        此方法会重新计算 H 矩阵
        参数:
            detections: 所有检测到的标签列表
        返回:
            (x, z, yaw) 世界坐标和偏航角（度），失败返回 (None, None, None)
        """
        # 第 1 步：更新单应矩阵
        self.compute_homography(detections)

        if not self.homography_valid or self.H is None:
            return None, None, None

        # 第 2 步：找到车标签的中心像素坐标
        car_det = next((d for d in detections if d["id"] == self.car_tag_id), None)
        if car_det is None:
            return None, None, None

        car_center_pixel = np.mean(car_det["corners"], axis=0)

        # 第 3 步：通过单应矩阵映射到世界坐标
        pixel_pt = np.array([[car_center_pixel[0], car_center_pixel[1]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pixel_pt.reshape(1, 1, 2), self.H)
        wx, wz = world_pt[0, 0, 0], world_pt[0, 0, 1]

        # 第 4 步：计算偏航角
        yaw = self.compute_yaw_from_corners(car_det["corners"])

        return float(wx), float(wz), yaw

    def locate_with_cached_H(self, car_det: Optional[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        使用缓存的 H 矩阵定位车标签（不重新计算 H）
        参数:
            car_det: 车标签检测结果（来自 detect_car_only）
        返回:
            (x, z, yaw) 世界坐标和偏航角（度），失败返回 (None, None, None)
        """
        # 检查 H 矩阵是否有效
        if not self.homography_valid or self.H is None:
            return None, None, None
        
        # 检查车标签是否检测到
        if car_det is None:
            return None, None, None

        # 计算车标签中心像素坐标
        car_center_pixel = np.mean(car_det["corners"], axis=0)

        # 通过缓存的 H 矩阵映射到世界坐标
        pixel_pt = np.array([[car_center_pixel[0], car_center_pixel[1]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pixel_pt.reshape(1, 1, 2), self.H)
        wx, wz = world_pt[0, 0, 0], world_pt[0, 0, 1]

        # 计算偏航角
        yaw = self.compute_yaw_from_corners(car_det["corners"])

        return float(wx), float(wz), yaw

    def warp_to_bird_eye(self, frame: np.ndarray, output_width: int = 800,
                         output_height: int = 640) -> Optional[Tuple[np.ndarray, dict]]:
        """将原始画面映射到场地鸟瞰图，并返回坐标换算参数。"""
        if not self.homography_valid or self.H is None:
            return None

        # 场地为 5m × 4m，四周留出 0.5m，便于观察边缘的车标签。
        margin = 0.5
        world_min_x, world_max_x = -margin, self.field_width + margin
        world_min_z, world_max_z = -margin, self.field_height + margin
        scale_x = output_width / (world_max_x - world_min_x)
        scale_z = output_height / (world_max_z - world_min_z)

        world_corners = np.array([
            [world_min_x, world_min_z, 1],
            [world_max_x, world_min_z, 1],
            [world_max_x, world_max_z, 1],
            [world_min_x, world_max_z, 1]
        ], dtype=np.float64)

        H_inv = np.linalg.inv(self.H)
        src_pts = []
        for corner in world_corners:
            pixel = H_inv @ corner
            pixel = pixel / pixel[2]
            src_pts.append([pixel[0], pixel[1]])
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array([
            [0, 0], [output_width, 0],
            [output_width, output_height], [0, output_height]
        ], dtype=np.float32)

        transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bird_eye = cv2.warpPerspective(
            frame, transform, (output_width, output_height), flags=cv2.INTER_CUBIC
        )
        return bird_eye, {
            "world_min_x": world_min_x,
            "world_min_z": world_min_z,
            "scale_x": scale_x,
            "scale_z": scale_z,
        }

    def locate_car_in_bird_eye(self, bird_eye: np.ndarray, transform_info: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """在已拉正的鸟瞰图中检测车标签，作为原图检测失败时的兜底。"""
        gray = cv2.cvtColor(bird_eye, cv2.COLOR_BGR2GRAY)
        car_det = self.detect_car_only(gray)
        if car_det is None:
            return None, None, None

        # 鸟瞰图的像素坐标可直接线性换算为场地 X/Z 坐标。
        center = np.mean(car_det["corners"], axis=0)
        world_x = transform_info["world_min_x"] + center[0] / transform_info["scale_x"]
        world_z = transform_info["world_min_z"] + center[1] / transform_info["scale_z"]

        # 同样将角点方向换算到世界坐标后计算偏航角。
        p0, p1 = car_det["corners"][0], car_det["corners"][1]
        direction_x = (p1[0] - p0[0]) / transform_info["scale_x"]
        direction_z = (p1[1] - p0[1]) / transform_info["scale_z"]
        yaw = float(np.degrees(np.arctan2(direction_z, direction_x)))
        return float(world_x), float(world_z), yaw

    def transform_output_coordinates(self, x: float, z: float,
                                     yaw: Optional[float]) -> Tuple[float, float, Optional[float]]:
        """按配置镜像场地坐标，并同步修正车辆朝向。"""
        if self.flip_x:
            x = self.field_width - x
            if yaw is not None:
                yaw = 180.0 - yaw
        if self.flip_z:
            z = self.field_height - z
            if yaw is not None:
                yaw = -yaw
        if yaw is not None:
            yaw = normalize_angle(yaw + self.car_heading_offset_degrees)
        return x, z, yaw

    def send_position(self, x: float, z: float, yaw: float = 0.0,
                      tracking_state: str = "raw"):
        """
        发送位置和偏航角数据
        参数:
            x: 世界坐标X（米）
            z: 世界坐标Z（米）
            yaw: 偏航角（度）
        """
        if self.udp is not None:
            self.udp.send(x, z, yaw, tracking_state)

    def draw_hud(self, frame: np.ndarray, detections: List[dict],
                 car_x: Optional[float], car_z: Optional[float], 
                 car_yaw: Optional[float], fps: float) -> np.ndarray:
        """
        在画面上绘制HUD信息
        参数:
            frame: 原始图像
            detections: 检测结果列表
            car_x: 车辆世界坐标X
            car_z: 车辆世界坐标Z
            car_yaw: 车辆偏航角（度）
            fps: 当前帧率
        返回:
            绘制了HUD的图像
        """
        display = frame.copy()

        # 画标签边框
        for d in detections:
            color = (0, 255, 0) if d["id"] in self.tag_positions else (0, 128, 255)
            pts = d["corners"].astype(np.int32)
            cv2.polylines(display, [pts], True, color, 2)
            cx, cy = int(d['corners'][0][0]), int(d['corners'][0][1])
            label = "CAR" if d["id"] == self.car_tag_id else f"ID:{d['id']}"
            cv2.putText(display, label, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # HUD 信息
        h, w = frame.shape[:2]
        fixed_count = sum(1 for d in detections if d["id"] in self.tag_positions)
        car_visible = any(d["id"] == self.car_tag_id for d in detections)

        lines = []
        if car_x is not None and car_z is not None:
            # 截断到两位小数后显示
            car_x_display = truncate_to_2_decimals(car_x)
            car_z_display = truncate_to_2_decimals(car_z)
            lines.append(f"Car: X={car_x_display:.2f} Z={car_z_display:.2f}")
            # 显示偏航角
            if car_yaw is not None:
                car_yaw_display = truncate_to_2_decimals(car_yaw)
                lines.append(f"Yaw: {car_yaw_display:.1f} deg")
        lines.append(f"Fixed: {fixed_count}/{len(self.tag_positions)}  Car tag: {'YES' if car_visible else 'NO'}")
        if self.homography_valid:
            lines.append(f"Homography: OK")
        else:
            lines.append(f"Homography: Need >=4 points")

        for idx, line in enumerate(lines):
            cv2.putText(display, line, (10, 30 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 右下角 FPS
        cv2.putText(display, f"FPS: {fps:.0f}", (w - 110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        return display

    def generate_bird_eye_view(self, frame: np.ndarray, detections: List[dict],
                               car_x: Optional[float], car_z: Optional[float],
                               car_yaw: Optional[float] = None) -> Optional[np.ndarray]:
        """
        生成鸟瞰图（俯视图）：对原图进行逆透视变换
        将相机视角的图像映射到世界平面视角
        参数:
            frame: 原始图像
            detections: 检测结果列表
            car_x: 车辆世界坐标X
            car_z: 车辆世界坐标Z
            car_yaw: 车辆偏航角（度）
        返回:
            鸟瞰图图像，失败返回None
        """
        if not self.homography_valid or self.H is None:
            return None

        output_width = 800
        output_height = 640
        warped = self.warp_to_bird_eye(frame, output_width, output_height)
        if warped is None:
            return None
        bird_eye, transform_info = warped
        if self.flip_x or self.flip_z:
            flip_code = -1 if self.flip_x and self.flip_z else (1 if self.flip_x else 0)
            bird_eye = cv2.flip(bird_eye, flip_code)
        world_min_x = transform_info["world_min_x"]
        world_min_z = transform_info["world_min_z"]
        scale_x = transform_info["scale_x"]
        scale_z = transform_info["scale_z"]

        # 绘制场地边界（红色矩形）
        boundary_color = (0, 0, 255)
        top_left = (int((0 - world_min_x) * scale_x), int((0 - world_min_z) * scale_z))
        top_right = (int((self.field_width - world_min_x) * scale_x), int((0 - world_min_z) * scale_z))
        bottom_right = (int((self.field_width - world_min_x) * scale_x), int((self.field_height - world_min_z) * scale_z))
        bottom_left = (int((0 - world_min_x) * scale_x), int((self.field_height - world_min_z) * scale_z))
        
        cv2.line(bird_eye, top_left, top_right, boundary_color, 2)
        cv2.line(bird_eye, top_right, bottom_right, boundary_color, 2)
        cv2.line(bird_eye, bottom_right, bottom_left, boundary_color, 2)
        cv2.line(bird_eye, bottom_left, top_left, boundary_color, 2)

        # 绘制固定标签位置（绿色圆圈）
        for tag_id, pos in self.tag_positions.items():
            marker_x, marker_z, _ = self.transform_output_coordinates(float(pos[0]), float(pos[2]), None)
            bx = int((marker_x - world_min_x) * scale_x)
            bz = int((marker_z - world_min_z) * scale_z)
            cv2.circle(bird_eye, (bx, bz), 10, (0, 255, 0), -1)
            cv2.circle(bird_eye, (bx, bz), 12, (0, 200, 0), 2)
            cv2.putText(bird_eye, str(tag_id), (bx + 12, bz + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制车标签位置（黄色圆圈 + 朝向箭头）
        if car_x is not None and car_z is not None:
            bx = int((car_x - world_min_x) * scale_x)
            bz = int((car_z - world_min_z) * scale_z)
            # 确保坐标在图像范围内
            if 0 <= bx < output_width and 0 <= bz < output_height:
                cv2.circle(bird_eye, (bx, bz), 12, (0, 255, 255), -1)
                cv2.circle(bird_eye, (bx, bz), 14, (0, 200, 200), 2)
                cv2.putText(bird_eye, "CAR", (bx + 15, bz + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 绘制朝向箭头（如果有yaw角）
                if car_yaw is not None:
                    arrow_length = 30  # 箭头长度（像素）
                    yaw_rad = np.radians(car_yaw)
                    # 计算箭头终点（注意：图像坐标系Y轴向下，所以Z方向需要取反）
                    # 世界坐标系：X向右，Z向上；图像坐标系：X向右，Y向下
                    end_x = int(bx + arrow_length * np.cos(yaw_rad))
                    end_z = int(bz + arrow_length * np.sin(yaw_rad))
                    cv2.arrowedLine(bird_eye, (bx, bz), (end_x, end_z), (0, 255, 255), 3, tipLength=0.4)

        # 添加标题
        cv2.putText(bird_eye, "Bird Eye View (Top-Down)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加坐标轴说明
        cv2.putText(bird_eye, f"X: 0-{self.field_width:g}m", (10, output_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(bird_eye, f"Z: 0-{self.field_height:g}m", (10, output_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return bird_eye

    def print_debug_info(self, detections: List[dict]):
        fixed = [d for d in detections if d["id"] in self.tag_positions]
        car_det = next((d for d in detections if d["id"] == self.car_tag_id), None)
        print(f"\n--- Debug ---")
        print(f"  Tags detected: {[d['id'] for d in detections]}")
        print(f"  Fixed tags: {len(fixed)}/{len(self.tag_positions)}")
        for d in fixed:
            print(f"    ID {d['id']}: corners[0]=({d['corners'][0][0]:.1f}, {d['corners'][0][1]:.1f}) w={d['width_px']}px")
        if car_det is not None:
            center = np.mean(car_det["corners"], axis=0)
            print(f"  Car tag(ID {self.car_tag_id}): center=({center[0]:.1f}, {center[1]:.1f}) w={car_det['width_px']}px")
        if self.homography_valid:
            print(f"  Homography: Valid")
        else:
            print(f"  Homography: Invalid (need >=4 points)")
        print("------------------------\n")

    def close(self):
        if self.udp is not None:
            self.udp.close()


def list_available_cameras(max_test: int = 5) -> List[int]:
    """
    检测可用的摄像头索引
    参数:
        max_test: 最大测试的摄像头索引数
    返回:
        可用的摄像头索引列表
    """
    available = []
    print("[Camera] 正在检测可用摄像头...")
    
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 尝试读取一帧来验证摄像头真正可用
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                available.append(i)
                print(f"  [{i}] 摄像头可用 - 分辨率: {w}x{h}")
            cap.release()
    
    if not available:
        print("[Camera] 未检测到可用摄像头!")
    else:
        print(f"[Camera] 共检测到 {len(available)} 个可用摄像头")
    
    return available


def select_camera() -> int:
    """
    自动检测摄像头并让用户选择
    返回:
        用户选择的摄像头索引
    """
    available = list_available_cameras()
    
    if not available:
        print("[ERROR] 没有可用的摄像头!")
        sys.exit(1)
    
    if len(available) == 1:
        print(f"[Camera] 只有一个摄像头，自动选择: {available[0]}")
        return available[0]
    
    # 多个摄像头，让用户选择
    print("\n请选择要使用的摄像头:")
    for i in available:
        print(f"  输入 {i} 选择摄像头 {i}")
    
    while True:
        try:
            choice = input("请输入摄像头索引: ").strip()
            choice = int(choice)
            if choice in available:
                print(f"[Camera] 已选择摄像头: {choice}")
                return choice
            else:
                print(f"[ERROR] 无效选择，请输入: {available}")
        except ValueError:
            print("[ERROR] 请输入数字!")
        except KeyboardInterrupt:
            print("\n[INFO] 用户取消")
            sys.exit(0)


def open_camera(camera_index: Optional[int] = None) -> cv2.VideoCapture:
    """
    打开本地 USB 摄像头
    参数:
        camera_index: 指定摄像头索引，None 则自动选择
    返回:
        cv2.VideoCapture 对象
    """
    # 如果没有指定摄像头索引，则自动检测并选择
    if camera_index is None:
        camera_index = select_camera()
    
    print(f"[Camera] 正在打开摄像头 {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_index}!")
        sys.exit(1)

    # 设置 MJPG 格式
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 尝试不同的分辨率（优先使用摄像头原生分辨率）
    resolutions = [
        (1920, 1080),  # 摄像头原生分辨率
        (1280, 720),   # 常用分辨率
        (640, 480),    # 备选分辨率
    ]
    
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(0.3)
        
        # 验证设置是否生效
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w == w and actual_h == h:
            print(f"[Camera] 成功设置分辨率: {w}x{h}")
            break
        else:
            print(f"[Camera] 分辨率 {w}x{h} 不支持，实际: {actual_w}x{actual_h}")

    # 获取最终设置
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
    print(f"[Camera] USB 摄像头 — {w}x{h} @ {fps:.0f}fps  编码={fourcc_str}")
    
    # 测试是否能正常读取帧
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[ERROR] 摄像头无法读取帧，请检查:")
        print("  1. 摄像头是否被其他程序占用")
        print("  2. 摄像头驱动是否正常")
        print("  3. 尝试重新插拔摄像头")
        cap.release()
        sys.exit(1)
    print(f"[Camera] 帧读取测试成功，图像尺寸: {test_frame.shape}")
    
    return cap


class TrackMapVisualizer:
    """赛道调参窗口：底图、固定标签、车辆轨迹和 PID 滑条都集中在这里。"""

    WINDOW_NAME = "Track Map - 4x5"

    def __init__(self, cfg: dict, tag_positions: Dict[int, np.ndarray]):
        self.field_width = float(cfg.get("field_width_m", 5.0))
        self.field_height = float(cfg.get("field_height_m", 4.0))
        self.tag_positions = tag_positions
        self.history: Deque[Tuple[int, int]] = deque(maxlen=2000)
        self.H_track: Optional[np.ndarray] = None
        self.clear_button_rect = (52, 76, 194, 116)
        self.send_button_rect = (206, 76, 348, 116)
        self.pending_tune_send = False
        self.tune_status_text = "Tune: edit sliders, click Save KPD"
        self.tune_status_until = 0.0

        base_dir = os.path.dirname(__file__)
        self.display_tune = self._load_display_tune(base_dir)
        self.image_path = self._resolve_image_path(base_dir)
        self.raw_base_img = self._load_base_image(self.image_path)
        self.track_h, self.track_w = self.raw_base_img.shape[:2]
        self.left_curve_inset_px = int(np.clip(
            self.display_tune.get("left_curve_centerline_inset_px", 0), 0, 80
        ))
        self._last_left_curve_inset_px = self.left_curve_inset_px
        self.base_img = self._apply_left_curve_centerline_inset(
            self.raw_base_img, self.left_curve_inset_px
        )
        self._load_calibration(base_dir)

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        defaults = cfg.get("tuning_defaults", {})
        cv2.createTrackbar("K", self.WINDOW_NAME, int(float(defaults.get("k", 30.0)) * 10), 2000, self._noop)
        cv2.createTrackbar("P", self.WINDOW_NAME, int(float(defaults.get("p", 1.0)) * 100), 2000, self._noop)
        cv2.createTrackbar("D", self.WINDOW_NAME, int(float(defaults.get("d", 0.10)) * 100), 2000, self._noop)
        cv2.createTrackbar("LeftIn", self.WINDOW_NAME, self.left_curve_inset_px, 80, self._noop)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

    @staticmethod
    def _noop(_: int):
        pass

    def _resolve_image_path(self, base_dir: str) -> str:
        candidates = [
            os.path.join(base_dir, "track_map_clean.png"),
            os.path.join(base_dir, "track_map_source.png"),
            os.path.join(base_dir, "image.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _load_base_image(self, image_path: str) -> np.ndarray:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                return img

        img = np.full((675, 884, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (844, 635), (40, 40, 40), 2)
        cv2.putText(img, "4m x 5m Track Map", (40, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        return img

    def _load_display_tune(self, base_dir: str) -> dict:
        tune_path = os.path.join(base_dir, "track_display_tune.json")
        defaults = {
            "left_curve_center_px": [305, 356],
            "left_curve_max_x_px": 540,
            "left_curve_centerline_inset_px": 32,
        }
        if not os.path.exists(tune_path):
            return defaults
        try:
            with open(tune_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            defaults.update(loaded)
        except Exception as e:
            print(f"[TrackMap] Failed to load track_display_tune.json: {e}")
        return defaults

    def _apply_left_curve_centerline_inset(self, image: np.ndarray, inset_px: int) -> np.ndarray:
        if inset_px <= 0:
            return image.copy()

        tuned = image.copy()
        center = self.display_tune.get("left_curve_center_px", [305, 356])
        cx, cy = float(center[0]), float(center[1])
        max_x = int(self.display_tune.get("left_curve_max_x_px", 540))

        red_mask = (
            (image[:, :, 2] > 170) &
            (image[:, :, 1] < 150) &
            (image[:, :, 0] < 150)
        )
        red_mask[:, max(0, min(max_x, image.shape[1] - 1)):] = False

        if not np.any(red_mask):
            return tuned

        erase_mask = cv2.dilate(red_mask.astype(np.uint8) * 255, np.ones((5, 5), dtype=np.uint8))
        tuned[erase_mask > 0] = (255, 255, 255)

        ys, xs = np.nonzero(red_mask)
        dx = xs.astype(np.float32) - cx
        dy = ys.astype(np.float32) - cy
        radius = np.sqrt(dx * dx + dy * dy)
        valid = radius > 1.0
        scale = np.ones_like(radius)
        scale[valid] = np.maximum(1.0, radius[valid] - float(inset_px)) / radius[valid]
        new_xs = np.rint(cx + dx * scale).astype(np.int32)
        new_ys = np.rint(cy + dy * scale).astype(np.int32)

        valid = (
            (new_xs >= 0) & (new_xs < tuned.shape[1]) &
            (new_ys >= 0) & (new_ys < tuned.shape[0])
        )
        shifted_mask = np.zeros(tuned.shape[:2], dtype=np.uint8)
        shifted_mask[new_ys[valid], new_xs[valid]] = 255
        shifted_mask = cv2.dilate(shifted_mask, np.ones((3, 3), dtype=np.uint8))
        tuned[shifted_mask > 0] = (80, 80, 255)
        return tuned

    def _refresh_centerline_tune(self):
        inset_px = cv2.getTrackbarPos("LeftIn", self.WINDOW_NAME)
        if inset_px == self._last_left_curve_inset_px:
            return
        self.left_curve_inset_px = inset_px
        self._last_left_curve_inset_px = inset_px
        self.base_img = self._apply_left_curve_centerline_inset(
            self.raw_base_img, self.left_curve_inset_px
        )

    def _load_calibration(self, base_dir: str):
        calib_path = os.path.join(base_dir, "track_calib.json")
        if not os.path.exists(calib_path):
            print("[TrackMap] No track_calib.json, using linear 4x5 mapping.")
            return

        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                cobj = json.load(f)
            raw_H = np.array(cobj["H"], dtype=np.float32)
            calib_image = cobj.get("image")
            H = raw_H

            # track_calib.json may have been clicked on a different-size image.
            # Scale the world->pixel homography into the image actually shown now.
            if calib_image:
                if not os.path.exists(calib_image):
                    print(f"[TrackMap] Calibration image missing, ignoring H: {calib_image}")
                    return
                src_img = cv2.imread(calib_image)
                if src_img is None:
                    print(f"[TrackMap] Calibration image unreadable, ignoring H: {calib_image}")
                    return
                src_h, src_w = src_img.shape[:2]
                if src_w > 0 and src_h > 0 and (src_w != self.track_w or src_h != self.track_h):
                    scale = np.array([
                        [self.track_w / src_w, 0.0, 0.0],
                        [0.0, self.track_h / src_h, 0.0],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float32)
                    H = scale @ raw_H

            self.H_track = H
            print(f"[TrackMap] Loaded {calib_path}: {os.path.basename(self.image_path)}")
        except Exception as e:
            print(f"[TrackMap] Failed to load track_calib.json: {e}")
            self.H_track = None

    def world_to_px(self, x: float, z: float) -> Tuple[int, int]:
        if self.H_track is not None:
            try:
                pt = np.array([[[float(x), float(z)]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(pt, self.H_track)
                px = int(round(float(dst[0, 0, 0])))
                py = int(round(float(dst[0, 0, 1])))
                return self._clip_px(px, py)
            except Exception:
                pass

        fx = min(max(0.0, float(x)), self.field_width)
        fz = min(max(0.0, float(z)), self.field_height)
        px = int((fx / self.field_width) * (self.track_w - 1))
        py = int(((self.field_height - fz) / self.field_height) * (self.track_h - 1))
        return self._clip_px(px, py)

    def _clip_px(self, px: int, py: int) -> Tuple[int, int]:
        return (
            max(0, min(px, self.track_w - 1)),
            max(0, min(py, self.track_h - 1)),
        )

    def append_position(self, x: Optional[float], z: Optional[float]):
        if x is None or z is None:
            return
        self.history.append(self.world_to_px(x, z))

    def draw(self, car_x: Optional[float], car_z: Optional[float],
             car_yaw: Optional[float], tracking_source: str) -> np.ndarray:
        self._refresh_centerline_tune()
        display = self.base_img.copy()
        self._draw_fixed_tags(display)

        if len(self.history) > 1:
            pts = np.array(self.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], isClosed=False, color=(0, 210, 255), thickness=2)

        if car_x is not None and car_z is not None:
            px, py = self.world_to_px(car_x, car_z)
            cv2.circle(display, (px, py), 7, (0, 0, 255), -1)
            if car_yaw is not None:
                heading_len = 28
                # Track map uses a visual arrow only; the detected tag yaw points opposite
                # to the car nose on the physical mounting, so flip it for this window.
                yaw_rad = math.radians(normalize_angle(car_yaw + 180.0))
                end = (
                    int(round(px + math.cos(yaw_rad) * heading_len)),
                    int(round(py - math.sin(yaw_rad) * heading_len)),
                )
                cv2.arrowedLine(display, (px, py), end, (0, 0, 255), 2,
                                cv2.LINE_AA, tipLength=0.35)

        k, p, d = self.get_tuning_values()
        cv2.rectangle(display, (52, 8), (420, 72), (255, 255, 255), -1)
        cv2.putText(display, f"K:{k:.1f} P:{p:.2f} D:{d:.2f}", (62, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 2)
        cv2.putText(display, f"Track:{tracking_source}", (62, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        cv2.putText(display, f"LeftIn:{self.left_curve_inset_px}px", (360, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        self._draw_clear_button(display)
        self._draw_send_button(display)
        self._draw_tune_status(display)
        return display

    def _draw_clear_button(self, display: np.ndarray):
        x1, y1, x2, y2 = self.clear_button_rect
        cv2.rectangle(display, (x1, y1), (x2, y2), (245, 245, 245), -1)
        cv2.rectangle(display, (x1, y1), (x2, y2), (40, 40, 40), 2)
        cv2.putText(display, "Clear Trail", (x1 + 11, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)

    def _draw_send_button(self, display: np.ndarray):
        x1, y1, x2, y2 = self.send_button_rect
        cv2.rectangle(display, (x1, y1), (x2, y2), (235, 250, 235), -1)
        cv2.rectangle(display, (x1, y1), (x2, y2), (35, 120, 35), 2)
        cv2.putText(display, "Save KPD", (x1 + 18, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 90, 20), 2)

    def _draw_tune_status(self, display: np.ndarray):
        text = self.tune_status_text
        color = (90, 90, 90)
        if time.time() < self.tune_status_until:
            color = (0, 130, 0) if text.startswith("Tune: saved") else (0, 0, 200)
        cv2.putText(display, text, (52, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = self.clear_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.clear_history()
            return
        x1, y1, x2, y2 = self.send_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.pending_tune_send = True

    def _draw_fixed_tags(self, display: np.ndarray):
        for tag_id, pos in self.tag_positions.items():
            px, py = self.world_to_px(float(pos[0]), float(pos[2]))
            label_px = max(18, min(px, self.track_w - 19))
            label_py = max(18, min(py, self.track_h - 19))
            cv2.circle(display, (label_px, label_py), 13, (0, 160, 255), -1)
            cv2.circle(display, (label_px, label_py), 13, (255, 255, 255), 2)
            cv2.putText(display, str(tag_id), (label_px - 7, label_py + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    def get_tuning_values(self) -> Tuple[float, float, float]:
        k = cv2.getTrackbarPos("K", self.WINDOW_NAME) / 10.0
        p = cv2.getTrackbarPos("P", self.WINDOW_NAME) / 100.0
        d = cv2.getTrackbarPos("D", self.WINDOW_NAME) / 100.0
        return k, p, d

    def consume_tune_send_requested(self) -> bool:
        if not self.pending_tune_send:
            return False
        self.pending_tune_send = False
        return True

    def mark_tune_sent(self, k: float, p: float, d: float, ok: bool):
        if ok:
            self.tune_status_text = f"Tune: saved K={k:.1f} P={p:.2f} D={d:.2f}, restart SmartCar"
        else:
            self.tune_status_text = "Tune: send failed or disabled"
        self.tune_status_until = time.time() + 3.0

    def show(self, image: np.ndarray):
        cv2.imshow(self.WINDOW_NAME, image)

    def clear_history(self):
        self.history.clear()
        print("[TrackMap] 已清空轨迹")


def run_calibration(cfg: dict):
    print("=== Calibration Mode — 不发 UDP，按 C 打印调试信息 ===")
    tracker = TagTracker(cfg)
    tracker.udp = None

    cap = open_camera()
    print(f"[Homography] 固定标签: {list(cfg['tag_positions'].keys())}")
    print(f"[Homography] 车标签ID: {cfg['car_tag_id']}")
    print(f"[Homography] 需要同时检测到 4 个固定标签（使用各标签中心点）")

    fps_time, fps_count, cur_fps = time.time(), 0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = tracker.detect(gray)
        car_x, car_z, car_yaw = tracker.locate(detections)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count
            fps_count = 0
            fps_time = time.time()

        display = tracker.draw_hud(frame, detections, car_x, car_z, car_yaw, cur_fps)
        
        # 生成鸟瞰图
        bird_eye = tracker.generate_bird_eye_view(frame, detections, car_x, car_z, car_yaw)

        # 同时显示两个窗口
        cv2.imshow("SeeingTag - Original View", display)
        if bird_eye is not None:
            cv2.imshow("SeeingTag - Bird Eye View", bird_eye)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            tracker.print_debug_info(detections)

    cap.release()
    cv2.destroyAllWindows()


def run_competition(cfg: dict):
    """
    比赛模式主循环
    支持三种 H 矩阵更新模式：
        - once: 启动时检测一次，之后不再更新
        - interval: 每 N 帧更新一次
        - every_frame: 每帧都更新
    """
    print("=== Competition Mode — 检测 → Homography映射 → UDP ===")
    print(f"  Unity → {cfg['unity_ip']}:{cfg['unity_port']}")
    
    # 获取 H 矩阵更新模式
    h_mode = cfg.get("homography_update_mode", "interval") 
    h_interval = cfg.get("homography_update_interval", 30)
    bird_eye_only = cfg.get("bird_eye_only", False)
    print(f"  H矩阵更新模式: {h_mode}" + (f" (每{h_interval}帧)" if h_mode == "interval" else ""))
    print(f"  鸟瞰图专用模式: {'ON' if bird_eye_only else 'OFF'}")
    print("  控制: 按 1 开启盲开保护，按 0 关闭盲开保护，点 Clear Trail 或按 T 清空轨迹，点 Save KPD 保存下次启动参数，按 R 重新校准，按 Q 退出")

    tracker = TagTracker(cfg)
    cap = open_camera()
    print(f"[Homography] 固定标签: {list(cfg['tag_positions'].keys())}")
    print(f"[Homography] 车标签ID: {cfg['car_tag_id']}")
    track_map = TrackMapVisualizer(cfg, tracker.tag_positions)
    tuning_udp = TuningUdpSender(
        cfg["tuning_ip"], cfg["tuning_port"], cfg["tuning_udp_enabled"]
    )
    print(
        f"[TuneUDP] {'ON' if cfg['tuning_udp_enabled'] else 'OFF'} "
        f"-> {cfg['tuning_ip']}:{cfg['tuning_port']} "
        "(click Save KPD to save for next SmartCar restart)"
    )

    # 等待 H 矩阵初始化（必须检测到全部 4 个固定标签）
    print(f"[Homography] 等待固定标签检测...")
    while not tracker.homography_valid:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = tracker.detect(gray)
        
        # 检查是否检测到全部 4 个固定标签
        fixed_ids = set(d["id"] for d in detections if d["id"] in tracker.tag_positions)
        fixed_count = len(fixed_ids)
        all_fixed_found = fixed_count == len(tracker.tag_positions)
        
        # 只有检测到全部固定标签才计算 H 矩阵
        if all_fixed_found:
            tracker.compute_homography(detections)
        
        # 显示等待状态（同时绘制检测框）
        display = frame.copy()
        
        # 绘制检测框
        for d in detections:
            color = (0, 255, 0) if d["id"] in tracker.tag_positions else (0, 128, 255)
            pts = d["corners"].astype(np.int32)
            cv2.polylines(display, [pts], True, color, 2)
            cx, cy = int(d['corners'][0][0]), int(d['corners'][0][1])
            label = "CAR" if d["id"] == tracker.car_tag_id else f"ID:{d['id']}"
            cv2.putText(display, label, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 显示等待提示
        status = "OK! Initializing..." if all_fixed_found else f"Waiting... ({fixed_count}/{len(tracker.tag_positions)})"
        cv2.putText(display, status, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("SeeingTag - Original View", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            tuning_udp.close()
            cv2.destroyAllWindows()
            return
    
    print(f"[Homography] 初始化完成！开始追踪车标签...")
    tracker.homography_initialized = True

    fps_time, fps_count, cur_fps = time.time(), 0, 0.0
    frame_no = 0
    # 盲开保护需要手动开启；未开启时，识别丢失只保持上一位置，避免无效阶段乱预测。
    last_position: Optional[Tuple[float, float, float]] = None
    last_detection_time = 0.0
    position_hold_seconds = cfg["position_hold_seconds"]
    motion_history_seconds = cfg["motion_history_seconds"]
    blind_drive_seconds = cfg["blind_drive_seconds"]
    yaw_log_interval_seconds = cfg["yaw_log_interval_seconds"]
    blind_speed_window_seconds = cfg["blind_speed_window_seconds"]
    blind_yaw_rate_window_seconds = cfg["blind_yaw_rate_window_seconds"]
    blind_yaw_rate_scale = cfg["blind_yaw_rate_scale"]
    blind_max_yaw_rate_dps = cfg["blind_max_yaw_rate_dps"]
    blind_yaw_rate_decay_per_second = cfg["blind_yaw_rate_decay_per_second"]
    blind_max_speed_mps = cfg["blind_max_speed_mps"]
    blind_max_distance_m = cfg["blind_max_distance_m"]
    blind_trigger_frames = cfg.get("blind_trigger_frames", 0)
    motion_history: Deque[Tuple[float, float, float, float]] = deque()
    blind_protection_armed = False
    blind_protection_active = False
    blind_keep_armed = False  # True = 按1持久模式，恢复后自动重挂
    lost_frame_count = 0
    blind_started_at: Optional[float] = None
    blind_start_position: Optional[Tuple[float, float]] = None
    last_yaw_log_time = 0.0
    blind_status_message = ""
    blind_status_until = 0.0
    visual_filtered_position: Optional[Tuple[float, float, float]] = None
    visual_filter_alpha = float(np.clip(cfg["filter_alpha"], 0.0, 1.0))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cv2.waitKey(10)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 根据检测模式决定策略
        detections = []
        if bird_eye_only:
            # ---- 鸟瞰图专用模式 ----
            # 仍需定期检测固定标签以更新 H 矩阵
            if h_mode == "every_frame":
                detections = tracker.detect(gray)
                fixed_detections = [d for d in detections if d["id"] in tracker.tag_positions]
                if fixed_detections:
                    tracker.compute_homography(detections, keep_existing=True)
            elif h_mode == "interval" and frame_no % h_interval == 0:
                detections = tracker.detect(gray)
                tracker.compute_homography(detections, keep_existing=True)

            # 跳过原图车检测，直接用鸟瞰图
            car_x, car_z, car_yaw = None, None, None
            bird_warped = tracker.warp_to_bird_eye(frame, 1600, 1280)
            if bird_warped is not None:
                bird_eye_img, transform_info = bird_warped
                car_x, car_z, car_yaw = tracker.locate_car_in_bird_eye(
                    bird_eye_img, transform_info
                )

            tracking_source = "BIRD-EYE"
            position_detected = car_x is not None and car_z is not None
        else:
            # ---- 默认模式：先原图识别，失败再鸟瞰图兜底 ----
            if h_mode == "every_frame":
                # 每帧检测所有标签；只有出现固定标签时才更新 H，
                # 防止固定标签暂时离开画面时把原本有效的 H 清空。
                detections = tracker.detect(gray)
                fixed_detections = [d for d in detections if d["id"] in tracker.tag_positions]
                if fixed_detections:
                    tracker.compute_homography(detections, keep_existing=True)
                car_det = next((d for d in detections if d["id"] == tracker.car_tag_id), None)
                car_x, car_z, car_yaw = tracker.locate_with_cached_H(car_det)
            else:
                # 模式2/3: 检查是否需要更新 H 矩阵
                need_update_H = False

                if h_mode == "once":
                    need_update_H = False
                elif h_mode == "interval":
                    need_update_H = (frame_no % h_interval == 0)

                if need_update_H:
                    detections = tracker.detect(gray)
                    tracker.compute_homography(detections, keep_existing=True)
                    car_det = next((d for d in detections if d["id"] == tracker.car_tag_id), None)
                else:
                    car_det = tracker.detect_car_only(gray)
                    detections = [car_det] if car_det else []

                car_x, car_z, car_yaw = tracker.locate_with_cached_H(car_det)

            tracking_source = "RAW"
            position_detected = car_x is not None and car_z is not None

            # 原图识别失败时，才生成高分辨率鸟瞰图进行第二次检测。
            if not position_detected:
                fallback_warped = tracker.warp_to_bird_eye(frame, 1600, 1280)
                if fallback_warped is not None:
                    fallback_bird_eye, transform_info = fallback_warped
                    car_x, car_z, car_yaw = tracker.locate_car_in_bird_eye(
                        fallback_bird_eye, transform_info
                    )
                    position_detected = car_x is not None and car_z is not None
                    if position_detected:
                        tracking_source = "BIRD-EYE FALLBACK"
                        if tracker.debug_logging:
                            print("[Tracking] 原图未识别到车标签，鸟瞰图兜底识别成功")

        now = time.time()
        # 连续丢帧计数：识别到归零，没识别到累加
        if position_detected:
            lost_frame_count = 0
        else:
            lost_frame_count += 1

        if position_detected:
            if blind_protection_active:
                print("[Blind] 车辆 Tag 已恢复识别，盲开保护自动关闭")
                blind_protection_active = False
                if not blind_keep_armed:
                    blind_protection_armed = False
                blind_started_at = None
                blind_start_position = None
                blind_status_message = "Blind protect auto OFF: tag recovered"
                blind_status_until = now + 2.0
            car_x, car_z, car_yaw = tracker.transform_output_coordinates(car_x, car_z, car_yaw)
            if visual_filtered_position is None:
                visual_filtered_position = (car_x, car_z, car_yaw if car_yaw is not None else 0.0)
            else:
                old_x, old_z, old_yaw = visual_filtered_position
                visual_filtered_position = (
                    old_x + visual_filter_alpha * (car_x - old_x),
                    old_z + visual_filter_alpha * (car_z - old_z),
                    smooth_angle(old_yaw, car_yaw if car_yaw is not None else old_yaw, visual_filter_alpha),
                )
            car_x, car_z, car_yaw = visual_filtered_position
            last_position = (car_x, car_z, car_yaw if car_yaw is not None else 0.0)
            last_detection_time = now
            append_motion_sample(
                motion_history, now, last_position[0], last_position[1],
                last_position[2], motion_history_seconds
            )
        elif blind_protection_armed and lost_frame_count >= blind_trigger_frames:
            if not blind_protection_active:
                print("[Blind] 车辆 Tag 识别丢失，开始按历史运动规律盲开")
                blind_status_message = "Blind driving..."
                blind_status_until = now + 2.0
                blind_protection_active = True
                blind_started_at = now
                if last_position is not None:
                    blind_start_position = (last_position[0], last_position[1])

            blind_elapsed = now - blind_started_at if blind_started_at is not None else 0.0
            blind_position = predict_blind_position(
                motion_history, now, blind_drive_seconds, blind_elapsed,
                blind_speed_window_seconds, blind_yaw_rate_window_seconds,
                blind_yaw_rate_scale, blind_max_yaw_rate_dps,
                blind_yaw_rate_decay_per_second, blind_max_speed_mps
            )
            blind_distance = 0.0
            if blind_position is not None and blind_start_position is not None:
                blind_distance = math.hypot(
                    blind_position[0] - blind_start_position[0],
                    blind_position[1] - blind_start_position[1]
                )

            if blind_position is not None and blind_distance <= blind_max_distance_m:
                car_x, car_z, car_yaw = blind_position
                last_position = blind_position
                visual_filtered_position = blind_position
                tracking_source = "BLIND"
                # 把短时盲开预测位置写回历史，实现连续 dead reckoning。
                append_motion_sample(
                    motion_history, now, blind_position[0], blind_position[1],
                    blind_position[2], motion_history_seconds
                )
            else:
                print("[Blind] 盲开保护达到时间/距离上限，切换为保持当前位置")
                blind_protection_active = False
                blind_protection_armed = False
                blind_started_at = None
                blind_start_position = None
                blind_status_message = "Blind protect limit: OFF"
                blind_status_until = now + 2.0
                if last_position is not None:
                    car_x, car_z, car_yaw = last_position
                    tracking_source = "LOST HOLD"
                else:
                    tracking_source = "LOST"
        elif last_position is not None:
            if blind_protection_active:
                print("[Blind] 盲开保护超时，切换为保持当前位置")
                blind_protection_active = False
                blind_protection_armed = False
                blind_started_at = None
                blind_start_position = None
                blind_status_message = "Blind protect timeout: OFF"
                blind_status_until = now + 2.0
            car_x, car_z, car_yaw = last_position
            tracking_source = "HOLD" if now - last_detection_time <= position_hold_seconds else "LOST HOLD"
        else:
            tracking_source = "LOST"

        # 发送位置
        if car_x is not None and car_z is not None:
            yaw_to_send = car_yaw if car_yaw is not None else 0.0
            tracking_state = tracking_source.lower().replace("-", "_").replace(" ", "_")
            tracker.send_position(car_x, car_z, yaw_to_send, tracking_state)
            if now - last_yaw_log_time >= yaw_log_interval_seconds:
                print(f"[Yaw] {yaw_to_send:.1f}° state={tracking_state}")
                last_yaw_log_time = now
            if tracker.debug_logging and frame_no % 30 == 0:
                x_trunc = truncate_to_2_decimals(car_x)
                z_trunc = truncate_to_2_decimals(car_z)
                yaw_trunc = truncate_to_2_decimals(yaw_to_send)
                print(f"[Send] Car=(X={x_trunc:.2f}, Z={z_trunc:.2f}, Yaw={yaw_trunc:.1f}°)")

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            cur_fps = fps_count
            fps_count = 0
            fps_time = time.time()

        display = tracker.draw_hud(frame, detections, car_x, car_z, car_yaw, cur_fps)
        draw_blind_status_overlay(
            display, blind_protection_armed, blind_protection_active,
            blind_status_message, time.time() < blind_status_until
        )
        
        # 添加模式显示
        mode_text = f"H-Mode: {h_mode}"
        if h_mode == "interval":
            mode_text += f" (N={h_interval})"
        cv2.putText(display, mode_text, (10, display.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Track: {tracking_source}", (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if tracking_source == "RAW" else (0, 255, 255), 1)
        blind_text = "Blind: ALWAYS ON" if blind_keep_armed else ("Blind: ON" if blind_protection_armed else "Blind: OFF")
        blind_color = (0, 255, 255) if blind_protection_armed else (160, 160, 160)
        if blind_keep_armed:
            blind_color = (255, 180, 0)  # 橙色表示持久模式
        cv2.putText(display, blind_text, (10, display.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    blind_color, 1)

        # 生成鸟瞰图
        bird_eye = tracker.generate_bird_eye_view(frame, detections, car_x, car_z, car_yaw)
        if bird_eye is not None:
            draw_blind_status_overlay(
                bird_eye, blind_protection_armed, blind_protection_active,
                blind_status_message, time.time() < blind_status_until
            )

        track_map.append_position(car_x, car_z)
        track_display = track_map.draw(car_x, car_z, car_yaw, tracking_source)
        if track_map.consume_tune_send_requested():
            k, p, d = track_map.get_tuning_values()
            ok = tuning_udp.send_once(k, p, d)
            track_map.mark_tune_sent(k, p, d, ok)

        # 同时显示窗口
        cv2.imshow("SeeingTag - Original View", display)
        if bird_eye is not None:
            cv2.imshow("SeeingTag - Bird Eye View", bird_eye)
        track_map.show(track_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            blind_keep_armed = True
            blind_protection_armed = True
            blind_protection_active = False
            blind_started_at = None
            blind_start_position = None
            blind_status_message = "Blind protect ALWAYS ON"
            blind_status_until = time.time() + 2.0
            print("[Blind] 盲开持久开启：标签恢复后自动重挂")
        elif key == ord('0'):
            blind_keep_armed = False
            blind_protection_armed = False
            blind_protection_active = False
            blind_started_at = None
            blind_start_position = None
            blind_status_message = "Blind protect OFF"
            blind_status_until = time.time() + 2.0
            print("[Blind] 已关闭盲开保护")
        elif key == ord('t'):
            track_map.clear_history()
        elif key == ord('r'):
            # 按 R 键重新校准 H 矩阵
            print("[Recalibrate] 重新检测固定标签...")
            tracker.homography_valid = False
            while not tracker.homography_valid:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = tracker.detect(gray)
                tracker.compute_homography(detections)
                fixed_count = sum(1 for d in detections if d["id"] in tracker.tag_positions)
                display = frame.copy()
                cv2.putText(display, f"Recalibrating... ({fixed_count}/{len(tracker.tag_positions)})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("SeeingTag - Original View", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    tracker.close()
                    tuning_udp.close()
                    cv2.destroyAllWindows()
                    return
            print("[Recalibrate] 完成！")

        frame_no += 1

    cap.release()
    tracker.close()
    tuning_udp.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="SeeingTag ARUCO 智能车定位系统"
    )
    parser.add_argument("--calibrate", "-c", action="store_true", help="校准模式：不发送 UDP")
    parser.add_argument(
        "--set-car-heading", metavar="ANGLE", type=float,
        help="设置车头相对 Tag 默认方向的角度偏移（度），并永久保存到配置文件"
    )
    parser.add_argument("--show-car-heading", action="store_true", help="显示当前永久保存的车头角度偏移后退出")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "tag_config.json")
    if args.set_car_heading is not None:
        try:
            angle = save_car_heading_offset(config_path, args.set_car_heading)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            parser.error(f"无法保存车头角度：{e}")
        print(f"[Config] 车头角度偏移已永久保存为 {angle:g}°")
        return

    cfg = load_config(config_path)
    if args.show_car_heading:
        print(f"[Config] 当前车头角度偏移：{cfg['car_heading_offset_degrees']:g}°")
        return

    mode = "calibration" if args.calibrate else "competition"
    print(f"[Config] 车头角度偏移：{cfg['car_heading_offset_degrees']:g}°")

    if mode == "calibration":
        run_calibration(cfg)
    else:
        run_competition(cfg)


if __name__ == "__main__":
    main()
