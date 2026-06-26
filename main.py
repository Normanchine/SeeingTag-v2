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
    blind_drive_seconds = cfg.get("blind_drive_seconds", 0.8)
    position_hold_seconds = cfg.get("position_hold_seconds", 0.35)
    if motion_history_seconds <= 0.1:
        motion_history_seconds = 4.0
    if blind_drive_seconds < 0.0:
        blind_drive_seconds = 0.0
    if position_hold_seconds < 0.0:
        position_hold_seconds = 0.0
    
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
        "homography_update_mode": homography_mode,
        "homography_update_interval": homography_interval,
        "bird_eye_only": cfg.get("bird_eye_only", False),
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
                           timestamp: float, max_blind_seconds: float
                           ) -> Optional[Tuple[float, float, float]]:
    """按最近历史的平均平移速度和 yaw 角速度，短时预测车辆位置。"""
    if len(history) < 2:
        return None

    first_t, first_x, first_z, first_yaw = history[0]
    last_t, last_x, last_z, last_yaw = history[-1]
    lost_seconds = timestamp - last_t
    if lost_seconds < 0.0 or lost_seconds > max_blind_seconds:
        return None

    history_seconds = last_t - first_t
    if history_seconds < 0.08:
        return None

    vx = (last_x - first_x) / history_seconds
    vz = (last_z - first_z) / history_seconds
    yaw_rate = normalize_angle(last_yaw - first_yaw) / history_seconds
    return (
        last_x + vx * lost_seconds,
        last_z + vz * lost_seconds,
        normalize_angle(last_yaw + yaw_rate * lost_seconds),
    )


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

    tracker = TagTracker(cfg)
    cap = open_camera()
    print(f"[Homography] 固定标签: {list(cfg['tag_positions'].keys())}")
    print(f"[Homography] 车标签ID: {cfg['car_tag_id']}")
    
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
            cv2.destroyAllWindows()
            return
    
    print(f"[Homography] 初始化完成！开始追踪车标签...")
    tracker.homography_initialized = True

    fps_time, fps_count, cur_fps = time.time(), 0, 0.0
    frame_no = 0
    # 原图与鸟瞰图都短暂丢失时，先按近期运动趋势短时外推，再退回保留上一位置。
    last_position: Optional[Tuple[float, float, float]] = None
    last_detection_time = 0.0
    position_hold_seconds = cfg["position_hold_seconds"]
    motion_history_seconds = cfg["motion_history_seconds"]
    blind_drive_seconds = cfg["blind_drive_seconds"]
    motion_history: Deque[Tuple[float, float, float, float]] = deque()
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
        if position_detected:
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
        elif (blind_position := predict_blind_position(motion_history, now, blind_drive_seconds)) is not None:
            car_x, car_z, car_yaw = blind_position
            last_position = blind_position
            tracking_source = "BLIND"
        elif last_position is not None and now - last_detection_time <= position_hold_seconds:
            car_x, car_z, car_yaw = last_position
            tracking_source = "HOLD"
        else:
            tracking_source = "LOST"

        # 发送位置
        if car_x is not None and car_z is not None:
            yaw_to_send = car_yaw if car_yaw is not None else 0.0
            tracking_state = tracking_source.lower().replace("-", "_").replace(" ", "_")
            tracker.send_position(car_x, car_z, yaw_to_send, tracking_state)
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
        
        # 添加模式显示
        mode_text = f"H-Mode: {h_mode}"
        if h_mode == "interval":
            mode_text += f" (N={h_interval})"
        cv2.putText(display, mode_text, (10, display.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Track: {tracking_source}", (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if tracking_source == "RAW" else (0, 255, 255), 1)
        
        # 生成鸟瞰图
        bird_eye = tracker.generate_bird_eye_view(frame, detections, car_x, car_z, car_yaw)

        # 同时显示两个窗口
        cv2.imshow("SeeingTag - Original View", display)
        if bird_eye is not None:
            cv2.imshow("SeeingTag - Bird Eye View", bird_eye)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
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
                    cv2.destroyAllWindows()
                    return
            print("[Recalibrate] 完成！")

        frame_no += 1

    cap.release()
    tracker.close()
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
