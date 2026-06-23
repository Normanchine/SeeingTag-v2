"""
test_ipm.py — 逆透视映射 (Inverse Perspective Mapping) 测试
用固定标签的角点 + 已知世界坐标 → 单应矩阵 H
直接把车标签像素坐标映射到世界平面 XZ，跳过 PnP 和相机标定
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time


def load_config(path: str):
    with open(path, 'r') as f:
        cfg = json.load(f)
    tag_positions = {}
    for tid, pos in cfg["tag_world_positions"].items():
        tag_positions[int(tid)] = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
    return {
        "tag_size_m": cfg["tag_size_m"],
        "tag_positions": tag_positions,
        "car_tag_id": cfg["car_tag_id"],
    }


def build_tag_corners_world(tag_center: np.ndarray, tag_size: float):
    """
    返回标签4个角点在世界 XZ 平面上的坐标 (x, z)
    标签在 y=0 平面，角点扣除 y 分量
    """
    s2 = tag_size / 2
    local = np.array([
        [-s2, -s2, 0],   # 角点0
        [ s2, -s2, 0],   # 角点1
        [ s2,  s2, 0],   # 角点2
        [-s2,  s2, 0],   # 角点3
    ])
    corners_3d = tag_center + local  # (4, 3)
    return corners_3d[:, [0, 2]]     # 只保留 (x, z)


def main():
    cfg = load_config("tag_config.json")
    tag_size = cfg["tag_size_m"]
    tag_positions = cfg["tag_positions"]
    car_tag_id = cfg["car_tag_id"]

    # 初始化 ArUco 检测器
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    # 打开摄像头
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 未找到摄像头")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.5)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] {w}x{h}")
    print(f"[IPM] 固定标签: {list(tag_positions.keys())}")
    print(f"[IPM] 车标签ID: {car_tag_id}")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # 收集所有固定标签的角点对应关系
        src_pts = []  # 像素坐标 (u, v)
        dst_pts = []  # 世界坐标 (x, z)
        car_center_pixel = None
        fixed_detected = {}

        if ids is not None:
            for i, tag_id in enumerate(ids.flatten()):
                c = corners[i][0]  # (4, 2)
                cx_px = int(np.mean(c[:, 0]))
                cy_px = int(np.mean(c[:, 1]))

                if tag_id in tag_positions:
                    # 固定标签 — 加入单应矩阵对应点集
                    world_corners_xz = build_tag_corners_world(
                        tag_positions[tag_id], tag_size
                    )
                    for j in range(4):
                        src_pts.append(c[j])
                        dst_pts.append(world_corners_xz[j])
                    fixed_detected[tag_id] = (cx_px, cy_px)

                    # 画固定标签：绿色
                    pts = c.astype(np.int32)
                    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                    cv2.putText(display, f"ID{tag_id}", (cx_px, cy_px - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif tag_id == car_tag_id:
                    # 车标签 — 记录中心像素
                    center = np.mean(c, axis=0)  # (2,)
                    car_center_pixel = center
                    # 画车标签：蓝色
                    pts = c.astype(np.int32)
                    cv2.polylines(display, [pts], True, (255, 128, 0), 3)
                    cv2.putText(display, f"CAR(ID{tag_id})", (cx_px, cy_px - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

        # --- 单应矩阵计算 ---
        car_world = None
        homography_ok = False

        if len(src_pts) >= 8:  # 至少 2 个标签 = 8 个角点
            src_pts = np.array(src_pts, dtype=np.float64)
            dst_pts = np.array(dst_pts, dtype=np.float64)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            if H is not None:
                homography_ok = True
                inlier_count = int(np.sum(mask))

                # 如果检测到车标签，用单应映射到世界坐标
                if car_center_pixel is not None:
                    pixel_pt = np.array([[car_center_pixel[0], car_center_pixel[1]]], dtype=np.float64)
                    world_pt = cv2.perspectiveTransform(pixel_pt.reshape(1, 1, 2), H)
                    wx, wz = world_pt[0, 0, 0], world_pt[0, 0, 1]
                    car_world = np.array([wx, 0.16, wz])

        # --- HUD ---
        info_lines = []
        if car_world is not None:
            info_lines.append(f"Car World: X={car_world[0]:.3f}  Z={car_world[2]:.3f}")
        if homography_ok:
            info_lines.append(f"Homography: {len(src_pts)}pts / {inlier_count} inliers")
        else:
            info_lines.append(f"H: need >=2 fixed tags (have {len(src_pts)//4})")
        info_lines.append(f"Fixed tags: {list(fixed_detected.keys())}")

        for idx, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 30 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 画世界坐标俯视图（小地图） ---
        # 右下角叠一个 300x200 的俯视图
        map_scale = 40  # 1m = 40px
        map_origin_x = w - 320
        map_origin_y = h - 220
        map_w, map_h = 300, 200

        # 画背景
        cv2.rectangle(display, (map_origin_x, map_origin_y),
                      (map_origin_x + map_w, map_origin_y + map_h), (30, 30, 30), -1)
        cv2.rectangle(display, (map_origin_x, map_origin_y),
                      (map_origin_x + map_w, map_origin_y + map_h), (100, 100, 100), 1)

        def world_to_map(wx, wz):
            """世界坐标 → 小地图像素坐标"""
            mx = map_origin_x + 20 + wx * map_scale
            my = map_origin_y + map_h - 20 - wz * map_scale
            return int(mx), int(my)

        # 画固定标签位置
        for tid, pos in tag_positions.items():
            mx, my = world_to_map(pos[0], pos[2])
            cv2.circle(display, (mx, my), 5, (0, 255, 0), -1)
            cv2.putText(display, str(tid), (mx + 5, my + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 画赛道边框
        corners_world = [
            (0, 0), (5, 0), (5, 4), (0, 4), (0, 0)
        ]
        for i in range(4):
            x1, y1 = world_to_map(*corners_world[i])
            x2, y2 = world_to_map(*corners_world[i + 1])
            cv2.line(display, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # 画车位置
        if car_world is not None:
            mx, my = world_to_map(car_world[0], car_world[2])
            cv2.circle(display, (mx, my), 6, (0, 255, 255), -1)
            cv2.putText(display, "CAR", (mx + 8, my + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("SeeingTag - IPM Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(f"\n--- IPM ---")
            print(f"  固定标签: {list(fixed_detected.keys())}")
            print(f"  内点数: {inlier_count if homography_ok else 0}/{len(src_pts)}")
            if car_world is not None:
                print(f"  车世界坐标: ({car_world[0]:.3f}, {car_world[1]:.3f}, {car_world[2]:.3f})")
            if car_center_pixel is not None:
                print(f"  车像素中心: ({car_center_pixel[0]:.1f}, {car_center_pixel[1]:.1f})")
            print("------------------------\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
