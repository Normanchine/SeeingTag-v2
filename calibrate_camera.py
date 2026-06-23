"""
SeeingTag — 棋盘格相机标定
从 USB 摄像头实时拍摄棋盘格照片 → 计算内参 K 和畸变系数
棋盘格规格：8×11 内角点（横8 竖11）
按 空格 采集当前帧，按 q 退出并计算标定结果
"""

import cv2
import numpy as np
import json
import sys
import time


# ==================== 配置 ====================
CHESSBOARD_COLS = 8     # 水平内角点数（横8格）
CHESSBOARD_ROWS = 11    # 垂直内角点数（竖11格）
SQUARE_SIZE_MM = 24     # 每个棋盘格的实际边长（毫米），根据你打印的尺寸改

# 棋盘格 3D 角点坐标（z=0 平面）
objp = np.zeros((CHESSBOARD_COLS * CHESSBOARD_ROWS, 3), np.float64)
objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM / 1000.0   # 转成米


# ==================== 主流程 ====================
def main():
    print("=" * 55)
    print("  SeeingTag — 棋盘格相机标定")
    print(f"  棋盘格: {CHESSBOARD_COLS}×{CHESSBOARD_ROWS} 内角点")
    print(f"  方格尺寸: {SQUARE_SIZE_MM}mm")
    print("=" * 55)
    print()
    print("操作说明:")
    print("  空格  — 采集当前帧（棋盘格需完整可见）")
    print("  r     — 重置已采集的图片")
    print("  q     — 完成采集，计算标定结果")
    print()

    # 打开摄像头
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 未找到摄像头!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.5)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] {w}×{h}")
    print()

    objpoints = []   # 世界坐标（3D）
    imgpoints = []   # 图像坐标（2D）
    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 找棋盘格角点
        ret_find, corners = cv2.findChessboardCorners(
            gray,
            (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame.copy()

        if ret_find:
            # 亚像素细化
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            cv2.drawChessboardCorners(display, (CHESSBOARD_COLS, CHESSBOARD_ROWS),
                                      corners_sub, ret_find)
            status = f"棋盘格已检测 ✓  已采集: {collected}"
            color = (0, 255, 0)
        else:
            status = f"未检测到棋盘格  已采集: {collected}"
            color = (0, 0, 255)

        # HUD
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, "空格=采集  r=重置  q=标定", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("SeeingTag - Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and ret_find:
            objpoints.append(objp.copy())
            imgpoints.append(corners_sub.copy())
            collected += 1
            print(f"[采集] 第 {collected} 张 — 保存成功")

        elif key == ord('r'):
            objpoints.clear()
            imgpoints.clear()
            collected = 0
            print("[重置] 已清空所有采集图片")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ==================== 计算标定结果 ====================
    if len(objpoints) < 5:
        print(f"\n[ERROR] 采集太少（{len(objpoints)}张），至少需要 5 张")
        sys.exit(1)

    print(f"\n正在标定（{len(objpoints)} 张图片）...")

    ret_cal, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # 重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    # ==================== 输出结果 ====================
    print()
    print("=" * 55)
    print("  标定结果")
    print("=" * 55)
    print(f"  图片数量:    {len(objpoints)} 张")
    print(f"  分辨率:      {w} × {h}")
    print(f"  重投影误差:  {mean_error:.4f} 像素  {'✓ 优秀' if mean_error < 0.5 else '⚠ 偏大'}")
    print()
    print(f"  相机内参 K:")
    print(f"    fx = {K[0, 0]:.2f}")
    print(f"    fy = {K[1, 1]:.2f}")
    print(f"    cx = {K[0, 2]:.2f}")
    print(f"    cy = {K[1, 2]:.2f}")
    print()
    print(f"  K = {K.tolist()}")
    print()
    print(f"  畸变系数 (k1, k2, p1, p2, k3):")
    print(f"    {dist.flatten().tolist()}")
    print()

    # 据此算实际 FOV
    fov_h = 2 * np.degrees(np.arctan(w / (2 * K[0, 0])))
    fov_v = 2 * np.degrees(np.arctan(h / (2 * K[1, 1])))
    print(f"  等效 FOV:    水平 {fov_h:.1f}°  垂直 {fov_v:.1f}°")
    print()

    # 保存到文件
    save_path = "camera_calib_result.json"
    result = {
        "image_size": [w, h],
        "K": K.tolist(),
        "distortion": dist.flatten().tolist(),
        "reprojection_error_px": round(mean_error, 4),
        "fov_h_deg": round(fov_h, 1),
        "fov_v_deg": round(fov_v, 1)
    }
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  已保存到: {save_path}")
    print("=" * 55)

    # 显示纠正前后的对比
    print("\n按任意键查看畸变校正对比（对比图）...")
    cap2 = cv2.VideoCapture(1 if cap.get(cv2.CAP_PROP_POS_FRAMES) == 0 else 0)
    if cap2.isOpened():
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ret, frame_test = cap2.read()
        if ret:
            h2, w2 = frame_test.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w2, h2), 1, (w2, h2))
            dst = cv2.undistort(frame_test, K, dist, None, newcameramtx)
            x, y, w_roi, h_roi = roi
            dst = dst[y:y + h_roi, x:x + w_roi]

            cv2.imshow("原始 (raw)", frame_test)
            cv2.imshow("去畸变 (undistorted)", dst)
            print("按任意键关闭对比图...")
            cv2.waitKey(0)
        cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
