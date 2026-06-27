"""
Interactive track calibration tool
Saves a mapping from world coordinates (meters) -> image pixels as a 3x3 homography in track_calib.json.

Usage:
  python track_calibrate.py --image path/to/track_image.png

Instructions (in window):
  - Click 4 points on the image in this order:
      1) world (0,0)   - left-top
      2) world (W,0)   - right-top
      3) world (W,H)   - right-bottom
      4) world (0,H)   - left-bottom
    (W,H default from tag_config.json: field_width_m, field_height_m; defaults 5x4 m)
  - Keys:
      s = save homography and exit
      u = undo last click
      r = reset
      q = quit without saving

Output:
  position/track_calib.json contains:
    { "H": [[...]], "field_width_m": W, "field_height_m": H, "image": "<path>" }

"""

import cv2
import numpy as np
import json
import os
import argparse


def load_field_dims(config_path: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return float(cfg.get('field_width_m', 5.0)), float(cfg.get('field_height_m', 4.0))
    except Exception:
        return 5.0, 4.0


def main():
    parser = argparse.ArgumentParser(description='Interactive track calibration (world→image homography)')
    parser.add_argument('--image', '-i', default=None,
                        help='Path to track image (default: track_map_clean.png)')
    parser.add_argument('--config', '-c', default=os.path.join(os.path.dirname(__file__), 'tag_config.json'),
                        help='Path to tag_config.json to read field size')
    parser.add_argument('--out', '-o', default=os.path.join(os.path.dirname(__file__), 'track_calib.json'),
                        help='Output JSON path')
    args = parser.parse_args()

    # find image: prefer provided --image; else use the cleaned map in this folder
    if args.image is None:
        cleaned = os.path.join(os.path.dirname(__file__), 'track_map_clean.png')
        source = os.path.join(os.path.dirname(__file__), 'track_map_source.png')
        old_default = os.path.join(os.path.dirname(__file__), 'image.png')
        if os.path.exists(cleaned):
            img_path = cleaned
        elif os.path.exists(source):
            img_path = source
        else:
            img_path = old_default
    else:
        img_path = args.image

    if not os.path.exists(img_path):
        print(f'[WARN] image not found: {img_path}. Creating blank canvas.')
        img = np.zeros((800, 800, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img_path)

    field_w, field_h = load_field_dims(args.config)
    print(f'[Info] Field size: {field_w}m x {field_h}m')

    display = img.copy()
    pts = []  # clicked pixel points

    def draw_overlay():
        d = display.copy()
        # draw instructions
        cv2.putText(d, 'Click 4 points: (0,0),(W,0),(W,H),(0,H)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(d, "s:save  u:undo  r:reset  q:quit", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        for i, p in enumerate(pts):
            cv2.circle(d, tuple(p), 6, (0, 255, 0), -1)
            cv2.putText(d, str(i+1), (p[0]+8, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                cv2.line(d, tuple(pts[i-1]), tuple(pts[i]), (0,200,200), 2)
        return d

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 4:
                pts.append([int(x), int(y)])
            else:
                print('[Info] Already 4 points. Press u to undo or r to reset.')

    win = 'Track Map - Calibrate'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = draw_overlay()
        cv2.imshow(win, disp)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print('Quit without saving.')
            break
        elif key == ord('u'):
            if pts:
                pts.pop()
        elif key == ord('r'):
            pts = []
        elif key == ord('s'):
            if len(pts) != 4:
                print('[Error] Need 4 points to compute homography. Current:', len(pts))
                continue
            # compute homography from world (meters) -> pixel
            world_pts = np.array([
                [0.0, 0.0],
                [field_w, 0.0],
                [field_w, field_h],
                [0.0, field_h]
            ], dtype=np.float32)
            pixel_pts = np.array(pts, dtype=np.float32)
            try:
                H = cv2.getPerspectiveTransform(world_pts, pixel_pts)
            except Exception as e:
                print('[Error] getPerspectiveTransform failed:', e)
                continue
            H_list = H.tolist()
            out = {
                'H': H_list,
                'field_width_m': float(field_w),
                'field_height_m': float(field_h),
                'image': os.path.abspath(img_path)
            }
            try:
                with open(args.out, 'w', encoding='utf-8') as f:
                    json.dump(out, f, indent=2)
                print(f'[Saved] Calibration saved to {args.out}')
            except Exception as e:
                print('[Error] Failed to save:', e)
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
