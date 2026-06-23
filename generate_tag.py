"""生成比赛用的 ARUCO 标签（带白边防干扰）"""

import cv2
import cv2.aruco as aruco

# ── 配置 ──
TAG_IDS = [1, 2, 3, 4, 10]   # 1-4 固定参考点，10 车上
TAG_SIZE = 400                  # 内部二维码像素
BORDER = 100                    # 白边像素
DICT = aruco.DICT_4X4_50        # 与检测端一致
# ────────
dictionary = aruco.getPredefinedDictionary(DICT)

def generate_tag_with_border(tag_id, size=TAG_SIZE, border=BORDER):
    marker = aruco.generateImageMarker(dictionary, tag_id, size)
    padded = cv2.copyMakeBorder(marker, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded

if __name__ == "__main__":
    for tag_id in TAG_IDS:
        img = generate_tag_with_border(tag_id)
        filename = f"tag_id{tag_id}.png"
        cv2.imwrite(filename, img)
        print(f"✓ 生成: {filename}")

    print(f"\n✔ 完成！共生成 {len(TAG_IDS)} 个标签")

