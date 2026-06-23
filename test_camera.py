import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0) # 如果黑屏，改回 0

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

print("正在启动摄像头...")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            # 提取这个码的四个角点坐标
            c = corners[i][0]
            
            # 计算这个码在画面里的像素宽度
            width = abs(c[0][0] - c[1][0])
            # 增加一个调试打印：看看它到底看到了什么鬼东西，多大尺寸
            print(f" 扫描到可疑目标: ID {ids[i][0]}, 宽度 {int(width)} 像素")
            
            # 核心过滤逻辑：只有宽度大于 40 个像素，我们才认为它是真正的 Tag 码！
            # 小于 40 像素的全部当作环境噪点扔掉
            if width > 15: 
                pts = c.astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
                cv2.putText(frame, f"ID:{ids[i][0]}", (pts[0][0], pts[0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                print(f"真实检测到的大码 ID: {ids[i][0]}")

    cv2.imshow("SmartCar AR Vision - Filtered", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()