import socket
import json
import time
import math

TARGET_IP = "10.61.148.125"
TARGET_PORT = 9005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"发送测试数据到 {TARGET_IP}:{TARGET_PORT}")
print("Z保持1，X和Y持续变化...")
print("按 Ctrl+C 停止")
print()

seq = 0
try:
    while True:
        t = time.time()

        x = math.sin(t * 0.5) * 2.0
        y = math.cos(t * 0.3) * 1.0
        z = 1.0

        data = {
            "type": "robot_position",
            "pos": [float(x), float(y), float(z)],
            "euler": [0.0, 0.0, 0.0],
            "seq": seq,
            "timestamp": t
        }

        json_str = json.dumps(data)
        sock.sendto(json_str.encode('utf-8'), (TARGET_IP, TARGET_PORT))

        print(f"发送: X={x:.3f} Y={y:.3f} Z={z:.3f}")

        seq += 1
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n停止测试")
    sock.close()
