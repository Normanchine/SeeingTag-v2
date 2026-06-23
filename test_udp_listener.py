import socket
import json

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('0.0.0.0', 9004))
print('监听9004端口，等待数据...')

while True:
    data, addr = s.recvfrom(2048)
    print(f'收到 from {addr}:')
    try:
        json_data = json.loads(data.decode())
        print(json.dumps(json_data, indent=2))
    except:
        print(data.decode())
