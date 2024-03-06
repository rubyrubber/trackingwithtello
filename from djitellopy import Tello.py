from djitellopy import Tello
import socket

# 设置服务端和客户端地址和端口
tello_address = ('192.168.10.1', 8889)
me_address = ('0.0.0.0', 8890)

# 创建UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 绑定服务器地址和端口
sock.bind(me_address)


def send_command(command):
    # 发送命令到无人机
    sock.sendto(command.encode('utf-8'), tello_address)


def receive_response():
    # 接收无人机响应
    response, _ = sock.recvfrom(1024)
    return response.decode('utf-8')


# 连接到无人机
t = Tello()
t.state_thread = False
t.connect(False)

# 发送命令“command”，进入SDK模式
send_command("command")

# 接收无人机响应
response = receive_response()
print(response)

# 获取电池状态
send_command("battery?")
battery_response = receive_response()
print(f"battery level:{battery_response}")

# 关闭套接字
sock.close
