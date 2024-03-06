import cv2
import signal
import argparse
import numpy as np

import sys
sys.path.append("D:/Desktop/tello2/my_first_project")
# sys.path.append("C:/users/ruby/.conda/envs/pytorch/Lib/site-packages/yolo_detectAPI")
from yolo_detectAPI import DetectAPI
from utils.telloconnect import TelloConnect
from utils.kcftracker import KCFTracker
import torch


# 输入变量
parser = argparse.ArgumentParser(description='trial\n')
parser.add_argument(
    '-vsize',
    type=list,
    help='Video size received from tello',
    default=(920, 720)
)
parser.add_argument(
    '-debug',
    type=bool,
    help='Enable debug, lists messages in console',
    default=False
)
parser.add_argument(
    '-video',
    type=str,
    help='Use as inputs a video file, no tello needed, debug must be True',
    default=""
)

args = parser.parse_args()

# 设置图片大小,后期可以通过参数设置改变它的值
imgsize = args.vsize
# 保存视频，被用作一个标志（flag），
# 用于控制是否将处理后的视频帧保存到文件中
writevideo = False
pspeed = 10
# 在主循环中添加一个标志来跟踪是否检测到了物体
object_detected = False
expected_shape = (480, 640, 3)  # 期望的图像形状

# 初始化 YOLO 模型
yolo_model = DetectAPI(weights='D:/Desktop/tello2/my_first_project/yolov5s.pt')


# 信号处理器
def signal_handler(sig, frame):
    raise Exception


# 捕捉信号
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# 在主循环开始前打印消息
# print("loading mp4 file...")

# 连接无人机
if args.debug and args.video is not None:
    print("调试模式已启用，并且指定了视频文件路径。")
    tello = TelloConnect(DEBUG=True, VIDEO_SOURCE=args.video)
else:
    print("调试模式未启用或未指定视频文件路径。")
    tello = TelloConnect(DEBUG=False)

# 在视频加载完成后打印消息
print("mp4 file loaded")

# 写入视频
videow = cv2.VideoWriter(
    'out.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    30, (imgsize)
)

# 检查视频写入器是否成功打开
if videow.isOpened():
    print("视频写入器已成功打开。")
else:
    print("无法打开视频写入器。")

# 周期性检查WiFi状态
tello.add_periodic_event('wifi?', 40, 'WiFi')

# 等待无人机连接
tello.wait_till_connected()
tello.start_communication()
tello.start_video()

# 初始化KCF跟踪器
tracker_initialized = False
kcf_tracker = KCFTracker(hog=True, fixed_window=True, multiscale=True)


# 激动人心的主循环！！！
# 主循环
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # 按 'ESC' 键退出
        break

    try:
        img = tello.get_frame()  # 从无人机获取视频帧

        if img is None:  # 如果没有获取到图像，则跳过当前迭代
            continue

        # 使用YOLO模型进行人脸检测
        with torch.no_grad():
            result_with_names = yolo_model.detect([img])
            result = result_with_names[0]  # 检测结果
            names = result_with_names[1]  # 检测到的类别名称
            # print("names 的值：", names)
            # print("要显示的图像尺寸：", img.shape)

        # 复制原始图像，用于绘制边界框
        img_with_boxes = img.copy()

        # 标记是否在当前帧中检测到人脸
        face_detected = False

        # 将识别结果写入控制台
        for idx, (img_result, name_result) in enumerate(result):
            print(f"图像 {idx + 1} 中检测到的物体数量：{len(name_result)}")
            for cls, (x1, y1, x2, y2), conf in name_result:
                print(f"物体类别: {names[cls]}, 位置: 左上角({x1}, {y1}), 右下角({x2}, {y2}), 置信度: {conf}")
                if names[cls] == "person":  # 假设YOLO模型识别人脸为"person"类
                    face_detected = True
                    x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))

                    # 如果KCF跟踪器未初始化或需要重新初始化
                    if not tracker_initialized or not face_detected:
                        kcf_tracker.init([x1, y1, x2-x1, y2-y1], img)
                        tracker_initialized = True

                    # 绘制由YOLO检测到的边界框
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{names[cls]}: {conf:.2f}"
                    cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 如果当前帧未检测到人脸，但跟踪器之前已初始化，则继续跟踪上一次检测到的位置
        if not face_detected and tracker_initialized:
            bbox = kcf_tracker.update(img)
            x1, y1, w, h = list(map(int, bbox))
            x2, y2 = x1 + w, y1 + h

            # 绘制KCF跟踪器跟踪到的边界框
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, "Tracking", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("Detection Result", img_with_boxes)

        # 如果开启了视频写入，则将帧写入视频文件
        if writevideo:
            videow.write(img_with_boxes)

    except Exception as e:
        print("An exception occurred:", e)
        break

# 退出前关闭所有资源
tello.stop_video()
tello.stop_communication()
cv2.destroyAllWindows()




