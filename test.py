import cv2
import signal
import argparse
import numpy as np

import sys
sys.path.append("D:/Desktop/tello2/my_first_project")
# sys.path.append("C:/users/ruby/.conda/envs/pytorch/Lib/site-packages/yolo_detectAPI")
from yolo_detectAPI import DetectAPI
from utils.telloconnect import TelloConnect
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
print("loading mp4 file...")

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

# 激动人心的主循环！！！
while True:
    k = cv2.waitKey(1)

    try:
        img = tello.get_frame()

        print("图像类型:", type(img))
        print("图像形状:", img.shape)

        if img is not None:
            print("成功获取图像。")
        else:
            print("未能获取图像。")

        # 等待关键帧
        if img is None:
            continue

        # 使用 YOLO 模型对图像进行识别
        with torch.no_grad():
            # 使用 TelloConnect 中的视频源进行检测
            print("开始调用 yolo_model.detect() 函数...")
            result_with_names = yolo_model.detect([img])
            result = result_with_names[0]
            names = result_with_names[1]
            print("yolo_model.detect() 函数执行完毕.")
            # print("detect() 方法返回的结果：", result)
            print("names 的值：", names)

        print("result 的类型:", type(result))
        print("result 的长度:", len(result))
        if len(result) > 0:
            print("result[0] 的类型:", type(result[0]))
            print("result[0] 的长度:", len(result[0]))

        print("要显示的图像尺寸：", img.shape)

        # 将识别结果写入控制台
        for idx, (img_result, name_result) in enumerate(result):
            print(f"图像 {idx + 1} 中检测到的物体数量：{len(name_result)}")
            for cls, (x1, y1, x2, y2), conf in name_result:
                print(f"物体类别: {names[cls]}, 位置: 左上角({x1}, {y1}), 右下角({x2}, {y2}), 置信度: {conf}")

            # 在检测到物体时设置标志以开始写入视频
            object_detected = True

            # 将图像写入视频
            if writevideo:
                videow.write(img)

            # 绘制检测到的物体框
            # 在检测到物体之前，复制原图像
            img_with_boxes = img.copy()

            if len(result) > 0:
                object_detected = True  # 检测到物体，设置标志
                for idx, (img_result, name_result) in enumerate(result):
                    for cls, (x1, y1, x2, y2), conf in name_result:
                        print(f"Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                        # 确保坐标是整数且在图像范围内
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        # 绘制矩形框和标签
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"{names[cls]}: {conf:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(img_with_boxes, (x1, y1 - text_height - baseline), (x1 + text_width, y1),
                                      (255, 0, 0), thickness=cv2.FILLED)
                        cv2.putText(img_with_boxes, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 2)
            else:
                # 如果没有检测到物体，则使用原图
                img_with_boxes = img

            # 检查img_with_boxes的形状和类型，确保它与原始图像一致
            print("Modified image shape:", img_with_boxes.shape)
            print("Modified image dtype:", img_with_boxes.dtype)

            # 显示图像
            cv2.imshow("Detection Result", img_with_boxes)


        if cv2.waitKey(1) == ord('q'):
            break

        writevideo = True

    except Exception as e:
        print("An exception occurred:", e)
        tello.stop_video()
        tello.stop_communication()
        break

cv2.destroyAllWindows()



