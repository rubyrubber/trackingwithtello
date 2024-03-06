import cv2
import os

from yolo.detect import DetectAPI



video_capture = cv2.VideoCapture(0)
detect_api = DetectAPI(exist_ok=True)

while True:
    k = cv2.waitKey(1)
    ret, frame = video_capture.read()

    path = 'D:/Desktop/tello/my_first_project/testimgoutput'
    cv2.imwrite(os.path.join(path, 'test.jpg'), frame)

    label = detect_api.run()
    print(str(label))

    image = cv2.imread('D:/Desktop/tello/my_first_project/testimgoutput/test.jpg', flags=1)
    cv2.imshow("video", image)

    if k == 27:  # 按下ESC退出窗口
        break  # 确保break语句在循环内

video_capture.release()
cv2.destroyAllWindows()  # 退出循环后关闭窗口
