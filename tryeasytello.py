from easytello import tello
import cv2

my_drone = tello.Tello()

videow = cv2.VideoWriter(
    'out.avi',
    cv2.VideoWriter_fourcc('X', '2', '6', '4'),
    30, (640, 480)
    )

my_drone.streamon()
