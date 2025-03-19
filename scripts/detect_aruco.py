import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

frame = cv2.imread('/home/rashik_shrestha/ws/sunflower/output/colors/color3.jpg')
frame = cv2.resize(frame, (640, 480))
cv2.imwrite('/home/rashik_shrestha/ws/sunflower/output/colors/color3_small.jpg', frame)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# Detect ArUco markers
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# If markers are detected, draw them on the frame
if ids is not None:
    aruco.drawDetectedMarkers(frame, corners, ids)
    for i, corner in enumerate(corners):
        # Example: print the ID and corner coordinates
        print(f"Marker ID: {ids[i][0]} at corners: {corner}")

# Display the frame with detected markers
cv2.imshow('ArUco Marker Detection', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
