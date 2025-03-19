from ultralytics import YOLO
from pathlib import Path
import os
import cv2

input_dir = Path("/home/rashik_shrestha/ws/sunflower/output/random")
output_dir = Path("/home/rashik_shrestha/ws/sunflower/output/yolo_detect")

images = os.listdir(input_dir)

model = YOLO("/home/rashik_shrestha/ws/sunflower/runs/detect/train/weights/best.pt")  # pretrained YOLO11n model

for iname in images:
    img = cv2.imread(input_dir/iname)
    img = cv2.resize(img, (1920, 1080))
    print(iname)
    print(img.shape)
    result = model(img, verbose=False)
    print(result[0].probs)
    # print(len(result))
    # print(type(result))
    result[0].save(filename=output_dir/iname)  # save to disk