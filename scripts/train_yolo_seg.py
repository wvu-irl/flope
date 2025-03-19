from ultralytics import YOLO
from tyro import cli


def main(
    resize_dim: int = 640,
    yolo_model: str = '11n',
    epochs: int = 100
):
    model = YOLO(f"yolo{yolo_model}-seg.pt") # n = nano
    results = model.train(data="config/flower_seg_yolo.yaml", epochs=epochs, imgsz=resize_dim)
    
if __name__=='__main__':
    cli(main)