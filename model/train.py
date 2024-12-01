# import torch
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")

result = model.train(
    data='D:/Projects/HackInHome2024/model/dataset/data.yaml',
    epochs=250,             
    imgsz=640,              
    batch=16,               
    name='car_segmentation',
    cache=True              
)
