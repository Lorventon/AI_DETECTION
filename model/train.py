# import torch
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")
# Запуск обучения
result = model.train(
    data='D:/Projects/HackInHome2024/model/dataset/data.yaml',       # путь к файлу конфигурации датасета
    epochs=25,             # количество эпох обучения
    imgsz=640,              # размер входного изображения
    batch=8,               # размер батча
    name='car_segmentation',# имя эксперимента
    cache=True              # кэширование данных для ускорения
)
