# import torch
from ultralytics import YOLO
<<<<<<< Updated upstream
model = YOLO("yolo11n-seg.pt")
=======

model = YOLO("yolo11s-seg.pt")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
>>>>>>> Stashed changes

# Запуск обучения
result = model.train(
    data='D:/Projects/HackInHome2024/model/dataset/data.yaml',       # путь к файлу конфигурации датасета
    epochs=25,             # количество эпох обучения
    imgsz=640,              # размер входного изображения
    batch=8,               # размер батча
    name='car_segmentation',# имя эксперимента
<<<<<<< Updated upstream
    cache=True              # кэширование данных для ускорения
)
=======
    cache=True,            # кэширование данных для ускорения
)

>>>>>>> Stashed changes
