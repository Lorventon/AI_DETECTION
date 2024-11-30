from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# Запуск обучения
result = model.train(
    data='D:/Projects/HackInHome2024/dataset/data.yaml',       # путь к файлу конфигурации датасета
    epochs=250,             # количество эпох обучения
    imgsz=640,              # размер входного изображения
    batch=16,               # размер батча
    name='car_segmentation',# имя эксперимента
    cache=True              # кэширование данных для ускорения
)

