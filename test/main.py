from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

# Запуск обучения
model.train(
    data='data.yaml',       # путь к файлу конфигурации датасета
    epochs=100,             # количество эпох обучения
    imgsz=640,              # размер входного изображения
    batch=16,               # размер батча
    name='car_segmentation',# имя эксперимента
    cache=True              # кэширование данных для ускорения
)

