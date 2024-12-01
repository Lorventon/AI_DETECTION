import json
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def calculate_paint_cost(masks, classes, class_names, front_door_area, torch_width, torch_extrusion, paint_cost_per_liter, img):
    # Список для хранения данных по элементам
    elements = []

    for i in range(masks.shape[0]):
        # Получение маски и класса
        mask = masks[i].cpu()
        class_index = int(classes[i])
        class_name = class_names[class_index]

        # Изменение размера маски до размера изображения
        mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Использование PyTorch для подсчета площади
        area_pixels = torch.sum(mask).item()  # Суммируем элементы тензора и конвертируем в число

        # Преобразуем площадь из пикселей в квадратные метры (с использованием эталонной площади)
        scale_factor = front_door_area / torch.sum(masks[0]).item()  # Эталонная передняя дверь
        area_m2 = area_pixels * scale_factor  # Площадь в квадратных метрах

        # Рассчитываем площадь обработки для ЛКМ
        contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True)  # Периметр элемента

        # Площадь обработки для ЛКМ
        processing_area = area_m2 * (1 + torch_width) * (1 + torch_extrusion)

        # Стоимость ЛКМ для данного элемента
        paint_required = processing_area / 10  # Примерная площадь покрытия 1 литра ЛКМ
        paint_cost = paint_required * paint_cost_per_liter  # Стоимость ЛКМ

        # Сохранение результатов
        elements.append({
            "element": class_name,
            "physical_area_m2": round(area_m2, 2),
            "processing_area_m2": round(processing_area, 2),
            "paint_cost": round(paint_cost, 2)
        })

    # Преобразование в JSON-формат
    return json.dumps(elements, indent=4, ensure_ascii=False)

def estimate_paint_costs():
    model = YOLO('../runs/segment/car_segmentation2/weights/best.pt')

    img = cv2.imread('../dataset/train/images/' + input("Введите название файла фотографии модели в формате (photo.jpg): "))

    # Параметры, вводимые пользователем
    front_door_area = float(input("Введите физический размер передней двери: "))  # Эталонная площадь передней двери (м²)
    torch_width = float(input("Введите ширину факела: "))  # Ширина факела (м)
    torch_extrusion = float(input("Введите вылет факела за границы элемента при одном проходе: "))  # Вылет факела за границу элемента (м)
    paint_cost_per_liter = float(input("Введите стоимость 1 литра ЛКМ: "))  # Стоимость 1 литра ЛКМ (в условных единицах)

    # Получение результатов сегментации
    results = model(img, imgsz=640, iou=0.4, conf=0.8, verbose=True)

    # Извлечение масок и классов
    masks = results[0].masks.data
    classes = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names

    # Пример вызова функции
    return calculate_paint_cost(
        masks=masks,
        classes=classes,
        class_names=class_names,
        front_door_area=front_door_area,
        torch_width=torch_width,
        torch_extrusion=torch_extrusion,
        paint_cost_per_liter=paint_cost_per_liter,
        img=img
    )

print(estimate_paint_costs())

