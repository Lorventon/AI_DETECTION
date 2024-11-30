import os
import cv2
import numpy as np
from pathlib import Path

# Папки с изображениями и масками
IMAGE_FOLDER = 'dataset/images/train'
MASK_FOLDER = 'dataset/masks'
LABEL_FOLDER = 'dataset/labels/train'

os.makedirs(LABEL_FOLDER, exist_ok=True)

COLOR_TO_CLASS = {
    (0, 153, 150): 0,      # bumper
    (178, 188, 49): 1,     # fog_lights
    (189, 188, 186): 2,    # radiator
    (183, 211, 134): 3,    # license_plate
    (231, 3, 121): 4,      # emblem
    (201, 171, 211): 5,    # hood
    (220, 195, 225): 6,    # fender
    (155, 216, 221): 7,    # windshield
    (137, 79, 116): 8,     # roof
    (248, 173, 192): 9,    # back_fender
    (235, 92, 139): 10,    # side_mirrors
    (148, 35, 110): 11,    # door_handle
    (44, 58, 105): 12,     # tires
    (21, 6, 84): 13,       # wheel
    (20, 68, 50): 14,      # back_door
    (0, 135, 97): 15       # front_door
}

def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def get_color_range(color, tolerance=10):
    """Функция для получения диапазона цветов с учётом допуска."""
    lower = np.array([max(0, c - tolerance) for c in color], dtype=np.uint8)
    upper = np.array([min(255, c + tolerance) for c in color], dtype=np.uint8)
    return lower, upper

def process_image_and_mask(image_file, mask_file):
    image = cv2.imread(image_file)
    mask = cv2.imread(mask_file)

    if image is None:
        print(f"Не удалось загрузить изображение: {image_file}")
        return

    if mask is None:
        print(f"Не удалось загрузить маску: {mask_file}")
        return

    print(f"Изображение и маска успешно загружены: {image_file}, {mask_file}")

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    image = resize_image(image)
    mask = resize_image(mask)

    height, width, _ = image.shape
    yolo_output = []

    for color_tuple, class_id in COLOR_TO_CLASS.items():
        lower, upper = get_color_range(color_tuple)
        binary_mask = cv2.inRange(mask, lower, upper)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Найдено {len(contours)} контуров для цвета {color_tuple}")

        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Фильтрация по площади
                # Нормализуем координаты полигонов
                normalized_contour = contour.astype(np.float32) / [width, height]
                normalized_contour = normalized_contour.flatten()

                # Проверка длины массива, чтобы YOLO формат поддерживал полигон
                if len(normalized_contour) >= 6:  # Минимум три точки для полигона
                    yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_contour])
                    yolo_output.append(yolo_line)

    if yolo_output:
        image_name = Path(image_file).stem
        label_file = os.path.join(LABEL_FOLDER, f"{image_name}.txt")
        with open(label_file, 'w') as f:
            f.write("\n".join(yolo_output))
        print(f"Файл аннотации сохранен: {label_file}")
    else:
        print(f"Аннотация для {image_file} пуста.")

# Проверка файлов в папках
print("Проверка файлов...")
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg')]

print(f"Изображения для обработки: {image_files}")

# Обрабатываем все изображения
for image_file in image_files:
    mask_file = os.path.join(MASK_FOLDER, f"{Path(image_file).stem}.png")
    if os.path.exists(mask_file):
        print(f"Обрабатываем: {image_file} и {mask_file}")
        process_image_and_mask(os.path.join(IMAGE_FOLDER, image_file), mask_file)
        print(f"Обработка завершена для: {image_file}")
    else:
        print(f"Маска для {image_file} не найдена.")
