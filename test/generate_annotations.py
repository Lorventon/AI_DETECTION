# import os
# import cv2
# import numpy as np
# from pathlib import Path

# # Папки с изображениями и масками
# IMAGE_FOLDER = 'dataset/images/train'
# MASK_FOLDER = 'dataset/masks/train'
# LABEL_FOLDER = 'dataset/labels/train'

# # Словарь с именами классов и их ID для YOLO
# CLASS_NAMES = {
#     "bumper": 0,
#     "fog_lights": 1,
#     "radiator": 2,
#     "license_plate": 3,
#     "emblem": 4,
#     "hood": 5,
#     "fender": 6,
#     "windshield": 7,
#     "roof": 8,
#     "back_fender": 9,
#     "side_mirrors": 10,
#     "door_handle": 11,
#     "tires": 12,
#     "wheel": 13,
#     "back_door": 14,
#     "front_door": 15
# }

# # Создаем папку для сохранения аннотаций
# os.makedirs(LABEL_FOLDER, exist_ok=True)

# def convert_mask_to_yolo(mask, image_shape):
#     """Конвертация маски в формат YOLO."""
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     height, width = image_shape[:2]
#     yolo_annotations = []

#     for contour in contours:
#         # Вычисляем ограничивающий прямоугольник
#         x, y, w, h = cv2.boundingRect(contour)
#         if w == 0 or h == 0:
#             continue  # Пропускаем пустые маски

#         # Нормализация координат
#         x_center = (x + w / 2) / width
#         y_center = (y + h / 2) / height
#         w_norm = w / width
#         h_norm = h / height

#         # Преобразуем контур в список координат (если нужно)
#         contour_points = contour.flatten().tolist()
#         yolo_annotations.append((x_center, y_center, w_norm, h_norm, contour_points))

#     return yolo_annotations

# def process_image_and_mask(image_file):
#     """Обработка изображений и соответствующих масок."""
#     image_name = Path(image_file).stem
#     image = cv2.imread(os.path.join(IMAGE_FOLDER, image_file))

#     yolo_output = []

#     for mask_file in os.listdir(MASK_FOLDER):
#         if mask_file.startswith(image_name):  # Ищем маску для текущего изображения
#             mask = cv2.imread(os.path.join(MASK_FOLDER, mask_file), cv2.IMREAD_GRAYSCALE)
#             class_name = mask_file.split('_')[1].split('.')[0]  # Извлекаем имя класса

#             if class_name in CLASS_NAMES:
#                 class_id = CLASS_NAMES[class_name]
#                 annotations = convert_mask_to_yolo(mask, image.shape)

#                 # Добавляем аннотации для каждого объекта
#                 for x_center, y_center, w_norm, h_norm, contour in annotations:
#                     yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
#                     yolo_output.append(yolo_line)

#     # Сохраняем аннотацию в файл
#     label_file = os.path.join(LABEL_FOLDER, f"{image_name}.txt")
#     with open(label_file, 'w') as f:
#         f.write("\n".join(yolo_output))

# # Обрабатываем все изображения в папке
# for image_file in os.listdir(IMAGE_FOLDER):
#     if image_file.endswith(".jpg") or image_file.endswith(".png"):
#         process_image_and_mask(image_file)


import os
import cv2
import numpy as np
from pathlib import Path

# Папки с изображениями и масками
IMAGE_FOLDER = 'test/dataset/images/train'
MASK_FOLDER = 'test/dataset/masks/train'
LABEL_FOLDER = 'test/dataset/labels'

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
        lower, upper = get_color_range(color_tuple)  # Получаем диапазон для цвета
        binary_mask = cv2.inRange(mask, lower, upper)  # Маска для диапазона

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Найдено {len(contours)} контуров для цвета {color_tuple}")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:  # Фильтруем маленькие контуры
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
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