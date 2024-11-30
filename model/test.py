import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('../runs/segment/car_segmentation8/weights/best.pt')

# Функция для автоматического нахождения контрольных точек
# Функция для автоматического нахождения контрольных точек
def find_keypoints(image):
    # Используем ORB для обнаружения ключевых точек
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    
    # Преобразуем ключевые точки в список координат
    points = np.array([kp[i].pt for i in range(len(kp))])
    
    return points

# Функция для нахождения гомографии и исправления перспективы
def correct_perspective(image, control_points_image, control_points_real):
    # Вычисляем матрицу гомографии
    H, _ = cv2.findHomography(control_points_image, control_points_real)
    
    # Применяем гомографию для исправления перспективы
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (width, height))
    
    return warped_image

# Функция для расчета реальной площади из пикселей
def convert_area_to_real(pixels_area, control_points_image, control_points_real):
    # Масштаб: пиксели в метр
    pixel_per_meter = (control_points_image[1][0] - control_points_image[0][0]) / (control_points_real[1][0] - control_points_real[0][0])
    
    # Преобразуем площадь в квадратные метры
    real_area = pixels_area / (pixel_per_meter ** 2)
    
    return real_area

# Основная функция для выполнения всех шагов
def process_image(img, known_size_real, model):
    # Сегментация: предполагаем, что мы получаем bounding box и маску объекта
    results = model(img, imgsz=640, iou=0.4, conf=0.8, verbose=True)
    
    # Проверяем структуру results, чтобы понять, как работать с ними
    print("jhdfgdhjgflkj", results[0].boxes)  # Выведем структуру результатов на экран

    # Предположим, что результат возвращается как список, содержащий объект с атрибутами
    # например, YOLO возвращает boxes как (x1, y1, x2, y2)
    
    if isinstance(results, list):
        boxes = results[0].boxes.xyxy  # Получаем bounding box
    else:
        boxes = results.xyxy[0]  # Если результат похож на тот, что был до этого
        
    print("Bounding boxes:", boxes)  # Проверим координаты bounding box

    # Извлекаем координаты bounding box для объекта (например, двери)
    x1, y1, x2, y2 = boxes[0]  # Получаем первую пару координат, если это список
    
    # Вычисляем площадь двери в пикселях
    pixels_area = (x2 - x1) * (y2 - y1)

    # Преобразуем изображение в оттенки серого
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Находим ключевые точки на изображении
    keypoints_image = find_keypoints(gray_image)
    
    # Для простоты выбираем 4 наиболее подходящих точки для гомографии
    # Можно использовать дополнительные алгоритмы для подбора соответствующих точек
    control_points_image = keypoints_image[:4]
    
    # Переводим точки в реальные координаты (например, если дверь имеет размеры 1.2м х 2м)
    control_points_real = np.array([[0, 0], [1.2, 0], [1.2, 2], [0, 2]])
    
    # Исправляем перспективу
    corrected_image = correct_perspective(img, control_points_image, control_points_real)
    
    # Преобразуем площадь в реальные метры
    real_area = convert_area_to_real(pixels_area, control_points_image, control_points_real)
    
    # Сохраняем исправленное изображение
    cv2.imwrite("corrected_image.png", corrected_image)
    
    print(f"Реальная площадь двери: {real_area} м²")
    
    return corrected_image, real_area

# Пример использования
img = cv2.imread('D:/Projects/HackInHome2024/model/dataset/train/images/IMG_9211.jpg')  # Загружаем изображение

# Известный реальный размер объекта (например, дверь)
known_size_real = (1.2, 2.0)  # 1.2 метра ширина, 2.0 метра высота

# Здесь должен быть ваш модельный объект, например:
# model = load_your_model()  # Загрузите вашу модель YOLO или любую другую модель

# Пример обработки изображения
corrected_image, real_area = process_image(img, known_size_real, model)