import cv2
import numpy as np
import matplotlib.pyplot as plt

# Визуализация аннотаций
image_file = 'dataset/images/train/IMG_9209.jpg'
label_file = 'dataset/labels/train/IMG_9209.txt'

# Загрузка изображения
image = cv2.imread(image_file)
height, width, _ = image.shape

# Цвета для отображения классов
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)
]

# Чтение файла аннотаций
with open(label_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    data = line.split()
    class_id = int(data[0])  # Чтение идентификатора класса
    polygon_points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)  # Чтение точек полигона
    polygon_points[:, 0] *= width  # Масштабирование координат по ширине
    polygon_points[:, 1] *= height  # Масштабирование координат по высоте
    polygon_points = polygon_points.astype(np.int32)  # Преобразование в целые числа

    # Отрисовка полигона
    cv2.polylines(image, [polygon_points], isClosed=True, color=COLORS[class_id % len(COLORS)], thickness=2)
    # Отображение имени класса рядом с полигоном
    label_text = f"Class {class_id}"
    cv2.putText(image, label_text, (polygon_points[0][0], polygon_points[0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id % len(COLORS)], thickness=1)

# Отображение изображения с аннотациями
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
