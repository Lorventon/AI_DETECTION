from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Параметры, вводимые пользователем
torch_width = 0.1  # Ширина факела краскопульта (в метрах)
torch_extrusion = 0.08  # Вылет факела за границу элемента (в метрах)
cost_per_liter_LKM = 100  # Стоимость 1 литра ЛКМ (единицы)

# Модель YOLO
model = YOLO(
    'C:\\Users\\Lorventon\\PycharmProjects\\HackInHome2024\\runs\\segment\\car_segmentation8\\weights\\best.pt')

# Загрузка изображения
img = cv2.imread('C:\\Users\\Lorventon\\PycharmProjects\\HackInHome2024\\model\\dataset\\train\\images\\IMG_9208.jpg')

print(img.shape)

# Получаем результаты модели YOLO
results = model(img, imgsz=640, iou=0.4, conf=0.8, verbose=True)

# Извлекаем классы объектов и имена классов
classes = results[0].boxes.cls.cpu().numpy()
class_names = results[0].names

# Получение бинарных масок и их количество
masks = results[0].masks.data  # Формат: [число масок, высота, ширина]
num_masks = masks.shape[0]

# Определение случайных цветов для каждой маски
colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]  # Случайные цвета

# Создание изображения для отображения масок
mask_overlay = np.zeros_like(img)

labeled_image = img.copy()

# Обработаем каждый элемент и его маску
for i in range(num_masks):
    color = colors[i]  # Случайный цвет
    mask = masks[i].cpu()

    # Изменение размера маски до размеров исходного изображения с использованием метода ближайших соседей
    mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Получение класса для текущей маски
    class_index = int(classes[i])
    class_name = class_names[class_index]

    # Получение контуров маски для рисования
    mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(labeled_image, mask_contours, -1, color, 5)
    cv2.putText(labeled_image, class_name,
                (int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean())),
                cv2.FONT_HERSHEY_SIMPLEX, 4, color, 6)

    # Расчет периметра элемента
    perimeter = cv2.arcLength(mask_contours[0], True)

    # Площадь обработки
    processing_area = perimeter * torch_width  # Площадь без учета вылета факела
    processing_area = max(processing_area, 0)  # Площадь не может быть отрицательной

    # Учитываем вылет факела
    processing_area += 2 * torch_extrusion * perimeter  # Добавляем площадь, которая учитывает вылет факела

    # Ограничение площади для очень крупных значений
    if processing_area > 500:  # Устанавливаем верхний предел для площади
        processing_area = 500  # Это значение можно подкорректировать

    # Рассчитываем физическую площадь на основе эталонных данных
    reference_area = 0.8  # Площадь передней двери (в метрах квадратных)
    # Пример того, как мы можем корректировать площади других объектов на основе эталона:
    physical_area = (reference_area / 0.8) * processing_area  # Пропорциональный расчет

    # Расчет стоимости ЛКМ
    liters_of_LKM = processing_area / 10  # Рассчитываем количество ЛКМ в литрах (примерно)
    cost_LKM = liters_of_LKM * cost_per_liter_LKM  # Стоимость ЛКМ для этого элемента

    # Печать результатов
    print(f"Элемент: {class_name}")
    print(f"  Физическая площадь: {physical_area:.2f} м²")
    print(f"  Площадь обработки: {processing_area:.2f} м²")
    print(f"  Стоимость ЛКМ: {cost_LKM:.2f} единиц")
    print("\n")

# Отобразим итоговое изображение с наложенными масками и подписями
plt.figure(figsize=(8, 8), dpi=150)
labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
plt.imshow(labeled_image)
plt.axis('off')
plt.show()
