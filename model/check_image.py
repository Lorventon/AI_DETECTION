import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO

<<<<<<< HEAD
def process_image_result():
    # Загрузка модели
    model = YOLO('../runs/segment/car_segmentation8/weights/best.pt')
=======
model = YOLO('../runs/segment/car_segmentation2/weights/best.pt')
>>>>>>> 275aad7f530a21fe9b8e478962069deedce07416

<<<<<<< HEAD
img = cv2.imread('../dataset/train/images/7958beu-960.jpg')
=======
    # Загрузка изображения
    img = cv2.imread('D:/Projects/HackInHome2024/model/test.png')
    if img is None:
        raise FileNotFoundError("Изображение не найдено по указанному пути.")
>>>>>>> b7170f5318c7be98a1630039dcb7b17762ed5d15

    print(f"Image shape: {img.shape}")

    # Прогон изображения через модель YOLO
    results = model(img, imgsz=640, iou=0.4, conf=0.8, verbose=True)

    # Получение классов, масок и имен классов
    classes = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    masks = results[0].masks.data
    num_masks = masks.shape[0]

    # Случайные цвета для каждой маски
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]

    # Копирование изображения для масок и подписей
    labeled_image = img.copy()

    # Наложение масок и добавление подписей
    for i in range(num_masks):
        color = colors[i]
        mask = masks[i].cpu()

        # Масштабирование маски до размеров изображения
        mask_resized = cv2.resize(
            np.array(mask),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Наложение маски на изображение
        labeled_image[mask_resized > 0.5] = color

        # Получение класса и имени
        class_index = int(classes[i])
        class_name = class_names[class_index]

        # Добавление подписей
        mask_contours, _ = cv2.findContours(
            mask_resized.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(mask_contours) > 0:
            x, y = int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean())
            cv2.putText(
                labeled_image,
                class_name,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

<<<<<<< HEAD
    # Конвертация в RGB для сохранения
    labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
=======
    # Получение класса для текущей маски
    class_index = int(classes[i])
    class_name = "" # class_names[class_index]
>>>>>>> 275aad7f530a21fe9b8e478962069deedce07416

    # Конвертация изображения в Base64
    pil_image = Image.fromarray(labeled_image_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Возврат Base64 изображения
    return image_base64
