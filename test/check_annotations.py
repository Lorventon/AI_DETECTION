import cv2
import matplotlib.pyplot as plt

# Визуализация аннотаций
image_file = 'test/dataset/images/train/IMG_9210.jpg'
label_file = 'test/dataset/labels/IMG_9210.txt'

image = cv2.imread(image_file)
height, width, _ = image.shape

with open(label_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    class_id, x_center, y_center, w, h = map(float, line.split()[:5])
    x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
    x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()