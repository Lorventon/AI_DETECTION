import os
import shutil
from sklearn.model_selection import train_test_split

# Пути к папкам с изображениями и аннотациями
images_path = 'dataset/images/train'
labels_path = 'dataset/labels/train'

# Списки всех файлов
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
label_files = [f.replace('.jpg', '.txt') for f in image_files]

# Разделяем данные на обучающие и валидационные
train_images, val_images = train_test_split(image_files, test_size=0.2)

# Создаем папки для обучающей и валидационной выборки
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)

# Перемещаем изображения и аннотации в соответствующие папки
for img in train_images:
    shutil.move(os.path.join(images_path, img), os.path.join('dataset/images/train', img))
    shutil.move(os.path.join(labels_path, img.replace('.jpg', '.txt')), os.path.join('dataset/labels/train', img.replace('.jpg', '.txt')))

for img in val_images:
    shutil.move(os.path.join(images_path, img), os.path.join('dataset/images/val', img))
    shutil.move(os.path.join(labels_path, img.replace('.jpg', '.txt')), os.path.join('dataset/labels/val', img.replace('.jpg', '.txt')))