import os
import json
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('/app/runs/segment/car_segmentation2/weights/best.pt')  # Путь к модели внутри контейнера


def calculate_paint_cost(masks, classes, class_names, front_door_area, torch_width, torch_extrusion, paint_cost_per_liter, img):
    elements = []
    for i in range(masks.shape[0]):
        mask = masks[i].cpu()
        class_index = int(classes[i])
        class_name = class_names[class_index]
        mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        area_pixels = torch.sum(mask).item()
        scale_factor = front_door_area / torch.sum(masks[0]).item()  # Пример: используем первую маску для эталона
        area_m2 = area_pixels * scale_factor
        contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True)
        processing_area = area_m2 * (1 + torch_width) * (1 + torch_extrusion)
        paint_required = processing_area / 10
        paint_cost = paint_required * paint_cost_per_liter
        elements.append({
            "element": class_name,
            "physical_area_m2": round(area_m2, 2),
            "processing_area_m2": round(processing_area, 2),
            "paint_cost": round(paint_cost, 2)
        })
    return elements


def get_labeled_image(class_names, classes, masks, labeled_image, img):
    for i in range(masks.shape[0]):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        mask_resized = cv2.resize(np.array(masks[i].cpu()), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(labeled_image, contours, -1, color, 5)
        class_index = int(classes[i])
        class_name = class_names[class_index]
        cv2.putText(labeled_image, class_name, (int(contours[0][:, 0, 0].mean()), int(contours[0][:, 0, 1].mean())), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)


@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    front_door_area = float(request.form.get('front_door_area', 1.0))
    torch_width = float(request.form.get('torch_width', 0.1))
    torch_extrusion = float(request.form.get('torch_extrusion', 0.05))
    paint_cost_per_liter = float(request.form.get('paint_cost_per_liter', 10.0))

    results = model(img, imgsz=640, iou=0.4, conf=0.8, verbose=True)

    masks = results[0].masks.data
    classes = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names

    labeled_image = get_labeled_image(class_names, classes, masks, img.copy(), img)
    paint_cost_info = calculate_paint_cost(masks, classes, class_names, front_door_area, torch_width, torch_extrusion, paint_cost_per_liter, img)

    _, buffer = cv2.imencode('.jpg', labeled_image)
    labeled_image_base64 = buffer.tobytes().hex()

    return jsonify({
        "paint_cost_info": paint_cost_info,
        "labeled_image": labeled_image_base64
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
