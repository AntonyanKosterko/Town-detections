import os
import json

# Определите соответствие имен классов и их ID
class_map = {
    "sitting person": 0,
    "rider": 1,
    # Добавьте другие классы при необходимости
}

# Функция для конвертации bbox в формат YOLO
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

# Папки с исходными аннотациями и целевыми
input_annotations_dir = "path/to/json_annotations"
output_labels_dir = "path/to/yolo_labels"

# Создаем выходную папку, если она не существует
os.makedirs(output_labels_dir, exist_ok=True)

# Обрабатываем все JSON файлы
for annotation_file in os.listdir(input_annotations_dir):
    if not annotation_file.endswith(".json"):
        continue

    input_path = os.path.join(input_annotations_dir, annotation_file)
    output_path = os.path.join(output_labels_dir, annotation_file.replace(".json", ".txt"))

    # Читаем JSON файл
    with open(input_path, "r") as f:
        data = json.load(f)

    img_width = data["imgWidth"]
    img_height = data["imgHeight"]

    yolo_annotations = []

    for obj in data["objects"]:
        label = obj["label"]
        if label == "ignore":  # Пропускаем объекты с меткой "ignore"
            continue

        # Получаем класс ID
        class_id = class_map.get(label)
        if class_id is None:  # Пропускаем неизвестные классы
            continue

        # Конвертируем bbox в YOLO формат
        bbox = obj["bbox"]
        yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
        yolo_annotations.append(f"{class_id} " + " ".join(map(str, yolo_bbox)))

    # Сохраняем результат в файл
    with open(output_path, "w") as f:
        f.write("\n".join(yolo_annotations))

print("Конвертация завершена.")
