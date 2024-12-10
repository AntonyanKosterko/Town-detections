from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels import name2label
import os

# Определяем пути
input_dir = "/path/to/annotations"
output_dir = "/path/to/yolo_labels"
os.makedirs(output_dir, exist_ok=True)

# Карта классов
class_map = {
    "ignore": 0,
    "pedestrian": 1,
    "person (other)": 2,
    "person group": 3,
    "rider": 4,
    "sitting person": 5,
}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

# Обработка файлов
for file_name in os.listdir(input_dir):
    if not file_name.endswith(".json"):
        continue

    annotation_file = os.path.join(input_dir, file_name)
    annotation = Annotation()
    annotation.fromJsonFile(annotation_file)

    img_width, img_height = annotation.imgWidth, annotation.imgHeight
    yolo_lines = []

    for obj in annotation.objects:
        label = obj.label
        if label not in class_map:
            continue

        bbox = obj.bbox
        yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
        yolo_lines.append(f"{class_map[label]} " + " ".join(map(str, yolo_bbox)))

    output_file = os.path.join(output_dir, file_name.replace(".json", ".txt"))
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_lines))
