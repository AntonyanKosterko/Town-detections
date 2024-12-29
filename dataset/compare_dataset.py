import os
import json
from tqdm import tqdm

# Определяем пути
input_dir = "dop/Town-detections/data/gtBbox_cityPersons_trainval/gtBboxCityPersons/val"  # Папка train, содержащая папки городов
output_dir = "dop/Town-detections/data/yolo_val_labels"
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

# Функция для преобразования bbox в формат YOLO
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

# Получение общего числа файлов для прогресса
total_files = sum(
    len(files) for _, _, files in os.walk(input_dir) if files
)

# Инициализация индикатора выполнения
with tqdm(total=total_files, desc="Обработка файлов") as pbar:
    # Проход по папкам
    for city_folder in os.listdir(input_dir):
        city_path = os.path.join(input_dir, city_folder)
        if not os.path.isdir(city_path):
            continue

        for file_name in os.listdir(city_path):
            if not file_name.endswith(".json"):
                pbar.update(1)  # Обновление прогресса для пропущенных файлов
                continue

            json_path = os.path.join(city_path, file_name)

            # Чтение JSON
            with open(json_path, "r") as f:
                data = json.load(f)

            img_width = data["imgWidth"]
            img_height = data["imgHeight"]
            yolo_lines = []

            for obj in data.get("objects", []):
                label = obj["label"]
                if label not in class_map:
                    continue  # Пропуск, если класса нет в class_map

                bbox = obj.get("bbox")
                if not bbox:
                    continue  # Пропуск, если bbox отсутствует

                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                yolo_lines.append(f"{class_map[label]} " + " ".join(map(str, yolo_bbox)))

            # Сохранение меток в YOLO-формате
            output_file = os.path.join(output_dir, f"{city_folder}_{file_name.replace('.json', '.txt')}")
            with open(output_file, "w") as f:
                f.write("\n".join(yolo_lines))

            pbar.update(1)  # Обновление индикатора выполнения

print("Конвертация завершена!")