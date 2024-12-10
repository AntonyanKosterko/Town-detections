import os
import json

# Укажите путь к папке с аннотациями
annotations_dir = r"E:\DataScience\Работа\Проекты\PET\Projects\Town-detections\data\gtBbox_cityPersons_trainval\gtBboxCityPersons\train"

# Множество для хранения уникальных классов
unique_classes = set()

# Рекурсивно проходим по всем папкам и файлам в annotations_dir
for root, dirs, files in os.walk(annotations_dir):
    for file_name in files:
        if not file_name.endswith(".json"):
            continue  # Пропускаем файлы, которые не являются JSON

        # Путь к текущему JSON файлу
        file_path = os.path.join(root, file_name)
        
        # Открываем и читаем файл
        with open(file_path, "r") as f:
            data = json.load(f)

        # Проходим по объектам в аннотации
        for obj in data.get("objects", []):
            label = obj.get("label")
            if label:
                unique_classes.add(label)

# Выводим уникальные классы
print("Уникальные классы в gtBbox_cityPersons:")
for cls in sorted(unique_classes):
    print(cls)
