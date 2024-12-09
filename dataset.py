import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

def visualize_cityscapes_dataset(dataset_root):
    # Путь к папкам с изображениями и метками
    images_path = os.path.join(dataset_root, 'leftImg8bit_trainvaltest/leftImg8bit/train')  # Папка с изображениями
    labels_path = os.path.join(dataset_root, 'gtBbox_cityPersons_trainval/gtBboxCityPersons/train')  # Папка с метками

    # Перебор городов в папке с изображениями
    for city in os.listdir(images_path):
        city_images_path = os.path.join(images_path, city)
        city_labels_path = os.path.join(labels_path, city)

        # Список всех файлов изображений в папке города
        img_files = sorted([f for f in os.listdir(city_images_path) if f.endswith('.png') or f.endswith('.jpg')])
        
        # Визуализация изображений и соответствующих меток
        for img_file in img_files:
            img_path = os.path.join(city_images_path, img_file)
            
            # Извлекаем базовое имя файла (например, aachen_000000_000019)
            base_name = img_file.replace('_leftImg8bit.png', '').replace('_leftImg8bit.jpg', '')
            json_file = f'{base_name}_gtBboxCityPersons.json'
            json_path = os.path.join(city_labels_path, json_file)

            # Загружаем и отображаем изображение
            image = Image.open(img_path)
            plt.imshow(image)
            plt.title(f'Изображение: {img_file}')
            plt.axis('off')

            # Загружаем метки из JSON-файла и рисуем ограничивающие рамки
            if os.path.exists(json_path):
                with open(json_path) as f:
                    annotations = json.load(f)

                # Проходим по всем объектам и рисуем их ограничивающие рамки
                for obj in annotations.get('objects', []):
                    bbox = obj['bbox']  # bbox - это [x_min, y_min, width, height]
                    
                    if len(bbox) == 4:
                        x_min, y_min, width, height = bbox
                        # Проверка на корректность координат
                        if x_min >= 0 and y_min >= 0 and (x_min + width) <= image.width and (y_min + height) <= image.height:
                            # Рисуем рамку на изображении
                            rect = patches.Rectangle((x_min, y_min), width, height,
                                                     linewidth=2, edgecolor='r', facecolor='none')
                            plt.gca().add_patch(rect)
                        else:
                            print(f'Warning: Bounding box out of image bounds for {img_file}: {bbox}')
                    else:
                        print(f'Warning: Invalid bounding box format for {img_file}: {bbox}')
            else:
                print(f'Warning: No JSON file found for {img_file}')

            plt.show()  # Отображаем текущее изображение
            
# Замените 'path_to_cityscapes' на фактический путь к вашему датасету
visualize_cityscapes_dataset('')