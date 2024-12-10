import torch
from ultralytics import YOLO

# Проверяем доступность GPU
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Is Available: {torch.cuda.is_available()}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_gpus}")
else:
    print("GPU is not available.")

# Загружаем YOLOv5 модель (через ultralytics)
model = YOLO("yolov5s.pt")  # Используем легкую модель yolov5s для экономии памяти

# Запускаем обучение
results = model.train(
    data='data.yaml',      # Путь к файлу конфигурации данных
    epochs=200,            # Количество эпох
    imgsz=640,             # Размер изображения (уменьшите до 512 или 320 при необходимости)
    batch=8,               # Размер батча (уменьшите, чтобы уложиться в 5 ГБ)
    device=0               # Используем GPU
)
