import os
import torch
from ultralytics import YOLO

# Check PyTorch GPU availability
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Is Available: {torch.cuda.is_available()}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_gpus}")
else:
    print("GPU is not available.")


model = YOLO("yolov9e.pt")
results = model.train(data='data.yaml', epochs=200, imgsz=800, batch=16)