import os
import cv2
import threading
import pandas as pd
from ultralytics import YOLO
from queue import Queue

image_folder = '/home/kosterin.anton2/images_all_fixed_rnd'
output_csv = 'detections.csv'

model_path = 'models/best14.pt'
device = 'cuda'
model = YOLO(model_path, device)

results_queue = Queue()

conf_threshold = 0.8
sentinel = object()

def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return

        results = model.predict(image)
        detections = []
        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    if box.conf > conf_threshold:
                        detections.append({
                            'file_name': os.path.basename(image_path),
                            'detection': box.xyxy,
                            'confidence': box.conf.item()
                        })

        if detections:
            output_image_path = os.path.join("output_images", os.path.basename(image_path))
            # annotated_image = results[0].plot()
            # draw_detections(image, detections)
            # cv2.imwrite(output_image_path, annotated_image)
            results_queue.put(detections)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def worker():
    while True:
        image_path = image_paths.get()
        if image_path is sentinel:
            break
        print(f"Processing: {image_path}")
        process_image(image_path)
        image_paths.task_done()
    image_paths.task_done()

image_paths = Queue()
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if os.path.isfile(image_path):
        image_paths.put(image_path)
    else:
        print(f"Skipping non-file: {image_path}")

num_cores = os.cpu_count()
print(f"Number of cores available: {num_cores}")

if not os.path.exists("output_images"):
    os.makedirs("output_images")

# num_threads = min(21, num_cores)
num_threads = 20
threads = []
for _ in range(num_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Добавление маркеров завершения в очередь
for _ in range(num_threads):
    image_paths.put(sentinel)

image_paths.join()
for t in threads:
    t.join()

results = []
while not results_queue.empty():
    results.extend(results_queue.get())

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
