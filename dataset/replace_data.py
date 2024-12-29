import os
import shutil

def flatten_directory(input_dir, output_dir):
    """
    Перемещает все файлы из подпапок input_dir в одну папку output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(output_dir, file)

            # Если файл с таким именем уже есть, добавим префикс
            if os.path.exists(dst):
                base, ext = os.path.splitext(file)
                dst = os.path.join(output_dir, f"{base}_dup{ext}")
            
            shutil.move(src, dst)
            print(f"Moved: {src} -> {dst}")

# Укажите папки для train и val
flatten_directory("/home/kosterin.anton2/dop/Town-detections/data/leftImg8bit_trainvaltest/leftImg8bit/train", "/home/kosterin.anton2/dop/Town-detections/train_data/images/train")
flatten_directory("/home/kosterin.anton2/dop/Town-detections/data/leftImg8bit_trainvaltest/leftImg8bit/val", "/home/kosterin.anton2/dop/Town-detections/train_data/images/val")
