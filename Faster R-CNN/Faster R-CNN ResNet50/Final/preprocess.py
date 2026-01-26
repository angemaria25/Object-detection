import os
from pathlib import Path
import cv2
import torch
from tqdm import tqdm

# ===============================
# CONFIGURACIÓN
# ===============================
IMG_DIR = Path("Data/test/images")      # imágenes originales
OUT_DIR = Path("Data/test/images_pt")   # salida .pt
IMG_SIZE = 800                           # debe coincidir con tu training

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# PREPROCESADO
# ===============================
def preprocess_and_save(img_path, out_path, img_size):
    img = cv2.imread(str(img_path))
    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    torch.save(tensor, out_path)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    img_files = [
        f for f in IMG_DIR.iterdir()
        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ]

    print(f"Procesando {len(img_files)} imágenes...")

    for img_path in tqdm(img_files):
        out_path = OUT_DIR / (img_path.stem + ".pt")
        preprocess_and_save(img_path, out_path, IMG_SIZE)

    print("Preprocesado finalizado.")
