# Evaluación Yolov8 en Test 
import torch
from pathlib import Path
import pandas as pd
from ultralytics import YOLO 

# Configuración
DEVICE = 0 if torch.cuda.is_available() else "cpu"

WEIGHTS_PATH = r"E:\Escuela\Redes Neuronales\Angelica\Resultados Yolo\yolo_final\weights\best.pt"
DATA_YAML = r"E:\Escuela\Redes Neuronales\Angelica\Data\data.yaml"

IMGSZ = 640
BATCH = 8
WORKERS = 8

OUTPUT_DIR = Path(r"E:\Escuela\Redes Neuronales\Angelica\Yolo\Test_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Main
if __name__ == "__main__":

    # Cargar modelo
    model = YOLO(WEIGHTS_PATH)

    # Validación  en TEST
    results = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        split="test",   # CLAVE
        verbose=True,
    )

    # Métricas Globales
    df_global = pd.DataFrame([{
        "Precision": results.box.mp,
        "Recall": results.box.mr,
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
    }])

    global_csv = OUTPUT_DIR / "test_global_metrics.csv"
    df_global.to_csv(global_csv, index=False)

    # Métricas por clase
    rows = []
    for i, name in results.names.items():
        rows.append({
            "Clase": name,
            "Precision": results.box.p[i],
            "Recall": results.box.r[i],
            "mAP50": results.box.ap50[i],
            "mAP50-95": results.box.ap[i],
        })

    df_classes = pd.DataFrame(rows)
    class_csv = OUTPUT_DIR / "test_metrics_per_class.csv"
    df_classes.to_csv(class_csv, index=False)

    print("\nEVALUACIÓN EN TEST COMPLETADA")
    print(f"Métricas globales: {global_csv}")
    print(f"Métricas por clase: {class_csv}")
