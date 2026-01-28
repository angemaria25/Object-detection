# Entrenamiento Final
import torch
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# Ajustes Globales
SETTINGS["hub"] = False
SETTINGS["api_key"] = ""

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Hiperparámetros
BEST_LR = 0.00013560072092105602
BEST_OPTIMIZER = "AdamW"
BEST_BATCH = 8

# Configuración General
DATA_YAML = r"E:\Escuela\Redes Neuronales\Angelica\Data\data.yaml"
MODEL_WEIGHTS = "yolov8l.pt"

EPOCHS = 50          
IMGSZ = 640
WORKERS = 8

PROJECT_DIR = Path(r"E:\Escuela\Redes Neuronales\Angelica\Yolo")
RUN_NAME = "yolo_final"

OUTPUT_DIR = PROJECT_DIR / "Resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Carga del modelo (Yolo maneja last.pt internamente)

    model = YOLO(MODEL_WEIGHTS)

    # Train

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=DEVICE,
        batch=BEST_BATCH,
        lr0=BEST_LR,
        optimizer=BEST_OPTIMIZER,
        workers=WORKERS,
        cache=True,
        amp=True,
        save=True,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        resume=True,     # CLAVE
        verbose=True,
    )

    
    # Validación Final

    val_results = model.val()

    # Métricas Globales (csv limpio)

    global_metrics = {
        "Precision": val_results.box.mp,
        "Recall": val_results.box.mr,
        "mAP50": val_results.box.map50,
        "mAP50-95": val_results.box.map,
    }

    df_global = pd.DataFrame([global_metrics])
    global_csv = OUTPUT_DIR / "global_metrics.csv"
    df_global.to_csv(global_csv, index=False)

    # Métricas por clase

    class_names = val_results.names
    box_metrics = val_results.box

    rows = []
    for i, name in class_names.items():
        rows.append({
            "Clase": name,
            "Precision": box_metrics.p[i],
            "Recall": box_metrics.r[i],
            "mAP50": box_metrics.ap50[i],
            "mAP50-95": box_metrics.ap[i],
        })

    df_classes = pd.DataFrame(rows)
    class_csv = OUTPUT_DIR / "metrics_per_class.csv"
    df_classes.to_csv(class_csv, index=False)

    # Resultados
    
    print("\nENTRENAMIENTO Y VALIDACIÓN FINALIZADOS")
    print(f"Gráficas automáticas YOLO: {PROJECT_DIR / RUN_NAME}")
    print(f"Métricas globales: {global_csv}")
    print(f"Métricas por clase: {class_csv}")
