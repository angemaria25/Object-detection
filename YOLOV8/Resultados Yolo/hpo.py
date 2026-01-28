import optuna
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import torch
import gc
import multiprocessing


SETTINGS["hub"] = False
SETTINGS["api_key"] = ""

# Configuración
DATA_YAML = r"E:\Escuela\Redes Neuronales\Angelica\Data\data.yaml"
YOLO_ROOT = "runs_hpo"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

N_EPOCHS = 6
N_TRIALS = 5
STORAGE = "sqlite:///optuna.db"


# Objective
def objective(trial: optuna.Trial):
    # Hiperparámetros 
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    print(
        f"\n=== Trial {trial.number + 1}/{N_TRIALS} ===\n"
        f"Hyperparameters:\n"
        f"  lr = {lr}\n"
        f"  optimizer = {optimizer_name}\n"
    )

    # Crear modelo nuevo por trial
    model = YOLO("yolov8l.pt")

    # Train completo en una sola llamada
    model.train(
        data=DATA_YAML,
        epochs=N_EPOCHS,
        lr0=lr,
        optimizer=optimizer_name,  
        imgsz=640,
        batch=8,
        device=DEVICE,
        project=YOLO_ROOT,
        name=f"trial_{trial.number}",
        save=False,
        exist_ok=True,
        verbose=True,
        workers=8,
    )

    # Validaciín final
    val_results = model.val()
    best_metric = float(val_results.box.map)  # mAP50-95

    print(f"\nTrial {trial.number} completed | Best mAP50-95={best_metric:.4f}")

    # Limpieza profunda
    del model
    torch.cuda.empty_cache()
    gc.collect()
    # Limpia estado global de Ultralytics

    # Optuna minimiza, así que devolvemos 1 - metric
    return 1.0 - best_metric


# Main
if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    study = optuna.create_study(
        study_name="yolo_hpo",
        direction="minimize",
        storage=STORAGE,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=2,
            min_early_stopping_rate=0,
        ),
    )

    # Todos los trials en una sola llamada
    study.optimize(objective, n_trials=N_TRIALS)

    if study.best_trial is not None:
        print(
            "\n>>> BEST OVERALL <<<\n"
            f"Trial {study.best_trial.number}\n"
            f"Score = {study.best_value:.4f}\n"
            f"Params = {study.best_trial.params}\n"
        )
