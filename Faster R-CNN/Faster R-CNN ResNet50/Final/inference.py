import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset de tensores pre-guardados (.pt)
class PreloadedPTDataset(Dataset):
    def __init__(self, img_pt_dir, lbl_dir, img_size=800):
        self.img_pt_dir = Path(img_pt_dir)
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size
        self.img_files = sorted(f for f in self.img_pt_dir.iterdir() if f.suffix == ".pt")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = torch.load(img_path)
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        boxes, labels = self.load_yolo_labels(lbl_path)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx], dtype=torch.int64)}
        return img, target

    def load_yolo_labels(self, path):
        boxes, labels = [], []
        if not path.exists():
            return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.int64)
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)!=5:
                    continue
                cls = int(float(parts[0]))
                x, y, bw, bh = map(float, parts[1:])
                x1 = (x-bw/2)*self.img_size
                y1 = (y-bh/2)*self.img_size
                x2 = (x+bw/2)*self.img_size
                y2 = (y+bh/2)*self.img_size
                if x2>x1 and y2>y1:
                    boxes.append([x1,y1,x2,y2])
                    labels.append(cls+1)
        if len(boxes)==0:
            return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.int64)
        return torch.tensor(boxes,dtype=torch.float32), torch.tensor(labels,dtype=torch.int64)

# Collate
def collate_fn(batch):
    return tuple(zip(*batch))

# Modelo
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    return model

def validate_model(model, test_loader, device, num_classes_plot):
    """
    Validación vectorizada optimizada del modelo Faster-RCNN.
    Devuelve métricas por clase, métricas globales y matriz de confusión.
    """

    model.eval()
    cm = torch.zeros((num_classes_plot, num_classes_plot), dtype=torch.int32, device=device)

    # Diccionarios para almacenar predicciones y GT por clase (como tensores)
    preds_per_class = {cls: [] for cls in range(1, num_classes_plot + 1)}
    gts_per_class   = {cls: [] for cls in range(1, num_classes_plot + 1)}

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for t, pred in zip(targets, outputs):
                gt_boxes = t["boxes"]            # [num_gt, 4]
                gt_labels = t["labels"]          # [num_gt]
                pred_boxes = pred["boxes"]       # [num_pred, 4]
                pred_labels = pred["labels"]     # [num_pred]
                pred_scores = pred["scores"]     # [num_pred]

                if len(gt_labels) > 0 and len(pred_labels) > 0:
                    ious_matrix = box_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt]
                    matched_gt = torch.zeros(len(gt_labels), dtype=torch.bool, device=device)

                    # Matching 1-a-1 y construcción de matriz de confusión
                    for i in range(len(pred_boxes)):
                        iou_max, idx = ious_matrix[i].max(0)
                        pred_cls = pred_labels[i].item()
                        if iou_max >= 0.5 and not matched_gt[idx]:
                            true_cls = gt_labels[idx].item()
                            cm[true_cls - 1, pred_cls - 1] += 1
                            matched_gt[idx] = True
                        else:
                            # FP: predicción incorrecta (marca en fila de GT = pred_cls)
                            cm[pred_cls - 1, pred_cls - 1] += 1
                elif len(pred_boxes) > 0:
                    # Predicciones sin GT: todas FP
                    for i, pred_cls in enumerate(pred_labels):
                        cm[pred_cls.item() - 1, pred_cls.item() - 1] += 1

                # Guardar predicciones y GT por clase como tensores (no listas gigantes)
                for cls in range(1, num_classes_plot + 1):
                    gts_cls = gt_boxes[gt_labels == cls]
                    preds_cls = pred_boxes[pred_labels == cls]
                    scores_cls = pred_scores[pred_labels == cls]
                    if len(gts_cls) > 0:
                        gts_per_class[cls].append(gts_cls)
                    if len(preds_cls) > 0:
                        preds_per_class[cls].append((scores_cls, preds_cls))

    
    # Cálculo de métricas por clase
    cls_metrics = {}
    for cls in range(1, num_classes_plot + 1):
        if len(preds_per_class[cls]) == 0 and len(gts_per_class[cls]) == 0:
            cls_metrics[cls] = {"precision":0.0,"recall":0.0,"mAP50":0.0,"mAP95":0.0}
            continue

        # Concatenar todos los tensores de predicciones y GT
        pred_scores_all = torch.cat([s for s, _ in preds_per_class[cls]], dim=0) if preds_per_class[cls] else torch.tensor([], device=device)
        pred_boxes_all  = torch.cat([b for _, b in preds_per_class[cls]], dim=0) if preds_per_class[cls] else torch.empty((0,4), device=device)
        gt_boxes_all    = torch.cat(gts_per_class[cls], dim=0) if gts_per_class[cls] else torch.empty((0,4), device=device)

        # Función para calcular AP vectorizado
        def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                return 0.0, 0.0, 0.0

            # Ordenar por score
            scores_sorted, idx_sort = pred_scores.sort(descending=True)
            boxes_sorted = pred_boxes[idx_sort]

            # Calcular IoU matrix
            ious = box_iou(boxes_sorted, gt_boxes)  # [num_pred, num_gt]
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)

            TP = torch.zeros(len(boxes_sorted), device=device)
            FP = torch.zeros(len(boxes_sorted), device=device)

            for i in range(len(boxes_sorted)):
                iou_max, idx = ious[i].max(0)
                if iou_max >= iou_thr and not matched_gt[idx]:
                    TP[i] = 1
                    matched_gt[idx] = True
                else:
                    FP[i] = 1

            tp_cum = torch.cumsum(TP, dim=0)
            fp_cum = torch.cumsum(FP, dim=0)
            recalls = tp_cum / (len(gt_boxes) + 1e-8)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

            # AP usando integración trapecio
            ap = torch.trapz(precisions.cpu(), recalls.cpu()).item()
            recall_final = recalls[-1].item() if len(recalls) > 0 else 0.0
            precision_final = precisions[-1].item() if len(precisions) > 0 else 0.0
            return ap, recall_final, precision_final

        # AP50
        mAP50, recall50, precision50 = compute_ap(pred_boxes_all, pred_scores_all, gt_boxes_all, 0.5)
        # AP95: promedio de thresholds 0.5:0.05:0.95
        ap_sum = 0.0
        for thr in torch.arange(0.5, 1.0, 0.05):
            ap, _, _ = compute_ap(pred_boxes_all, pred_scores_all, gt_boxes_all, thr.item())
            ap_sum += ap
        mAP95 = ap_sum / 10

        cls_metrics[cls] = {
            "precision": precision50,
            "recall": recall50,
            "mAP50": mAP50,
            "mAP95": mAP95
        }


    # Métricas globales
    global_metrics = {metric: sum(cls_metrics[c][metric] for c in cls_metrics)/num_classes_plot 
                        for metric in ["precision","recall","mAP50","mAP95"]}

    return cls_metrics, global_metrics, cm


# Main
if __name__ == "__main__":

    TEST_IMG = r"E:\Escuela\Redes Neuronales\Angelica\Data\test\images_pt"
    TEST_LBL = r"E:\Escuela\Redes Neuronales\Angelica\Data\test\labels"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = PreloadedPTDataset(TEST_IMG, TEST_LBL)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    output_dir = Path(r"E:\Escuela\Redes Neuronales\Angelica\Resultados")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 5
    num_classes_plot = 4

    checkpoints = [12, 25]

    for epoch in checkpoints:
        checkpoint_path = f"saves/checkpoint_epoch_{epoch}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found, skipping.")
            continue

        print(f"Loading checkpoint for epoch {epoch}")
        model = get_model(num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # Run inference
        cls_metrics, global_metrics, cm = validate_model(model, test_loader, device, num_classes_plot)

        # Create subdirectory for this checkpoint
        checkpoint_dir = output_dir / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        
        # Tabla de métricas por clase
        df_metrics_class = pd.DataFrame({
            cls: {metric: cls_metrics[cls][metric] for metric in ["precision","recall","mAP50","mAP95"]}
            for cls in range(1, num_classes_plot + 1)
        }).T

        plt.figure(figsize=(12,6), dpi=200)
        plt.axis('off')
        plt.table(
            cellText=df_metrics_class.values,
            colLabels=df_metrics_class.columns,
            rowLabels=[f"Clase {c}" for c in df_metrics_class.index],
            loc='center',
            cellLoc='center'
        )
        plt.title(f"Métricas por Clase (Epoch {epoch})", fontsize=16)
        plt.savefig(checkpoint_dir / "metrics_table_per_class.png")
        plt.close()

        # Tabla de métricas globales
        df_metrics_global = pd.DataFrame({
            metric: global_metrics[metric] for metric in ["precision","recall","mAP50","mAP95"]
        }, index=["Global"])

        plt.figure(figsize=(10,4), dpi=200)
        plt.axis('off')
        plt.table(
            cellText=df_metrics_global.values,
            colLabels=df_metrics_global.columns,
            rowLabels=df_metrics_global.index,
            loc='center',
            cellLoc='center'
        )
        plt.title(f"Métricas Globales (Epoch {epoch})", fontsize=16)
        plt.savefig(checkpoint_dir / "metrics_table_global.png")
        plt.close()

        
        # Matriz de confusión global
        plt.figure(figsize=(8,6), dpi=150)
        sns.heatmap(cm.cpu(), annot=True, fmt='d', cmap="Blues",
                    xticklabels=[f"C{c}" for c in range(1,num_classes_plot+1)],
                    yticklabels=[f"C{c}" for c in range(1,num_classes_plot+1)])
        plt.xlabel("Predicho", fontsize=14)
        plt.ylabel("Verdadero", fontsize=14)
        plt.title(f"Matriz de Confusión Global (Epoch {epoch})", fontsize=16)
        plt.savefig(checkpoint_dir / "confusion_matrix_global.png")
        plt.close()

        print(f"Inference for epoch {epoch} completed. Results saved in {checkpoint_dir}")