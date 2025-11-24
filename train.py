# ------------------------------------
# ไฟล์: train.py
# ------------------------------------
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path
import yaml


def _estimate_accuracy(metrics):
    """
    ประมาณค่า Accuracy จาก confusion matrix (ถ้ามี)
    """
    confusion = getattr(metrics, 'confusion_matrix', None)
    matrix = getattr(confusion, 'matrix', None) if confusion is not None else None
    if matrix is None:
        return None
    total = matrix.sum()
    if total == 0:
        return None
    return float(np.trace(matrix) / total)


def _print_metric(label, value):
    if value is None:
        print(f"{label}: N/A")
    else:
        print(f"{label}: {value:.4f}")


def summarize_evaluation(metrics):
    """
    แสดงค่า Accuracy, Precision, Recall, F1 และ mAP (50-95)
    """
    precision = getattr(metrics.box, 'mp', None)
    recall = getattr(metrics.box, 'mr', None)
    map50_95 = getattr(metrics.box, 'map', None)
    map50 = getattr(metrics.box, 'map50', None)
    accuracy = _estimate_accuracy(metrics)
    f1_score = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = (2 * precision * recall) / (precision + recall)

    print("=== Evaluation Metrics ===")
    _print_metric("Accuracy (approx.)", accuracy)
    _print_metric("Precision (mP)", precision)
    _print_metric("Recall (mR)", recall)
    _print_metric("F1-Score", f1_score)
    _print_metric("mAP50-95", map50_95)
    _print_metric("mAP50", map50)

DEFAULT_TRAIN_CONFIG = {
    "data": "waste-detection/data.yaml",
    "imgsz": 640,
    "epochs": 100,
    "batch": 8,
    "patience": 20,
    "save_period": 1,
    "name": "yolo12m_final",
    "base_weights": "yolo12m.pt",
}


def load_train_config():
    """
    โหลดค่าพารามิเตอร์การเทรนจาก params.yaml (ถ้ามี) เพื่อให้ DVC/CI
    สามารถปรับเปลี่ยนค่าได้จากไฟล์เดียว
    """
    params_path = Path("params.yaml")
    config = DEFAULT_TRAIN_CONFIG.copy()
    if params_path.exists():
        with params_path.open("r", encoding="utf-8") as fp:
            params = yaml.safe_load(fp) or {}
        config.update(params.get("train", {}))
    return config


def train_waste_sorter():
    # 1. ตรวจสอบว่ามี GPU (NVIDIA) หรือไม่
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on device: {device}")

    config = load_train_config()

    # 2. โหลดโมเดล YOLOv12
    # เราใช้ 'yolo12m.pt' (Medium) สำหรับงานตรวจจับทั่วไป
    # หากต้องการความเร็วขึ้น ให้ใช้ 'yolo12n.pt' (Nano)
    # ไฟล์ .pt จะถูกดาวน์โหลดอัตโนมัติในครั้งแรก
    print("Loading YOLOv12 model...")
    model = YOLO(config["base_weights"])

    # 3. เริ่มต้นการเทรน
    print("Starting model training...")
    results = model.train(
        data=config["data"],     # ไฟล์ตั้งค่า Dataset
        imgsz=config["imgsz"],            # ขนาดรูปภาพมาตรฐาน
        epochs=config["epochs"],           # จำนวนรอบในการเทรน (สำหรับเทรนจริงจัง)
        batch=config["batch"],              # ลดตัวเลขนี้ถ้า GPU memory ไม่พอ (เช่น 4)
        patience=config["patience"],          # หยุดเทรนถ้า mAP ไม่ดีขึ้น
        save_period=config["save_period"],        # เซฟ checkpoint ทุก epoch
        name=config["name"], # ชื่อโฟลเดอร์ที่จะเซฟผลลัพธ์
        device=device         # ระบุอุปกรณ์ (GPU/CPU)
    )
    
    print("Training finished.")
    print("-----------------------------------")
    print("ผลการทดสอบ (Validation results):")
    
    # 4. (Optional) รัน validation อีกครั้งเพื่อดูผลสรุป mAP
    metrics = model.val(
        data=config["data"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=device,
    )
    summarize_evaluation(metrics)
    print("-----------------------------------")
    print(f"Model saved to: {metrics.save_dir}")
    print(f"Best model weights (best.pt) are in: {metrics.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_waste_sorter()

# standard fine-tune augmentation dataset before train model
# Augmentations
# Outputs per training example: 2
# Flip: Horizontal, Vertical
# 90° Rotate: Clockwise, Counter-Clockwise
# Rotation: Between -15° and +15°
# Grayscale: Apply to 25% of images
# Hue: Between -25° and +25°
# Saturation: Between -25% and +25%
# Brightness: Between -25% and +25%
# Exposure: Between -25% and +25%
# Blur: Up to 2.5px
# Noise: Up to 5% of pixels
# Cutout: 7 boxes with 3% size each