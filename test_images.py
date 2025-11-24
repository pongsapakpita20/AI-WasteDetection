import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run trained YOLO model on image(s) to verify detections."
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/yolo12m_final/weights/best.pt",
        help="Path to trained weights (.pt)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to image, directory, video file, or webcam index",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--project",
        default="runs/test_images",
        help="Directory to store annotated outputs",
    )
    parser.add_argument(
        "--name",
        default="exp",
        help="Run name inside project directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display windows with detections (OpenCV required)",
    )
    return parser.parse_args()


def validate_paths(args):
    if not os.path.isfile(args.weights):
        print(f"ไม่พบไฟล์ weights: {args.weights}", file=sys.stderr)
        sys.exit(1)
    # allow aliases like "webcam" -> "0"
    if isinstance(args.source, str) and args.source.lower() == "webcam":
        args.source = "0"
    if not (os.path.exists(args.source) or str(args.source).isdigit()):
        print(f"ไม่พบ source: {args.source}", file=sys.stderr)
        sys.exit(1)
    Path(args.project).mkdir(parents=True, exist_ok=True)


def summarize_results(results):
    print("=====================================")
    print("         Inference Summary           ")
    print("=====================================")
    for idx, result in enumerate(results):
        path = result.path
        boxes = result.boxes
        masks = getattr(result, "masks", None)
        num_boxes = len(boxes) if boxes is not None else 0
        num_masks = len(masks) if masks is not None else 0
        print(f"[{idx}] {path} -> {num_boxes} boxes, {num_masks} masks")
        if num_boxes:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            counts = {}
            for cid in cls_ids:
                counts[cid] = counts.get(cid, 0) + 1
            counts_str = ", ".join(f"class {cid}: {cnt}" for cid, cnt in counts.items())
            print(f"     {counts_str}")
    print("=====================================")


def run_inference(args):
    validate_paths(args)
    model = YOLO(args.weights)
    print("เริ่มรันโมเดลตรวจจับภาพ...")
    try:
        results = model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            save=True,
            show=args.show,
        )
    except Exception as e:
        # Friendly hints for common webcam/display issues
        msg = str(e)
        if "Failed to open" in msg and str(args.source).isdigit():
            print("ไม่สามารถเปิดกล้องได้: ลองเปลี่ยน --source เป็น 0 หรือ 1 และปิดโปรแกรมที่ใช้กล้องอยู่ก่อน", file=sys.stderr)
        if "cv2.imshow" in msg or "The function is not implemented" in msg:
            print("สภาพแวดล้อมนี้ไม่รองรับการแสดงผลด้วย cv2.imshow(). ให้เอา --show ออก หรือใช้ python app.py ที่เป็นเว็บแทน", file=sys.stderr)
        raise
    summarize_results(results)
    save_dir = Path(args.project) / args.name
    print(f"ภาพที่มีกรอบ annotation ถูกบันทึกไว้ที่: {save_dir.resolve()}")


if __name__ == "__main__":
    arguments = parse_args()
    run_inference(arguments)

