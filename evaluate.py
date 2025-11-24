import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

try:
    from train import summarize_evaluation
except ImportError:
    summarize_evaluation = None


def ensure_summary_fn():
    if summarize_evaluation is None:
        print("ไม่สามารถนำเข้า summarize_evaluation จาก train.py ได้", file=sys.stderr)
        print("กรุณาตรวจสอบว่าไฟล์ train.py อยู่ในโฟลเดอร์เดียวกันและมีฟังก์ชัน summarize_evaluation", file=sys.stderr)
        sys.exit(1)


DEFAULT_EVAL_CONFIG = {
    "data": "waste-detection/data.yaml",
    "split": "val",
    "imgsz": 640,
    "batch": 8,
    "conf": 0.25,
    "iou": 0.7,
    "device": None,
    "weights": "artifacts/models/waste-sorter-best.pt",
    "metrics_out": "artifacts/eval/metrics.json",
}


def load_eval_config():
    params_path = Path("params.yaml")
    config = DEFAULT_EVAL_CONFIG.copy()
    if params_path.exists():
        with params_path.open("r", encoding="utf-8") as fp:
            params = yaml.safe_load(fp) or {}
        config.update(params.get("evaluate", {}))
    return config


def parse_args():
    defaults = load_eval_config()
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model on specified dataset split"
    )
    parser.add_argument(
        "--weights",
        default=defaults["weights"],
        help="Path to trained weights (.pt)",
    )
    parser.add_argument(
        "--data",
        default=defaults["data"],
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--split",
        default=defaults["split"],
        choices=["train", "val", "test"],
        help="Which dataset split to evaluate",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=defaults["imgsz"],
        help="Image size used for evaluation",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=defaults["batch"],
        help="Batch size used for evaluation",
    )
    parser.add_argument(
        "--device",
        default=defaults["device"],
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=defaults["conf"],
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=defaults["iou"],
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--metrics-out",
        default=defaults["metrics_out"],
        help="Path to save evaluation metrics summary (JSON)",
    )
    return parser.parse_args()


def evaluate_model(args):
    ensure_summary_fn()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"ไม่พบไฟล์ weights: {args.weights}")

    print("=====================================")
    print("        YOLO Evaluation Script       ")
    print("=====================================")
    print(f"Weights : {args.weights}")
    print(f"Data    : {args.data}")
    print(f"Split   : {args.split}")
    print(f"ImageSz : {args.imgsz}")
    print(f"Batch   : {args.batch}")
    print(f"Device  : {args.device or 'auto'}")
    print("-------------------------------------")

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )

    summarize_evaluation(metrics)
    write_metrics_summary(metrics, args.metrics_out)

    print("-------------------------------------")
    print(f"Reports saved to: {metrics.save_dir}")
    if args.metrics_out:
        print(f"Saved metrics summary to: {args.metrics_out}")


def write_metrics_summary(metrics, output_path):
    if not output_path:
        return
    summary = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            summary[key] = _maybe_float(value)
    # เพิ่มค่าที่สำคัญเพื่อดูง่าย
    summary.update(
        {
            "precision": _maybe_float(getattr(metrics.box, "mp", None)),
            "recall": _maybe_float(getattr(metrics.box, "mr", None)),
            "map50": _maybe_float(getattr(metrics.box, "map50", None)),
            "map50_95": _maybe_float(getattr(metrics.box, "map", None)),
        }
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)


def _maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate_model(cli_args)

