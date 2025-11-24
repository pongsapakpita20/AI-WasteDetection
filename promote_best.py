"""
Utility script to copy the latest YOLO best weights into a canonical artifact
path so DVC can track the single file instead of the entire `runs` directory.
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import yaml

try:
    from train import load_train_config
except ImportError:
    load_train_config = None


DEFAULT_DEST = "artifacts/models/waste-sorter-best.pt"


def _load_default_dest():
    params_path = Path("params.yaml")
    if params_path.exists():
        params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
        evaluate_cfg = params.get("evaluate", {})
        if evaluate_cfg.get("weights"):
            return evaluate_cfg["weights"]
    return DEFAULT_DEST


def parse_args():
    parser = argparse.ArgumentParser(description="Copy best.pt into artifacts")
    parser.add_argument(
        "--source",
        help="Path to YOLO best.pt (default: derived from params.yaml)",
    )
    parser.add_argument(
        "--dest",
        default=_load_default_dest(),
        help="Destination path for promoted weights",
    )
    return parser.parse_args()


def resolve_source(user_source: Optional[str]) -> Path:
    if user_source:
        return Path(user_source)
    if load_train_config is None:
        raise RuntimeError(
            "Cannot infer source path without train.load_train_config; "
            "pass --source explicitly."
        )
    config = load_train_config()
    run_name = config.get("name", "yolo12m_final")
    return Path("runs") / "detect" / run_name / "weights" / "best.pt"


def main():
    args = parse_args()
    source = resolve_source(args.source)
    dest = Path(args.dest)

    if not source.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {source}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"Copied latest weights from {source} -> {dest}")


if __name__ == "__main__":
    main()

