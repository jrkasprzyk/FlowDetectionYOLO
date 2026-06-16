import argparse
import json
from pathlib import Path


def _as_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def summarize_metrics(metrics, split):
    """Pull the classification accuracy fields off an ultralytics metrics object.

    Classification val exposes top1/top5 accuracy; the per-class confusion
    matrix is written under save_dir when plots are enabled (the default).
    """
    summary = {"split": split}
    top1 = getattr(metrics, "top1", None)
    top5 = getattr(metrics, "top5", None)
    if top1 is not None:
        summary["top1"] = _as_float(top1)
    if top5 is not None:
        summary["top5"] = _as_float(top5)
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        summary["save_dir"] = str(save_dir)
    return summary


def print_summary(summary, as_json):
    if as_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"split: {summary['split']}")
    if "top1" in summary:
        print(f"  top1 accuracy: {summary['top1']:.4f}")
    if "top5" in summary:
        print(f"  top5 accuracy: {summary['top5']:.4f}")
    if "save_dir" in summary:
        print(f"  outputs: {summary['save_dir']}")


def eval_model(
    weights,
    data,
    split,
    imgsz,
    project,
    name,
    exist_ok,
    verbose,
    as_json,
    device=None,
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run 'poetry install' before evaluation.")
        return 1

    model = YOLO(weights)
    val_kwargs = {
        "data": data,
        "split": split,
        "imgsz": imgsz,
        "exist_ok": exist_ok,
        "verbose": verbose,
    }
    # Resolve to an absolute path: ultralytics nests a relative project under
    # its global runs_dir setting, which does not point at this repository.
    if project:
        val_kwargs["project"] = str(Path(project).resolve())
    else:
        val_kwargs["project"] = str(Path("runs").resolve())
    if name:
        val_kwargs["name"] = name
    if device:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)
    summary = summarize_metrics(metrics, split)
    print_summary(summary, as_json)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO classifier on a labeled split "
        "(reports top1/top5 accuracy over the class/image folder layout)."
    )
    parser.add_argument(
        "--weights",
        default="from_train_2026-04-28.pt",
        help="Trained YOLO classification weights.",
    )
    parser.add_argument(
        "--data",
        default="data/classification_test_split",
        help="Dataset root containing train/val/test subdirectories.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Which split to evaluate.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 0 (first NVIDIA GPU), cpu, or mps (Apple Silicon "
        "GPU via Metal Performance Shaders). Auto-detection prefers CUDA but "
        "never picks mps, so pass --device mps explicitly on a Mac.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional output project directory for saved eval plots.",
    )
    parser.add_argument(
        "--name", default=None, help="Optional output run name for saved eval plots."
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing output run directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Ultralytics evaluation progress output.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the metrics summary as JSON."
    )
    return parser


def main():
    args = build_parser().parse_args()
    return eval_model(
        args.weights,
        args.data,
        args.split,
        args.imgsz,
        args.project,
        args.name,
        args.exist_ok,
        args.verbose,
        args.json,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
