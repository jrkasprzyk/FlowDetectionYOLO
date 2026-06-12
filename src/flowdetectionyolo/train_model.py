import argparse
from pathlib import Path


def resolve_project(project, cfg=None):
    """Return the run output directory as an absolute path.

    Ultralytics nests a relative 'project' under its global runs_dir setting,
    which points at wherever ultralytics first ran on the machine, not at this
    repository. An absolute path is used verbatim.
    """
    if project is None and cfg and Path(cfg).is_file():
        import yaml

        with open(cfg, encoding="utf-8") as fh:
            project = (yaml.safe_load(fh) or {}).get("project")
    return str(Path(project or "runs").resolve())


def train_model(model_config, weights, cfg=None, **train_args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Run 'poetry install' before training.")
        return 1

    model = YOLO(model_config)
    if weights:
        model = model.load(weights)

    model.info()
    # Only pass explicitly-set arguments so unset CLI flags fall through to the
    # cfg yaml (and from there to ultralytics defaults).
    overrides = {key: value for key, value in train_args.items() if value is not None}
    overrides["project"] = resolve_project(overrides.get("project"), cfg)
    if cfg:
        overrides["cfg"] = cfg
    model.train(**overrides)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the YOLO image classifier. "
        "Precedence: CLI flags > --cfg yaml > ultralytics defaults."
    )
    parser.add_argument("--model-config", default="yolo26n-cls.yaml", help="YOLO model config path.")
    parser.add_argument(
        "--weights",
        default="yolo26n-cls.pt",
        help="Optional weights path. Default is the official ImageNet-pretrained "
        "checkpoint; repo-trained checkpoints saw images now in val/test, so "
        "starting from them inflates eval metrics.",
    )
    parser.add_argument(
        "--cfg",
        default="configs/train_default.yaml",
        help="Training config yaml. Accepts any ultralytics train setting; "
        "the flags below override it when given.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Pre-split training data directory (train/val subdirectories). "
        "Pointing at an unsplit directory makes ultralytics re-split it on every run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 0 (first NVIDIA GPU), cpu, or mps (Apple Silicon "
        "GPU via Metal Performance Shaders). Auto-detection prefers CUDA but "
        "never picks mps, so pass --device mps explicitly on a Mac.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size.")
    parser.add_argument("--project", default=None, help="Run output directory (relative to cwd).")
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name; results land in project/name. Auto-numbered when omitted.",
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader worker processes. Each worker is a full python process on "
        "Windows; high values can exhaust system memory and surface as CUDA OOM.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    return train_model(
        args.model_config,
        args.weights,
        args.cfg,
        data=args.data,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        batch=args.batch,
        workers=args.workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
