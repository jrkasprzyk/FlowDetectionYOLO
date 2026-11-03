import argparse
import contextlib
import json
import logging
import sys
import textwrap
from pathlib import Path


def _as_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _class_name(names, class_index):
    if isinstance(names, dict):
        return names.get(class_index, str(class_index))
    if isinstance(names, (list, tuple)) and class_index < len(names):
        return names[class_index]
    return str(class_index)


def summarize_result(result, top_k):
    probs = getattr(result, "probs", None)
    path = getattr(result, "path", "")

    if probs is None:
        return {
            "path": path,
            "predictions": [],
            "error": "No classification probabilities were returned.",
        }

    names = getattr(result, "names", {}) or {}
    top_indices = list(getattr(probs, "top5", []))[:top_k]
    top_confidences = list(getattr(probs, "top5conf", []))[:top_k]

    if not top_indices and hasattr(probs, "top1"):
        top_indices = [probs.top1]
        top_confidences = [probs.top1conf]

    predictions = []
    for rank, class_index in enumerate(top_indices, start=1):
        confidence = top_confidences[rank - 1] if rank <= len(top_confidences) else 0.0
        predictions.append(
            {
                "rank": rank,
                "class_index": int(class_index),
                "class_name": _class_name(names, int(class_index)),
                "confidence": _as_float(confidence),
            }
        )

    return {"path": path, "predictions": predictions}


def print_summary(summaries, as_json):
    """Print summaries as they arrive, so a long run keeps nothing in memory.

    Each one is flushed, so redirected output survives an interrupted run.
    """
    if as_json:
        # Incremental equivalent of print(json.dumps(all_summaries, indent=2)).
        separator = "[\n"
        for summary in summaries:
            body = textwrap.indent(json.dumps(summary, indent=2), "  ")
            print(f"{separator}{body}", end="", flush=True)
            separator = ",\n"
        print("[]" if separator == "[\n" else "\n]", flush=True)
        return

    for summary in summaries:
        print(summary["path"])
        if summary.get("error"):
            print(f"  {summary['error']}", flush=True)
            continue
        for prediction in summary["predictions"]:
            print(
                "  "
                f"{prediction['rank']}. "
                f"{prediction['class_name']} "
                f"({prediction['confidence']:.4f})",
                flush=True,
            )


def use_utf8_stdout_on_windows():
    """Let redirected results hold any filename, whatever the ANSI code page."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def send_ultralytics_logs_to_stderr():
    """Keep stdout to results only, so redirected output stays parseable."""
    for handler in logging.getLogger("ultralytics").handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.setStream(sys.stderr)


def _saved_output_key(path):
    """Identify the file ultralytics saves for an input path.

    Images and videos are saved with different extensions, so they never share
    a file. Filenames differing only in case do on Windows and default macOS.
    """
    from ultralytics.data.utils import VID_FORMATS

    input_path = Path(path)
    stem = input_path.stem
    if sys.platform in {"win32", "darwin"}:
        stem = stem.lower()
    return stem, input_path.suffix[1:].lower() in VID_FORMATS


def warn_on_collisions(results):
    """Pass results through, warning when saved outputs land in the same file.

    Ultralytics names saved outputs after the input stem alone, so inputs are
    compared by stem. Frames of one video share a path and do not count.
    """
    last_path_for_output = {}
    collisions = 0
    for result in results:
        path = result.path
        key = _saved_output_key(path)
        earlier_path = last_path_for_output.get(key)
        if earlier_path is not None and earlier_path != path:
            collisions += 1
            print(
                f"Warning: saved output for {path} collided with the one for "
                f"{earlier_path}; both are named {Path(path).stem}.",
                file=sys.stderr,
                flush=True,
            )
        last_path_for_output[key] = path
        yield result
    if collisions:
        print(
            f"Warning: {collisions} saved output(s) collided because inputs "
            "share a filename.",
            file=sys.stderr,
            flush=True,
        )


def predict_model(
    weights,
    source,
    imgsz,
    top_k,
    save,
    project,
    name,
    exist_ok,
    verbose,
    as_json,
    device=None,
):
    use_utf8_stdout_on_windows()
    try:
        # The first import can log, for example when creating the settings
        # file, and binds ultralytics' log handler to the current sys.stdout.
        with contextlib.redirect_stdout(sys.stderr):
            from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed. Run 'poetry install' before prediction."
        )
        return 1

    send_ultralytics_logs_to_stderr()

    model = YOLO(weights)
    predict_kwargs = {
        "source": source,
        "imgsz": imgsz,
        "save": save,
        "exist_ok": exist_ok,
        "verbose": verbose,
    }
    # Resolve to an absolute path: ultralytics nests a relative project under
    # its global runs_dir setting, which does not point at this repository.
    if project:
        predict_kwargs["project"] = str(Path(project).resolve())
    elif save:
        predict_kwargs["project"] = str(Path("runs").resolve())
    if name:
        predict_kwargs["name"] = name
    if device:
        predict_kwargs["device"] = device

    # Stream: each Results holds its full-resolution image, so collecting them
    # into a list exhausts RAM on folders of more than a few hundred photos.
    results = model.predict(stream=True, **predict_kwargs)
    if save:
        results = warn_on_collisions(results)
    summaries = (summarize_result(result, top_k) for result in results)
    print_summary(summaries, as_json)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict classes with a trained YOLO image classifier."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Trained YOLO classification model checkpoint (.pt).",
    )
    parser.add_argument(
        "--source",
        default="data/classification_test_split/val",
        help="Image, directory, glob, URL, or video source.",
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
        "--top-k",
        type=int,
        default=3,
        choices=range(1, 6),
        help="Number of top classes to print.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated prediction outputs under runs/.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional output project directory for saved predictions.",
    )
    parser.add_argument(
        "--name", default=None, help="Optional output run name for saved predictions."
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing output run directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Ultralytics prediction progress output.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print prediction summaries as JSON."
    )
    return parser


def main():
    args = build_parser().parse_args()
    return predict_model(
        args.model,
        args.source,
        args.imgsz,
        args.top_k,
        args.save,
        args.project,
        args.name,
        args.exist_ok,
        args.verbose,
        args.json,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
