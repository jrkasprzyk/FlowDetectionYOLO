# Command line reference

The project installs three commands, run as `poetry run <command>` from the repository root. Relative paths in defaults and configuration are resolved against the current working directory.

All commands exit with status 0 on success and 1 on failure. Each prints a full option listing with `--help`.

## split

Splits a classification dataset into train, val, and optionally test directories. Reads `--source`, writes `<source>_split` alongside it. Source files are copied, not moved. An existing `<source>_split` directory is deleted and rebuilt.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--source` | path | `data/classification_test` | Dataset root directory, containing one subdirectory per class. |
| `--train-ratio` | float | `0.8` | Approximate fraction of images assigned to the train split. Must be strictly between 0 and 1. |
| `--val-ratio` | float | `0.1` | Approximate fraction assigned to the val split. Must be strictly between 0 and 1. The remainder after train and val becomes the test split; if no remainder exists, no test split is created. |
| `--group-regex` | regex | none | Applied to each filename. Images sharing the match (first capture group if present, otherwise the whole match) are assigned to the same split. Non-matching files are assigned per image. Validated at argument parsing time. |
| `--seed` | int | none | Salt mixed into the assignment hash. Each value produces a different, equally deterministic split. |

Constraints: `--train-ratio` plus `--val-ratio` must not exceed 1. The command fails with exit status 1 if the source directory does not exist or contains no class subdirectories.

## train

Trains a YOLO classification model. Settings precedence: command line flags, then the `--cfg` file, then ultralytics defaults. Flags not passed on the command line do not override the configuration file.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model-config` | path | `yolo26n-cls.yaml` | YOLO model architecture definition. |
| `--weights` | path | `yolo26n-cls.pt` | Initial weights. The default is the official ImageNet-pretrained checkpoint. Resolved against the working directory; ultralytics downloads the official checkpoint automatically if the file is absent. |
| `--cfg` | path | `configs/train_default.yaml` | Training configuration file. Accepts any ultralytics training setting. See the [training configuration reference](training-config.md). |
| `--data` | path | from cfg | Pre-split dataset directory containing `train` and `val` subdirectories. Pointing at an unsplit directory causes ultralytics to re-split it on every run. |
| `--device` | string | auto | Compute device: a GPU index such as `0`, `cpu`, or `mps`. Automatic selection prefers CUDA and never selects MPS. |
| `--epochs` | int | from cfg | Number of training epochs. |
| `--imgsz` | int | from cfg | Training image size (square). |
| `--project` | path | from cfg | Parent directory for run outputs. |
| `--name` | string | auto-numbered | Run name. Outputs are written to `<project>/<name>`. |
| `--batch` | int | from cfg | Images per gradient step. `-1` or a float in (0, 1] enables automatic batch sizing. |
| `--workers` | int | from cfg | Dataloader worker processes. On Windows each worker is a separate Python process; large values can exhaust memory and surface as a CUDA out-of-memory error. |

Entries marked "from cfg" have no command line default; the value in the configuration file applies, and the ultralytics default applies if the key is absent there as well.

## predict

Runs classification inference with a trained checkpoint and prints the top classes per image.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--weights` | path | `from_train_2026-04-28.pt` | Trained classification weights. |
| `--source` | path or URL | `data/classification_test_split/val` | Image, directory, glob, URL, or video. |
| `--imgsz` | int | `640` | Inference image size. |
| `--device` | string | auto | Compute device, as for `train`. |
| `--top-k` | int, 1 to 5 | `3` | Number of top classes printed per image. |
| `--save` | boolean flag | off | Save annotated prediction images under the run directory. |
| `--project` | path | ultralytics default | Parent directory for saved outputs. |
| `--name` | string | ultralytics default | Run name for saved outputs. |
| `--exist-ok` | boolean flag | off | Reuse an existing output run directory instead of creating a numbered one. |
| `--verbose` | boolean flag | off | Print ultralytics per-image progress output. |
| `--json` | boolean flag | off | Print results as JSON instead of plain text. |

Output format, plain text: one block per image, with the image path followed by one line per predicted class, giving rank, class name, and confidence to four decimal places. Output format, JSON: a list of objects, each with `path` and `predictions`; each prediction has `rank`, `class_index`, `class_name`, and `confidence`. Images for which the model returns no classification probabilities produce an `error` field instead of predictions.
