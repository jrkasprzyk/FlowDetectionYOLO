# Command line reference

The project installs four commands, run as `poetry run <command>` from the repository root. Relative paths in defaults and configuration are resolved against the current working directory.

All commands exit with status 0 on success and 1 on failure. Each prints a full option listing with `--help`.

## Dataset inputs: `--source` vs `--data`

The commands follow the ultralytics naming convention for their inputs. A `--source` flag names a raw input to process: for `split`, `--source-dir` is the unsplit dataset with one folder per class; for `predict`, `--source` is an arbitrary inference source, which may be a single image, a directory, a glob, a URL, or a video. A `--data` flag names a prepared, already-split dataset root fed to the model, as in `train` and `eval`; this mirrors the `data` argument of ultralytics `train` and `val`.

The checkpoint arguments follow the same convention. `predict` and `eval` take `--model`, a required path to a trained checkpoint, matching ultralytics' `model=` argument. `train` instead takes `--model-config`, the architecture yaml, plus `--initial-weights`, the checkpoint whose weights initialize training.

## split

Splits a classification dataset into train, val, and optionally test directories. Reads `--source-dir`, writes `<source-dir>_split` alongside it. Source files are copied, not moved. An existing `<source-dir>_split` directory is deleted and rebuilt.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--source-dir` | path | `data/classification_test` | Dataset source directory, containing one subdirectory per class. |
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
| `--initial-weights` | path | `yolo26n-cls.pt` | Checkpoint whose weights initialize training. The default is the official ImageNet-pretrained checkpoint. Resolved against the working directory; ultralytics downloads the official checkpoint automatically if the file is absent. |
| `--cfg` | path | `configs/train_default.yaml` | Training configuration file. Accepts any ultralytics training setting. See the [training configuration reference](training-config.md). |
| `--data` | path | from cfg | Pre-split dataset directory containing `train` and `val` subdirectories. Pointing at an unsplit directory causes ultralytics to re-split it on every run. |
| `--device` | string | auto | Compute device: a GPU index such as `0`, `cpu`, or `mps`. Automatic selection prefers CUDA and never selects MPS. |
| `--epochs` | int | from cfg | Number of training epochs. |
| `--imgsz` | int | from cfg | Training image size (square). |
| `--project` | path | from cfg | Parent directory for run outputs. Resolved to an absolute path against the working directory; without this resolution, ultralytics would place relative paths under its global `runs_dir` setting. |
| `--name` | string | auto-numbered | Run name. Outputs are written to `<project>/<name>`. |
| `--batch` | int | from cfg | Images per gradient step. `-1` or a float in (0, 1] enables automatic batch sizing. |
| `--workers` | int | from cfg | Dataloader worker processes. On Windows each worker is a separate Python process; large values can exhaust memory and surface as a CUDA out-of-memory error. |

Entries marked "from cfg" have no command line default; the value in the configuration file applies, and the ultralytics default applies if the key is absent there as well.

## predict

Runs classification inference with a trained checkpoint and prints the top classes per image.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | path | (required) | Trained classification model checkpoint (.pt). |
| `--source` | path or URL | `data/classification_test_split/val` | Image, directory, glob, URL, or video. |
| `--imgsz` | int | `640` | Inference image size. |
| `--device` | string | auto | Compute device, as for `train`. |
| `--top-k` | int, 1 to 5 | `3` | Number of top classes printed per image. |
| `--save` | boolean flag | off | Save annotated prediction images under the run directory. |
| `--project` | path | `runs` when `--save` is passed | Parent directory for saved outputs. Resolved to an absolute path against the working directory. |
| `--name` | string | ultralytics default | Run name for saved outputs. |
| `--exist-ok` | boolean flag | off | Reuse an existing output run directory instead of creating a numbered one. |
| `--verbose` | boolean flag | off | Print ultralytics per-image progress output. |
| `--json` | boolean flag | off | Print results as JSON instead of plain text. |

Output format, plain text: one block per image, with the image path followed by one line per predicted class, giving rank, class name, and confidence to four decimal places. Output format, JSON: a list of objects, each with `path` and `predictions`; each prediction has `rank`, `class_index`, `class_name`, and `confidence`. Images for which the model returns no classification probabilities produce an `error` field instead of predictions.

`predict` treats a directory source as a flat collection of images and does not descend into class subdirectories. To score a labeled split laid out as `<split>/<class>/<image>`, use `eval`, which reads that structure directly and reports accuracy.

## eval

Evaluates a trained checkpoint on a labeled split and prints top1 and top5 accuracy. Reads the standard `<split>/<class>/<image>` folder layout, so the class subdirectory names supply the ground-truth labels. Ultralytics writes a confusion matrix and related plots under the run directory.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | path | (required) | Trained classification model checkpoint (.pt). |
| `--data` | path | `data/classification_test_split` | Dataset root containing `train`, `val`, and `test` subdirectories. |
| `--split` | `train`, `val`, or `test` | `test` | Which split under the dataset root to evaluate. |
| `--imgsz` | int | `640` | Evaluation image size. |
| `--device` | string | auto | Compute device, as for `train`. |
| `--project` | path | `runs` | Parent directory for saved plots. Resolved to an absolute path against the working directory. |
| `--name` | string | ultralytics default | Run name for saved plots. |
| `--exist-ok` | boolean flag | off | Reuse an existing output run directory instead of creating a numbered one. |
| `--verbose` | boolean flag | off | Print ultralytics per-batch progress output. |
| `--json` | boolean flag | off | Print the metrics summary as JSON instead of plain text. |

Output format, plain text: the split name, then top1 and top5 accuracy to four decimal places, then the output directory holding the plots. Output format, JSON: an object with `split`, `top1`, `top5`, and `save_dir`.

Caveat: meaningful numbers require a checkpoint trained on a split that excluded the evaluated set; evaluating a checkpoint whose training data overlaps val or test inflates accuracy.
