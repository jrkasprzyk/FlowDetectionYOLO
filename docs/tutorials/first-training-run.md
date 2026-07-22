# Your first training run

This tutorial walks through the complete workflow once: installing the project, splitting a dataset, training a classifier, running a prediction, and evaluating accuracy. At the end you will have a trained model checkpoint, a classification result for one of your own images, and an accuracy score on the test split.

## Prerequisites

- Python 3.12 or 3.13 (not 3.14)
- [Poetry](https://python-poetry.org/)
- A classification dataset: a directory containing one subdirectory per class, with the images of each class inside its subdirectory. For example:

```
data/classification_test/
├── flowing/
│   ├── img_001.jpg
│   └── img_002.jpg
└── not_flowing/
    ├── img_101.jpg
    └── img_102.jpg
```

The directory names define the class labels. Your dataset can live in any directory; `data/classification_test` is only the default the tools assume, and this tutorial uses it in its examples. If your dataset is elsewhere, point the split step at it with `split --source-dir <path>`. The [command line reference](../reference/cli.md) lists every flag and default.

## Step 1: Install

From the repository root:

```sh
poetry install
```

This creates a virtual environment and installs all dependencies, including PyTorch and ultralytics. On Windows and Linux the CUDA build of PyTorch is installed; on macOS the standard build is installed. See [Choose a device](../how-to/choose-a-device.md) for what this means for training speed.

If you are starting from a machine without Python 3.12 or Poetry, follow [Install from scratch](../how-to/install.md) first; it sets up Python, pipx, and Poetry on macOS and Windows before this `poetry install` step.

## Step 2: Split the dataset

Training requires the dataset to be divided into training, validation, and test subsets. Run:

```sh
poetry run split
```

This reads the directory given by `--source-dir` (default `data/classification_test`) and writes a new directory, `data/classification_test_split`, containing `train`, `val`, and `test` subdirectories. For a dataset in a custom location, run `poetry run split --source-dir <path>` instead. By default 80% of images are assigned to `train`, 10% to `val`, and the remaining 10% to `test`. The original dataset is not modified. All split options are listed in the [command line reference](../reference/cli.md#split).

The command prints a summary:

```
Split dataset written to data\classification_test_split (train: 160, val: 21, test: 19)
```

The counts approximate the requested ratios rather than matching them exactly. The reasons are described in [Dataset splitting design](../explanation/dataset-splitting-design.md).

## Step 3: Train

Run:

```sh
poetry run train --epochs 5
```

This trains a YOLO classification model on the split dataset. The `--data` flag selects the pre-split dataset directory; when omitted, as here, it comes from the configuration file. Training starts from the checkpoint given by `--initial-weights`, by default the official ImageNet-pretrained checkpoint. This is distinct from the `--model` flag of `predict` and `eval`, which takes a finished, trained checkpoint. The `--epochs 5` flag keeps this first run short; the default configuration specifies 100 epochs. Training settings come from `configs/train_default.yaml`, and any flag passed on the command line takes precedence over the file. All train options are listed in the [command line reference](../reference/cli.md#train).

On a machine with an NVIDIA GPU, training uses CUDA automatically. On an Apple Silicon Mac, add `--device mps` to train on the GPU; otherwise training runs on the CPU.

Ultralytics prints the model architecture, then a progress line per epoch. When training finishes, results are written to a run directory under `runs/`, for example `runs/train/`. The directory contains:

- `weights/best.pt`: the checkpoint with the best validation accuracy
- `weights/last.pt`: the checkpoint from the final epoch
- `args.yaml`: the complete effective settings of the run
- plots of training curves and a confusion matrix

## Step 4: Predict

Use the trained checkpoint to classify the validation images. The `--model` flag is required and takes the checkpoint produced in Step 3:

```sh
poetry run predict --model runs/train/weights/best.pt
```

By default this classifies every image in `data/classification_test_split/val` and prints the top three classes for each, with confidence values. The results go to the terminal only; no files are written unless `--save` is passed, in which case annotated copies of the images are saved under `runs/` (for example `runs/predict/`):

```
data\classification_test_split\val\flowing\img_042.jpg
  1. flowing (0.9812)
  2. not_flowing (0.0188)
```

To classify a single image instead, pass its path with `--source`:

```sh
poetry run predict --model runs/train/weights/best.pt --source path/to/image.jpg
```

Here `--source` is an inference input: a single image, a flat directory of images, a glob, a URL, or a video. It is not the same as `split --source-dir`, which names a class-foldered dataset. The `--imgsz` and `--device` flags mean the same here as in `train`. All predict options are listed in the [command line reference](../reference/cli.md#predict).

## Step 5: Evaluate

`predict` reports per-image guesses but no overall score, and it does not read the `<split>/<class>/<image>` folder layout. `eval` does: it treats each class subdirectory as the ground-truth label for the images inside it, and from that computes top1 and top5 accuracy plus a confusion matrix, which `predict` cannot do. To measure accuracy on the held-out test split:

```sh
poetry run eval --model runs/train/weights/best.pt --split test
```

It reads `data/classification_test_split/test` and prints the accuracies:

```
split: test
  top1 accuracy: 0.9333
  top5 accuracy: 1.0000
  outputs: <repository>/runs/val
```

A confusion matrix is written to the output directory. All eval options are listed in the [command line reference](../reference/cli.md#eval).

## Where to go next

- [Configure training](../how-to/configure-training.md) describes how to adjust epochs, image size, augmentation, and other settings permanently rather than per command.
- [Split a dataset](../how-to/split-a-dataset.md) covers split ratios, the test holdout, and grouping related images.
- The [command line reference](../reference/cli.md), linked from each step above, collects every flag of all four commands in one place.
