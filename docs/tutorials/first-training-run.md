# Your first training run

This tutorial walks through the complete workflow once: installing the project, splitting a dataset, training a classifier, and running a prediction. At the end you will have a trained model checkpoint and a classification result for one of your own images.

## Prerequisites

- Python 3.12 or later
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

The directory names define the class labels. This tutorial assumes the dataset is at `data/classification_test`, which is the default location expected by the tools. If your dataset is elsewhere, pass its path with `--source`.

## Step 1: Install

From the repository root:

```sh
poetry install
```

This creates a virtual environment in `.venv` and installs all dependencies, including PyTorch and ultralytics. On Windows and Linux the CUDA build of PyTorch is installed; on macOS the standard build is installed. See [Choose a device](../how-to/choose-a-device.md) for what this means for training speed.

## Step 2: Split the dataset

Training requires the dataset to be divided into training, validation, and test subsets. Run:

```sh
poetry run split
```

This reads `data/classification_test` and writes a new directory, `data/classification_test_split`, containing `train`, `val`, and `test` subdirectories. By default 80% of images are assigned to `train`, 10% to `val`, and the remaining 10% to `test`. The original dataset is not modified.

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

This trains a YOLO classification model on the split dataset. The `--epochs 5` flag keeps this first run short; the default configuration specifies 100 epochs. Training settings come from `configs/train_default.yaml`, and any flag passed on the command line takes precedence over the file.

On a machine with an NVIDIA GPU, training uses CUDA automatically. On an Apple Silicon Mac, add `--device mps` to train on the GPU; otherwise training runs on the CPU.

Ultralytics prints the model architecture, then a progress line per epoch. When training finishes, results are written to a run directory under `runs/`, for example `runs/train/`. The directory contains:

- `weights/best.pt`: the checkpoint with the best validation accuracy
- `weights/last.pt`: the checkpoint from the final epoch
- `args.yaml`: the complete effective settings of the run
- plots of training curves and a confusion matrix

## Step 4: Predict

Use the trained checkpoint to classify the validation images:

```sh
poetry run predict --weights runs/train/weights/best.pt
```

By default this classifies every image in `data/classification_test_split/val` and prints the top three classes for each, with confidence values:

```
data\classification_test_split\val\flowing\img_042.jpg
  1. flowing (0.9812)
  2. not_flowing (0.0188)
```

To classify a single image instead, pass its path with `--source`:

```sh
poetry run predict --weights runs/train/weights/best.pt --source path/to/image.jpg
```

## Where to go next

- [Configure training](../how-to/configure-training.md) describes how to adjust epochs, image size, augmentation, and other settings permanently rather than per command.
- [Split a dataset](../how-to/split-a-dataset.md) covers split ratios, the test holdout, and grouping related images.
- The [command line reference](../reference/cli.md) lists every available flag.
