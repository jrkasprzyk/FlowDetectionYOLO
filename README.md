# flowdetectionYOLO

Image classification for ephemeral streams using [YOLO](https://docs.ultralytics.com/).

Refer to the full documentation [docs/](docs/README.md) for a getting-started
[tutorial](docs/tutorials/first-training-run.md), how-to guides, command line
and configuration reference, and a discussion of the dataset splitting design.

## Install

Requires Python 3.12 or 3.13 and [Poetry](https://python-poetry.org/). With those in place:

```sh
poetry install
```

The [install guide](docs/how-to/install.md) demonstrates how to install on a
fresh machine, including Python 3.12, pipx, and Poetry step by step on macOS and Windows.
To verify installation, see [Run a smoke test](docs/how-to/run-a-smoke-test.md).

See below for special considerations about Mac and Windows depending on the availability of GPUs:

| Platform | torch wheel | GPU support |
| --- | --- | --- |
| Windows / Linux + NVIDIA GPU | `+cu128` (CUDA 12.8) | CUDA, used automatically |
| Windows / Linux, no NVIDIA GPU | `+cu128` | none — runs on CPU (wheel is large but works) |
| macOS (Apple Silicon) | PyPI arm64 | MPS, **must be requested with `--device mps`** |

By default, ultralytics auto-detects the compute device: it picks the first
NVIDIA GPU (CUDA) if present, otherwise CPU. 

However, it never auto-selects the Apple GPUs (`mps`: [Metal Performance Shaders](https://developer.apple.com/metal/pytorch/) ), so on a Mac you must opt in explicitly by adding `--device mps` to the commands below.

Other accepted values: `--device 0` (first NVIDIA GPU), `--device cpu`.

Since the repo is meant to be cross-platform, please do not commit `device:` into `configs/train_default.yaml`.

## Usage

Paths in the default config are relative to the working directory.

```sh
poetry run split     # data/classification_test -> data/classification_test_split (train/val/test)
poetry run train     # trains per configs/train_default.yaml, outputs to runs/
poetry run predict   # classifies images with trained weights
poetry run eval      # top1/top5 accuracy on a labeled split (default: test)
```

Each command takes `--help` for full options. 

Due to the fact that the framework is built on ultralytics YOLO, the precedence chain for training follows this precedence:
CLI flags > `--cfg` yaml > ultralytics defaults.
