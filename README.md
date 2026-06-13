# flowdetectionyolo

YOLO image classification pipeline: split a dataset, train a classifier, run
predictions.

Full documentation is in [docs/](docs/README.md): a getting-started
[tutorial](docs/tutorials/first-training-run.md), how-to guides, command line
and configuration reference, and a discussion of the dataset splitting design.

## Install

Requires Python 3.12 or 3.13 (not 3.14 — PyTorch has no 3.14 wheels yet) and
[Poetry](https://python-poetry.org/). With those in place:

```sh
poetry install
```

Starting from a fresh machine? The [install guide](docs/how-to/install.md)
walks through Python 3.12, pipx, and Poetry step by step on macOS and Windows.
To verify an installation end to end in about a minute, see
[Run a smoke test](docs/how-to/run-a-smoke-test.md).

Works on all platforms from the same lock file:

| Platform | torch wheel | GPU support |
| --- | --- | --- |
| Windows / Linux + NVIDIA GPU | `+cu128` (CUDA 12.8) | CUDA, used automatically |
| Windows / Linux, no NVIDIA GPU | `+cu128` | none — runs on CPU (wheel is large but works) |
| macOS (Apple Silicon) | PyPI arm64 | MPS, **must be requested with `--device mps`** |

## Usage

Run everything from the repo root (paths in the default config are relative
to the working directory).

```sh
poetry run split     # data/classification_test -> data/classification_test_split (train/val/test)
poetry run train     # trains per configs/train_default.yaml, outputs to runs/
poetry run predict   # classifies images with trained weights
```

Each command takes `--help` for full options. Training setting precedence:
CLI flags > `--cfg` yaml > ultralytics defaults.

## Device selection

By default, ultralytics auto-detects the compute device: it picks the first
NVIDIA GPU (CUDA) if present, otherwise CPU. It never auto-selects Apple
GPUs, so on a Mac you must opt in explicitly:

```sh
poetry run train --device mps
poetry run predict --device mps
```

`mps` is [Metal Performance Shaders](https://developer.apple.com/metal/pytorch/),
Apple's GPU compute framework. PyTorch uses it as the backend for Apple
Silicon GPUs, in the role CUDA serves for NVIDIA GPUs.

Other accepted values: `--device 0` (first NVIDIA GPU), `--device cpu`.
Don't commit `device:` into `configs/train_default.yaml` — the right value
differs per machine.
