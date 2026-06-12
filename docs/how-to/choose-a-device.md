# Choose a device

This guide covers selecting the compute device for training and prediction on each supported platform.

## Default behavior

When no device is specified, ultralytics selects one automatically: the first NVIDIA GPU if CUDA is available, otherwise the CPU. Apple GPUs are never selected automatically. On an Apple Silicon Mac, omitting the device flag therefore results in CPU execution even though a GPU is present.

## Specify a device

Both `train` and `predict` accept a `--device` flag:

```sh
poetry run train --device mps
poetry run predict --device mps
```

Accepted values:

| Value | Meaning |
| --- | --- |
| `0`, `1`, ... | NVIDIA GPU with that index, via CUDA |
| `cpu` | CPU execution |
| `mps` | Apple Silicon GPU, via Metal Performance Shaders |

MPS (Metal Performance Shaders) is Apple's GPU compute framework. PyTorch uses it as the backend for Apple Silicon GPUs, in the same role CUDA serves for NVIDIA GPUs.

The device may also be set with a `device:` key in the training configuration file. Avoid committing such a value to the repository, because the appropriate device differs across machines. Use the command line flag, or keep the key in a local, uncommitted copy of the configuration.

## Platform summary

| Platform | Installed PyTorch build | Recommended usage |
| --- | --- | --- |
| Windows or Linux with NVIDIA GPU | CUDA 12.8 (`+cu128`) | No flag required; CUDA is selected automatically |
| Windows or Linux without NVIDIA GPU | CUDA 12.8 (`+cu128`) | No flag required; execution falls back to CPU |
| macOS, Apple Silicon | Standard PyPI build | Pass `--device mps` to use the GPU |

All three cases install from the same `poetry.lock`. The CUDA build is installed on Windows and Linux regardless of whether a GPU is present; on machines without one, the only cost is the larger download, and execution proceeds on the CPU.

## Verify which device was used

For training, the device appears in the run banner that ultralytics prints at startup and is recorded in `runs/<name>/args.yaml`. To check device availability directly:

```sh
poetry run python -c "import torch; print(torch.cuda.is_available(), torch.backends.mps.is_available())"
```

The first value reports CUDA availability, the second MPS availability.
