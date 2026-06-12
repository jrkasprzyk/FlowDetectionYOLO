# Training configuration reference

The training configuration file (default: `configs/train_default.yaml`) is passed to ultralytics through the `--cfg` flag of `poetry run train`. It accepts any ultralytics training setting. This page documents the keys present in the default file. The complete list of settings is in the [ultralytics configuration documentation](https://docs.ultralytics.com/usage/cfg/) or in the installed copy at `.venv/Lib/site-packages/ultralytics/cfg/default.yaml`.

Precedence: command line flags override this file; keys absent from this file take ultralytics defaults. Each run records its effective settings in `runs/<name>/args.yaml`.

Settings specific to detection, segmentation, or pose estimation (`box`, `dfl`, `mosaic`, and others) are accepted but have no effect on classification training and are omitted from the default file.

## Active keys

These keys are set in the default file and therefore apply to every run unless overridden on the command line.

| Key | Value | Description |
| --- | --- | --- |
| `data` | `data/classification_test_split` | Pre-split dataset directory containing `train`, `val`, and optionally `test` subdirectories. Pointing at an unsplit directory causes ultralytics to re-split it on every run. |
| `epochs` | `100` | Number of training epochs. |
| `imgsz` | `640` | Training image size (square). See also `scale` under augmentation. |
| `batch` | `16` | Images per gradient step. `-1` or a float in (0, 1] enables automatic batch sizing. |
| `workers` | `2` | Dataloader worker processes. On Windows each worker is a separate Python process; large values can exhaust memory and surface as a CUDA out-of-memory error. |
| `project` | `runs` | Parent directory for run outputs, relative to the working directory. |

## Commented keys

These keys appear in the default file as comments showing the ultralytics default value. Uncomment a key to change it.

### Core

| Key | Default | Description |
| --- | --- | --- |
| `time` | unset | Maximum training time in hours. Overrides `epochs` when set. |
| `patience` | `100` | Stop early after this many epochs without validation improvement. |
| `name` | auto-numbered | Run name; outputs are written to `<project>/<name>`. |
| `exist_ok` | `False` | Overwrite an existing `<project>/<name>` directory instead of creating a numbered one. |
| `device` | auto | Compute device. Automatic selection prefers CUDA and never selects MPS. Machine-specific; do not commit a value. See [Choose a device](../how-to/choose-a-device.md). |
| `seed` | `0` | Random seed for reproducibility. |
| `resume` | `False` | Resume training from the last checkpoint in the run directory. |
| `fraction` | `1.0` | Fraction of the training dataset to use. |
| `cache` | `False` | Cache images in RAM (`True`) or on `disk` to accelerate loading. |
| `save_period` | `-1` | Additionally save a checkpoint every N epochs. Values below 1 disable periodic saving. |
| `plots` | `True` | Save plots and images (confusion matrix, training batches, curves). |
| `val` | `True` | Run validation each epoch during training. |

### Model

| Key | Default | Description |
| --- | --- | --- |
| `dropout` | `0.0` | Dropout for the classification head. Values of 0.1 to 0.3 are typical countermeasures against overfitting. |
| `freeze` | unset | Freeze the first N backbone layers. Useful for small datasets. |
| `amp` | `True` | Mixed-precision training. Reduces GPU memory use and increases speed. |

### Optimizer and learning rate

| Key | Default | Description |
| --- | --- | --- |
| `optimizer` | `auto` | Optimizer: `SGD`, `Adam`, `AdamW`, and others. `auto` selects based on the setup. |
| `lr0` | `0.01` | Initial learning rate (SGD: 1e-2; Adam and AdamW: 1e-3). |
| `lrf` | `0.01` | Final learning rate as a fraction of `lr0`. |
| `momentum` | `0.937` | SGD momentum, or Adam beta1. |
| `weight_decay` | `0.0005` | L2 regularization. |
| `warmup_epochs` | `3.0` | Learning rate warmup duration in epochs. Fractional values are accepted. |
| `cos_lr` | `False` | Cosine learning rate schedule instead of linear. |

### Augmentation

Augmentation applies at training time only.

| Key | Default | Description |
| --- | --- | --- |
| `auto_augment` | `randaugment` | Augmentation policy: `randaugment`, `autoaugment`, or `augmix`. |
| `erasing` | `0.4` | Probability of blanking a random image region. |
| `fliplr` | `0.5` | Horizontal flip probability. |
| `flipud` | `0.0` | Vertical flip probability. |
| `scale` | `0.5` | Random resize and crop gain. |
| `degrees` | `0.0` | Random rotation range in degrees. |
| `translate` | `0.1` | Random translation fraction. |
| `shear` | `0.0` | Shear in degrees. |
| `perspective` | `0.0` | Perspective warp. Typical values are 0 to 0.001. |
| `hsv_h` | `0.015` | Hue jitter. |
| `hsv_s` | `0.7` | Saturation jitter. |
| `hsv_v` | `0.4` | Brightness jitter. |
| `bgr` | `0.0` | Probability of swapping RGB and BGR channels. |
| `mixup` | `0.0` | Probability of blending two images and their labels. A regularizer for small datasets. |
| `cutmix` | `0.0` | Probability of pasting a patch of one image onto another. |
