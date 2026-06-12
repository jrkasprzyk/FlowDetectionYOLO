# Configure training

This guide covers adjusting training settings through the configuration file and the command line.

## Setting precedence

Training settings are resolved in the following order, from highest to lowest precedence:

1. Command line flags (for example `--epochs 50`)
2. The configuration file passed with `--cfg` (default: `configs/train_default.yaml`)
3. Ultralytics built-in defaults

A flag that is not passed on the command line falls through to the configuration file; a key that is absent from the configuration file falls through to the ultralytics default. Each run records its complete effective settings in `runs/<name>/args.yaml`, which is the authoritative record of what a given run actually used.

## Override a setting for one run

Pass the corresponding flag on the command line:

```sh
poetry run train --epochs 50 --batch 32
```

The configuration file is unchanged; the override applies to this run only. The flags available on the command line are a subset of all training settings, chosen to cover common adjustments. See the [command line reference](../reference/cli.md) for the list.

## Change a setting permanently

Edit `configs/train_default.yaml`. The file contains a small number of active keys and a larger number of commented entries showing the ultralytics default values. To change a commented setting, uncomment it and edit the value:

```yaml
# Before
# dropout: 0.0       # dropout for the classification head; try 0.1-0.3 vs overfit

# After
dropout: 0.2
```

Any setting accepted by ultralytics for classification training may be added to the file, including settings that have no command line flag. The [training configuration reference](../reference/training-config.md) documents the keys present in the default file.

Two cautions apply:

- Do not commit machine-specific values such as `device`. The appropriate value differs across machines; see [Choose a device](choose-a-device.md).
- Detection-specific settings (for example `box`, `dfl`, `mosaic`) are accepted without error but have no effect on classification training.

## Maintain multiple configurations

To keep several training setups side by side, create additional configuration files and select one with `--cfg`:

```sh
cp configs/train_default.yaml configs/train_highres.yaml
# edit configs/train_highres.yaml: imgsz: 1280, batch: 8
poetry run train --cfg configs/train_highres.yaml
```

Each file must be complete on its own; configuration files are not merged with one another. A key absent from the selected file takes the ultralytics default, not the value from `train_default.yaml`.

## Name and locate run outputs

By default, runs are written under `runs/` (the `project` key) in automatically numbered subdirectories. To give a run a meaningful name:

```sh
poetry run train --name baseline_5class
```

Results are then written to `runs/baseline_5class/`. If the directory already exists, ultralytics appends a numeric suffix rather than overwriting it.
