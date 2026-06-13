# Documentation

This documentation is organized according to the [Diátaxis framework](https://diataxis.fr/). Each section serves a distinct purpose.

## Tutorials

Learning-oriented lessons for newcomers.

- [Your first training run](tutorials/first-training-run.md): from installation to a trained classifier and a first prediction.

## How-to guides

Task-oriented recipes for specific problems.

- [Install from scratch](how-to/install.md): set up Python 3.12, pipx, Poetry, and the project on a fresh macOS or Windows machine.
- [Configure training](how-to/configure-training.md): use the configuration file, override settings, and maintain multiple configurations.
- [Split a dataset](how-to/split-a-dataset.md): control split ratios, create a test holdout, and keep related images in the same split.
- [Choose a device](how-to/choose-a-device.md): select CUDA, CPU, or Apple MPS on each platform.
- [Run a smoke test](how-to/run-a-smoke-test.md): verify the pipeline on a new machine with a small public dataset.

## Reference

Technical descriptions of the command line interfaces and configuration keys.

- [Command line reference](reference/cli.md): all flags for `split`, `train`, and `predict`.
- [Training configuration reference](reference/training-config.md): all keys accepted in the training configuration file.

## Explanation

Discussion of design decisions.

- [Dataset splitting design](explanation/dataset-splitting-design.md): why split assignment is hash-based rather than shuffle-based, and the consequences of that choice.
