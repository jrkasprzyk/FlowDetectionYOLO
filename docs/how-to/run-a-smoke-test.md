# Run a smoke test

This guide verifies the full pipeline (installation, training, prediction) on a machine in about a minute, using a small public dataset instead of project data. Use it after setting up a new machine or changing dependencies.

## Dataset

Ultralytics downloads several classification datasets automatically when their name is passed as the training data argument. The smallest is `mnist160`: 160 MNIST digit images (8 per class, 10 classes, under 1 MB). The download is stored in the ultralytics global datasets directory, not in this repository. To display that directory:

```sh
poetry run python -c "from ultralytics import settings; print(settings['datasets_dir'])"
```

## Procedure

Train for one epoch:

```sh
poetry run train --data mnist160 --epochs 1 --imgsz 64 --name smoke
```

On an Apple Silicon Mac, append `--device mps` (see [Choose a device](choose-a-device.md)). The first run additionally downloads the pretrained checkpoint `yolo26n-cls.pt` into the working directory.

Expected result: ultralytics downloads the dataset if absent, trains one epoch in a few seconds, and reports `Results saved to <repository>/runs/smoke`.

Then predict with the new checkpoint against one class folder of the downloaded dataset:

```sh
poetry run predict --weights runs/smoke/weights/best.pt --source <datasets_dir>/mnist160/test/7 --top-k 1
```

Replace `<datasets_dir>` with the directory printed above. Expected result: one block per image, each naming a predicted digit class with a confidence value.

## Interpreting the outcome

The smoke test verifies mechanics, not model quality. After a single epoch on 80 training images, accuracy near chance (top-1 around 0.1 to 0.3) is normal, and most predictions will be wrong. The test passes if both commands complete without error, the run directory appears under `runs/`, and predictions are printed.

Larger datasets are available by the same mechanism when a longer test is wanted: `imagenette160` (about 100 MB), `cifar10` (about 170 MB). Substitute the name in the `--data` flag.

## Note on output locations

Ultralytics maintains global settings (`datasets_dir`, `runs_dir`, `weights_dir`) that default to wherever it was first used on a machine. The training and prediction commands in this repository resolve their output directory to an absolute path under the current working directory, so run outputs land in this repository's `runs/` regardless of those settings. Dataset downloads still follow the global `datasets_dir`.
