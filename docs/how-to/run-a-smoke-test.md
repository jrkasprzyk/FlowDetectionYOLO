# Run a smoke test

This guide verifies the full pipeline (installation, training, prediction, evaluation) on a machine in about a minute, using a small public dataset instead of project data. Use it after setting up a new machine or environment to confirm commands and dependencies work end to end.

## Dataset

Ultralytics downloads several classification datasets automatically when their name is passed as the training data argument. The smallest is `mnist160`: 160 MNIST digit images (8 per class, 10 classes).

`mnist160` is organized as:

- `train/<digit>/`
- `test/<digit>/`

There is no separate `val/` directory in this dataset.

Find your datasets directory:

```sh
poetry run python -c "from ultralytics import settings; print(settings['datasets_dir'])"
```

### Split naming note (`test` vs `val`)

For this smoke test, Ultralytics uses `mnist160/test` as the validation source during training because the dataset ships with only `train` and `test` folders. As a result, training logs may label metrics as `val/*` even though files come from `.../mnist160/test/...`.

This is expected and does not indicate a mismatch.

Quick mapping:

- Dataset folder name: `test`
- Role during `train`: validation (`val` in logs)
- Role during explicit evaluation with `--split test`: test

## Procedure

Train for one epoch:

```sh
poetry run train --data mnist160 --epochs 1 --imgsz 64 --name smoke
```

On an Apple Silicon Mac, append `--device mps` (see [Choose a device](choose-a-device.md)). The first run additionally downloads the pretrained checkpoint `yolo26n-cls.pt` into the working directory.

Expected result: ultralytics downloads the dataset if absent, trains one epoch in a few seconds, and reports `Results saved to <repository>/runs/smoke`.

Then predict with the new checkpoint against one class folder of the downloaded dataset:

```sh
poetry run predict --model runs/smoke/weights/best.pt --source <datasets_dir>/mnist160/test/7 --top-k 1
```

Replace `<datasets_dir>` with the directory printed above. Expected result: one block per image, each naming a predicted digit class with a confidence value.

Then score the whole test split, which `mnist160` already lays out as `test/<digit>/`:

```sh
poetry run eval --model runs/smoke/weights/best.pt --data <datasets_dir>/mnist160 --split test
```

Expected result: a top1 and top5 accuracy line and an output directory under `runs/`.

## Interpreting the outcome

The smoke test verifies mechanics, not model quality. After a single epoch on 80 training images, accuracy near chance (top-1 around 0.1 to 0.3) is normal, and most predictions will be wrong. The test is successful if commands complete and produce artifacts in the expected locations.

Larger datasets are available by the same mechanism when a longer test is wanted: `imagenette160` (about 100 MB), `cifar10` (about 170 MB). Substitute the name in the `--data` flag.

## Note on output locations

Ultralytics maintains global settings (`datasets_dir`, `runs_dir`, `weights_dir`) that default to wherever it was first used on a machine. The training, prediction, and evaluation commands in this repo use relative run paths (for example `runs/smoke/...`) and dataset paths rooted at `datasets_dir`. If output appears in an unexpected location, inspect:

```sh
poetry run python -c "from ultralytics import settings; print(settings)"
```
