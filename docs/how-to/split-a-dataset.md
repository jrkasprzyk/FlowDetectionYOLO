# Split a dataset

This guide covers controlling the behavior of `poetry run split`: ratios, the test holdout, grouping related images, and regenerating splits.

The split tool reads a source directory containing one subdirectory per class and writes a new directory named `<source>_split` with `train`, `val`, and optionally `test` subdirectories. Source files are copied, not moved. Split assignment is deterministic and based on hashing; the consequences of this design are described in [Dataset splitting design](../explanation/dataset-splitting-design.md).

## Choose split ratios

The defaults are 0.8 train, 0.1 val, with the remainder (0.1) becoming the test split. To change them:

```sh
poetry run split --train-ratio 0.7 --val-ratio 0.2
```

This requests approximately 70% train, 20% val, and 10% test. Both ratios must be strictly between 0 and 1, and their sum must not exceed 1. Actual split sizes approximate the requested ratios; they do not match exactly.

## Omit the test split

Set the ratios so that train and val sum to 1:

```sh
poetry run split --train-ratio 0.8 --val-ratio 0.2
```

When no fraction remains for test, only `train` and `val` directories are created.

## Keep related images in the same split

Datasets often contain near-duplicate images: multiple shots of the same scene, or frames from the same camera session. If such images are assigned to splits independently, near-duplicates of training images can appear in val or test, which inflates evaluation metrics.

The `--group-regex` flag prevents this. The regular expression is applied to each filename. All images whose filenames produce the same match are assigned to the same split. If the expression contains a capture group, the first capture group is used as the group identifier; otherwise the whole match is used.

For example, if filenames embed a numeric set identifier (`site_104233_a.jpg`, `site_104233_b.jpg`):

```sh
poetry run split --group-regex "(\d{6,})"
```

Both images match group `104233` and are therefore assigned to the same split.

Files that do not match the expression fall back to per-image assignment. The command reports how many groups were formed and how many images did not match:

```
Grouped into 412 groups; 7 images did not match the group regex and were assigned per-image
```

A nonzero unmatched count indicates filenames that do not follow the expected pattern. Verify that the expression is correct before relying on the split for evaluation.

Grouping makes split sizes lumpier, because whole groups move between splits rather than single images.

For expressions tailored to this project's camera-trap filenames, including grouping by year, month, flow event, or camera, see [Customize groupings with regex](customize-groupings-with-regex.md).

## Produce a different split

Assignment is deterministic: rerunning the command with the same arguments reproduces the same split. To obtain a different assignment, for example to check that results are not an artifact of one particular split, pass an integer seed:

```sh
poetry run split --seed 1
```

Each seed value produces a different, equally stable assignment. Omitting the seed keeps the canonical split. Note that comparing models across different seeds means comparing them on different test sets.

## Regenerate after adding images

Rerun the same command:

```sh
poetry run split
```

The existing `<source>_split` directory is deleted and rebuilt in full. Images already present keep their previous split assignment, because assignment depends only on the image (or group) key and the seed. New images are distributed across the splits. A test image therefore never migrates into train as the dataset grows.

Changing the ratios or the seed does reassign images. Treat the combination of ratios and seed as fixed once evaluation has begun.

## Split a dataset at a non-default location

```sh
poetry run split --source data/my_dataset
```

The output is written alongside the source, in this case to `data/my_dataset_split`. To train on it, point training at the new directory:

```sh
poetry run train --data data/my_dataset_split
```
