# Dataset splitting design

This page explains why the split tool assigns images to train, val, and test by hashing rather than by shuffling, and what follows from that choice.

## The problem with shuffling

The conventional way to split a dataset is to shuffle the file list and cut it at the ratio boundaries. This produces exact split sizes, but the assignment of any particular image depends on the entire list: adding one image changes the shuffled order, and images move between splits on every regeneration.

For a dataset that grows over time, this is a problem for evaluation. A model is trained, images are added, the split is regenerated, and images that were previously in the test set are now in the training set. Metrics computed on the new test set are no longer comparable with earlier metrics, and any image that has crossed from test into train contaminates the evaluation of models trained after the move.

Fixing a random seed does not solve this; the assignment still depends on the position of every file in the list, so insertion of a new file still reshuffles assignments.

## Hash-based assignment

The split tool instead computes an MD5 hash of a key derived from each image and scales the digest to a fraction in [0, 1). The fraction determines the split: values below the train ratio go to train, values below train plus val go to val, and the remainder go to test.

The key is the image's path relative to the source directory (or a group identifier; see below). The assignment of an image therefore depends only on its own key and the optional seed. It does not depend on which other images exist. Consequences:

- **Stability under growth.** Re-running the split after adding images never moves an existing image between splits. The test holdout established at the start of a project remains uncontaminated for its lifetime.
- **No state.** Stability requires no record of previous assignments. The split directory can be deleted and rebuilt at any time, and any collaborator who runs the tool on the same source data with the same arguments obtains the same split.
- **Approximate ratios.** Split sizes follow the requested ratios only in expectation. With small datasets the deviation is proportionally larger. This is the price of per-image independence: exact sizes would require ranking images against each other, which reintroduces dependence on the whole list.

MD5 is used here as a uniform deterministic function, not for any security property. Cryptographic weakness is irrelevant to this use.

Two operations do reassign images and should be treated as breaking changes to the evaluation setup: changing the ratios, and changing the seed. Both alter the mapping from keys to splits. The seed exists precisely to produce an alternative split when one is wanted, for example to check that a result is not an artifact of one particular partition.

## Grouping related images

Per-image assignment has a failure mode of its own: near-duplicates. Two photographs of the same scene are distinct files with distinct hashes, and nothing prevents one landing in train and the other in test. A model can then score well on the test image by having memorized its near-duplicate during training, which inflates metrics without any genuine generalization.

The `--group-regex` option addresses this by changing what is hashed. When a filename matches the expression, the match (or its first capture group) replaces the relative path as the hash key. All images sharing a group identifier share a key, hash to the same fraction, and land in the same split. A group can never straddle a split boundary.

The cost is again in ratio accuracy: assignment now moves whole groups, so split sizes deviate further from the requested ratios when groups are large or unevenly sized.

Files that do not match the expression fall back to per-image assignment rather than being excluded. The tool reports the number of groups and the number of unmatched files so that an incorrect expression is visible rather than silent.

## Relationship to the ultralytics splitter

Ultralytics provides `ultralytics.data.split.split_classify_dataset`, which copies class folders into a new `<source>_split` directory in the same layout this tool produces. This tool exists because the ultralytics version supports only a train/val division, with no test holdout, and assigns by shuffling rather than hashing. The output layout is kept compatible so that ultralytics training consumes the result directly.

## Summary of trade-offs

| Property | Shuffle split | Hash split (this tool) |
| --- | --- | --- |
| Exact split sizes | yes | no, approximate |
| Stable as dataset grows | no | yes |
| Requires stored state for stability | yes | no |
| Test holdout protected over time | no | yes |
| Near-duplicate leakage protection | no | yes, with `--group-regex` |
