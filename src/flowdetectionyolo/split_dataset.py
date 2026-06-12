"""Split a classification dataset into train/val/test directories.

Mirrors ultralytics.data.split.split_classify_dataset (copies files into a new
'{source}_split' directory, preserving class folders) but adds an optional test
holdout, which the ultralytics version does not support.

Split assignment is based on a hash of each image's path relative to the source
directory, not a shuffle. An image's split is therefore stable when the dataset
grows: re-running the split after adding new images never moves an existing
image between splits, so the test holdout stays uncontaminated. The trade-off
is that split sizes only approximate the requested ratios.

With --group-regex, related images (e.g. shots from the same photo set, or
frames from the same camera and date) are hashed by a group id extracted from
the filename instead of per-image, so a group never straddles train/val/test.
This avoids testing on near-duplicates of training images at the cost of
lumpier split sizes.
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path

# Treat float ratio sums within this tolerance of 1.0 as exactly 1.0.
RATIO_TOLERANCE = 1e-6

# Number of hex digits in an MD5 digest; scales the digest to a [0, 1) fraction.
MD5_HEX_DIGITS = 32


def assign_split(key, train_ratio, val_ratio, has_test, seed=None):
    """Deterministically assign an image to a split from a hash of its key."""
    if seed is not None:
        key = f"{seed}:{key}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    fraction = int(digest, 16) / 16**MD5_HEX_DIGITS
    if fraction < train_ratio:
        return "train"
    if not has_test or fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def split_key(img, source_path, group_pattern):
    """Hash key for an image: its group id when the filename matches the group
    pattern (first capture group, or the whole match), else its relative path.

    Returns (key, matched) so callers can report how many images fell back to
    per-image assignment rather than silently losing the grouping guarantee.
    """
    if group_pattern is not None:
        match = group_pattern.search(img.name)
        if match:
            key = match.group(1) if match.groups() else match.group(0)
            return key, True
    return img.relative_to(source_path).as_posix(), False


def split_dataset(source, train_ratio, val_ratio, seed=None, group_regex=None):
    source_path = Path(source)
    if not source_path.is_dir():
        print(f"Source directory not found: {source_path}")
        return 1

    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"No class directories found in {source_path}")
        return 1

    group_pattern = re.compile(group_regex) if group_regex else None

    test_ratio = 1.0 - train_ratio - val_ratio
    has_test = test_ratio > RATIO_TOLERANCE
    split_names = ["train", "val", "test"] if has_test else ["train", "val"]

    # Regenerate from scratch: stale copies from a previous run with different
    # ratios or seed would otherwise leave an image in two splits at once.
    split_path = Path(f"{source_path}_split")
    if split_path.is_dir():
        shutil.rmtree(split_path)
    counts = dict.fromkeys(split_names, 0)
    groups = set()
    ungrouped = 0

    for class_dir in class_dirs:
        for split_name in split_names:
            (split_path / split_name / class_dir.name).mkdir(parents=True, exist_ok=True)

        for img in sorted(p for p in class_dir.glob("*.*") if p.is_file()):
            key, matched = split_key(img, source_path, group_pattern)
            if matched:
                groups.add(key)
            else:
                ungrouped += 1
            split_name = assign_split(key, train_ratio, val_ratio, has_test, seed)
            shutil.copy2(img, split_path / split_name / class_dir.name / img.name)
            counts[split_name] += 1

    summary = ", ".join(f"{name}: {count}" for name, count in counts.items())
    print(f"Split dataset written to {split_path} ({summary})")
    if group_pattern is not None:
        print(f"Grouped into {len(groups)} groups; {ungrouped} images did not match the group regex and were assigned per-image")
    return 0


def parse_regex(value):
    try:
        re.compile(value)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regex {value!r}: {exc}")
    return value


def parse_ratio(value):
    try:
        ratio = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}")
    if not 0 < ratio < 1:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1 (exclusive)")
    return ratio


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Split a classification dataset into train/val/test directories. "
            "The test fraction is whatever remains after train and val. "
            "Assignment is hash-based, so existing images keep their split "
            "when the dataset grows."
        )
    )
    parser.add_argument(
        "--source",
        default="data/classification_test",
        help="Classification dataset root directory (one subdirectory per class).",
    )
    parser.add_argument(
        "--train-ratio",
        type=parse_ratio,
        default=0.8,
        help="Approximate fraction of images assigned to the train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=parse_ratio,
        default=0.1,
        help="Approximate fraction of images assigned to the val split; the rest become the test split.",
    )
    parser.add_argument(
        "--group-regex",
        type=parse_regex,
        default=None,
        help=(
            "Optional regex applied to each filename; images sharing the match "
            "(first capture group if present) always land in the same split, so "
            "related images never straddle train/val/test. Non-matching files "
            "fall back to per-image assignment. Example: \"(\\d{6,})\" groups "
            "this repo's images by their numeric set id."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional salt mixed into the assignment hash. Changing it produces a "
            "different (still stable) split; omit it to keep one canonical split."
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.train_ratio + args.val_ratio > 1 + RATIO_TOLERANCE:
        parser.error("train ratio plus val ratio must not exceed 1")
    return split_dataset(args.source, args.train_ratio, args.val_ratio, args.seed, args.group_regex)


if __name__ == "__main__":
    raise SystemExit(main())
