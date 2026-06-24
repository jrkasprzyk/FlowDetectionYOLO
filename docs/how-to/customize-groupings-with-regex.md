# Customize groupings with regex

This guide builds regular expressions for the `--group-regex` flag of `poetry run split` against the project's camera-trap filenames. It supplements the worked example in [Split a dataset](split-a-dataset.md#keep-related-images-in-the-same-split), which keeps a string of six or more digits together; here the patterns target the specific fields of this dataset's filenames.

## Filename format

A sample filename looks like this:

```
2023_06_04_0616_Camera_B_PICT1361.JPG
```

In other words:

```
(year) (month) (day) (time) (camera name) (picture id).JPG
```

with underscores between the fields.

These fields let you group files so that the train/val/test split is not "polluted" by related images landing on opposite sides of the boundary. Common goals:

- keep flow events together: a contiguous block of frames from a starting time to an ending time
- keep each year together
- keep each camera together: in other words, all the pictures from a particular camera would "land" on the same side of a split. This also facilitates having one camera held out for testing (see [Send a camera to a chosen split](#send-a-camera-to-a-chosen-split)), although this requires extra processing
- perform groupings that are combinations of the above, by [combining two fields](#combine-two-fields)

The sections below give a pattern for each, then cover combining fields and routing a camera to a chosen split.

## How does the expression get used to create a group identifier?

The expression is applied to each filename to produce a **group identifier**: the text of the first capture group if the expression has one, otherwise the whole matched text. Files that produce the same identifier are kept together in the same split.

The split that a group lands in is not chosen directly. The identifier is hashed, and the hash selects the split. So you control which files stay together, but not where they go. The mechanics and their consequences are covered in [Dataset splitting design](../explanation/dataset-splitting-design.md#grouping-related-images).

Two facts shape the patterns below:

- The fields sit at fixed positions, so anchoring with `^` makes the captures unambiguous and avoids accidental matches inside the picture id.
- A capture group is a single contiguous span, which matters when you want a group identifier built from two separate fields. See [Combine two fields](#combine-two-fields).

After each run, check the report line. It states how many groups formed and how many files failed to match:

```
Grouped into 412 groups; 7 images did not match the group regex and were assigned per-image
```

A nonzero unmatched count means some filenames do not fit the pattern. Fix the expression before relying on the split.

## Examples

Each pattern is read left to right against the filename. The same handful of regex pieces appears throughout:

| Piece | Meaning |
| --- | --- |
| `^` | Anchor: the match must start at the beginning of the filename. |
| `\d` | Any single digit, `0`–`9`. |
| `{n}` | Repeat the preceding piece exactly `n` times, so `\d{4}` is four digits in a row. |
| `_` | A literal underscore, matching the separator the filenames put between fields. |
| `\w` | Any single word character: a letter, digit, or underscore. |
| `( … )` | A capture group. Whatever it surrounds becomes the group identifier; everything outside it only positions the match. |

The trick in every example is to capture the fields you want to group by, while matching just enough of the surrounding text to land on them. The examples build up from the start of the filename one field at a time.

### Group by year

Capture the leading four digits:

```sh
poetry run split --group-regex "^(\d{4})"
```

Read as a sentence, `^(\d{4})` says: starting at the beginning of the filename, capture four digits. In `2023_06_04_0616_Camera_B_PICT1361.JPG` those are `2023`, the year. The `^` keeps the match pinned to the front, so the four digits can only be the year and never some run of digits later in the picture id.

Every file from 2023 shares the identifier `2023` and lands in one split; every file from 2024 shares `2024`, and so on. This forms as many groups as there are years, which is very few. With only a handful of groups, the split sizes deviate sharply from the requested ratios, and a single year can dominate one split. Year grouping is useful mainly when you deliberately want each year kept whole.

### Group by month

Extend the capture to include the month field:

```sh
poetry run split --group-regex "^(\d{4}_\d{2})"
```

The added `_\d{2}` reaches across the underscore separator and captures the two-digit month, so the capture now spans year and month together. `2023_06_04_0616_Camera_B_PICT1361.JPG` yields the identifier `2023_06`. All June 2023 frames stay together. This gives more groups than year grouping and correspondingly less lumpy splits, while still keeping seasonally similar imagery from crossing the boundary.

### Group by flow event (by day)

The filename does not contain an event identifier, so a flow event cannot be matched directly. Grouping by **day** is a conservative proxy: capture the date down to the day field.

```sh
poetry run split --group-regex "^(\d{4}_\d{2}_\d{2})"
```

One more `_\d{2}` extends the capture to the day field. `2023_06_04_0616_Camera_B_PICT1361.JPG` yields `2023_06_04`. Every frame from 4 June 2023 stays in one split.

This keeps each flow event whole, as long as the event does not span midnight. It may also bundle several distinct events from the same day into one group, which is harmless: the only requirement is that no group be split across the boundary, and a day group satisfies it. Because the day group can hold many near-identical frames, this is the strongest protection against near-duplicate leakage of the patterns here, at the cost of the lumpiest splits.

### Group by camera

Capture the camera-name field rather than the date:

```sh
poetry run split --group-regex "Camera_(\w)"
```

This pattern works differently from the date ones. `Camera_` is literal text: the regex finds that exact string wherever it occurs in the filename. The capture group then takes the single `\w` character right after it — the camera letter. `2023_06_04_0616_Camera_B_PICT1361.JPG` yields `B`.

No `^` anchor is needed here. The date patterns relied on position because a year is just digits that could appear anywhere; the camera field is found instead by its unique `Camera_` prefix, so anchoring to the front would only get in the way.

With two cameras this forms two groups, so all of camera A stays together and all of camera B stays together. Each camera lands wholesale in one split, hashed from its identifier. If you need a *specific* camera in a *specific* split — for example camera B held out for testing — see [Send a camera to a chosen split](#send-a-camera-to-a-chosen-split).

## Advanced: Combine two fields

To group by both date and camera — say, one camera on one day — the identifier must join the date field and the camera-name field. They are not adjacent in the filename (the time sits between them), and a single capture group is one contiguous span, so one expression cannot capture both at once.

Two options:

- Group by day alone (`^(\d{4}_\d{2}_\d{2})`). This already keeps each same-day event whole regardless of camera, which is usually what the combined key was meant to protect.
- If the cameras genuinely must be split apart as well, prepare the filenames upstream so the two fields are adjacent, or add an explicit group token (such as `2023_06_04_CamB`) that a single capture group can match.

## Advanced: Send a camera to a chosen split

`--group-regex` keeps a camera together but does not let you name which split it goes to; the split follows from hashing the identifier. To deterministically put, for example, camera A in train and camera B in test, partition the cameras before splitting rather than relying on the hash:

- Separate the cameras into their own source directories and run `split` on each, or assign each camera's directory directly to a split.
- Alternatively, run a normal grouped split and then move the cameras to the splits you want.

Either approach gives full control over placement; grouping by camera only guarantees that a camera is never divided.

## Choosing a grouping

| Pattern | Identifier example | Group count | Use when |
| --- | --- | --- | --- |
| `^(\d{4})` | `2023` | one per year | each year must stay whole |
| `^(\d{4}_\d{2})` | `2023_06` | one per month | seasonal similarity should not leak |
| `^(\d{4}_\d{2}_\d{2})` | `2023_06_04` | one per day | keep flow events together (recommended default) |
| `Camera_(\w)` | `B` | one per camera | each camera must stay whole |

Finer groupings (day, month) yield many small groups and splits close to the requested ratios; coarser groupings (year, camera) yield few large groups and lumpier splits. Verify the group and unmatched counts in the report before treating any split as final.

## Related

- [Split a dataset](split-a-dataset.md): ratios, the test holdout, seeds, and the original six-digit grouping example.
- [Command line reference: split](../reference/cli.md#split): the full flag listing for `poetry run split`.
- [Dataset splitting design](../explanation/dataset-splitting-design.md): why assignment is hash-based and how grouping changes the hash key.
- [Python regular expression syntax](https://docs.python.org/3/library/re.html#regular-expression-syntax): the full reference for the regex dialect `--group-regex` uses, since the split tool matches with Python's `re` module.
- [regex101](https://regex101.com/): an interactive tester for building and debugging an expression against sample filenames before running the split. Select the Python flavor to match this tool.
