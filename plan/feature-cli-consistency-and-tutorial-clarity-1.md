---
goal: Resolve Issue #1 — CLI flag consistency (ultralytics alignment) and tutorial/reference documentation clarity
version: 2.0
date_created: 2026-07-01
last_updated: 2026-07-01
owner: Joseph Kasprzyk
status: 'Planned'
tags: [documentation, cli, refactor, bug]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan resolves [Issue #1](https://github.com/jrkasprzyk/FlowDetectionYOLO/issues/1): a CLI flag consistency audit plus documentation clarity fixes in the tutorial and reference.

Audit finding: the `--source` vs `--data` distinction the issue flagged already matches ultralytics semantics (`source` = arbitrary inference input for `predict`; `data` = dataset spec for `train`/`val`), so it needs documentation, not renaming. The genuine divergences from ultralytics terminology are (1) `split --source`, where ultralytics' own helper `split_classify_dataset(source_dir=...)` uses `source_dir`, and (2) `--weights`, where ultralytics names the checkpoint argument `model` (`yolo predict model=best.pt`).

Decision (breaking changes accepted): rename `split --source` → `--source-dir` and rename `--weights` → `--model` on `predict` and `eval`. The `train` command keeps `--model-config` + `--weights` unchanged (its arch-yaml-vs-pretrained-weights split is intentional and out of scope). Documentation is updated for both the renames and the tutorial clarity gaps.

## 1. Requirements & Constraints

- **REQ-001**: `split` accepts `--source-dir` (was `--source`), matching ultralytics `split_classify_dataset(source_dir=)`.
- **REQ-002**: `predict` and `eval` accept `--model` (was `--weights`), matching ultralytics' `model=` checkpoint argument.
- **REQ-003**: `predict --source` is unchanged (correct ultralytics inference-input term).
- **REQ-004**: `train` flags are unchanged: `--model-config`, `--weights`, `--data`, etc.
- **REQ-005**: Tutorial Prerequisites must state the dataset may live anywhere; `data/classification_test` is only the default; the flag to point elsewhere is `--source-dir`.
- **REQ-006**: Tutorial command examples must show the relevant dataset/checkpoint flag per step and briefly explain each flag shown.
- **REQ-007**: Tutorial Step 4 (predict) must state where results go (stdout by default; files only under `runs/` when `--save` is passed) and clarify which flags share meaning across commands.
- **REQ-008**: Tutorial Step 4/5 front matter must clearly contrast `predict` (per-image guesses over a flat source) with `eval` (accuracy over the labeled `<split>/<class>/<image>` layout).
- **REQ-009**: Tutorial must link to `docs/reference/cli.md` at least once per relevant step.
- **REQ-010**: `docs/reference/cli.md` must document the `--source` vs `--data` rule so the ultralytics-inherited distinction is discoverable.
- **CON-001**: Renames are breaking with no backward-compatible aliases (see ALT-002); old flags will error via argparse "unrecognized arguments".
- **CON-002**: All example outputs, defaults, and paths in docs must match current source behavior after edits.
- **GUD-001**: Follow the Diátaxis structure already used (`docs/README.md`): tutorial stays learning-oriented, reference stays descriptive.
- **PAT-001**: argparse converts `--source-dir` to dest `source_dir` automatically; update `main()` accordingly.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Rename the CLI flags in source (`src/flowdetectionyolo/`).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | `split_dataset.py`: in `build_parser`, change `--source` to `--source-dir` (keep `default="data/classification_test"` and help text, updating the help wording to say "source directory"). In `main`, change `args.source` to `args.source_dir` in the `split_dataset(...)` call. Function `split_dataset` signature and internal variable names may stay (`source` param) since they are not user-facing. | | |
| TASK-002 | `predict_model.py`: in `build_parser`, change `--weights` to `--model` (keep `default="from_train_2026-04-28.pt"`; update help to "Trained YOLO classification model checkpoint."). In `main`, change `args.weights` to `args.model`. Internal `predict_model(weights, ...)` param may stay. | | |
| TASK-003 | `eval_model.py`: same as TASK-002 — `--weights` → `--model`, `args.weights` → `args.model`, help text updated. | | |
| TASK-004 | Confirm `train_model.py` is left unchanged (still `--model-config` and `--weights`). No edit; recorded for traceability. | | |

### Implementation Phase 2

- GOAL-002: Update the reference (`docs/reference/cli.md`) for renames and the source/data rule.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | In the `## split` section: rename the `--source` row and the prose ("Reads `--source`, writes `<source>_split`") to `--source-dir`. | | |
| TASK-006 | In the `## predict` and `## eval` sections: rename each `--weights` row to `--model` (keep defaults/descriptions). Leave `## train`'s `--weights` and `--model-config` rows untouched. | | |
| TASK-007 | Add a subsection after the intro paragraph, before `## split`, titled "Dataset inputs: `--source` vs `--data`", stating: `--source` is a raw input to process (`split --source-dir` is the unsplit class-foldered dataset; `predict --source` is an arbitrary inference source — image, dir, glob, URL, video); `--data` is a prepared/split dataset root fed to the model (`train`, `eval`), mirroring ultralytics' `data` argument. Note that the checkpoint argument is `--model` on predict/eval (matching ultralytics) while `train` uses `--model-config` + `--weights`. | | |

### Implementation Phase 3

- GOAL-003: Clarify the tutorial (`docs/tutorials/first-training-run.md`).

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-008 | Prerequisites: revise the paragraph after the tree so it leads with "your dataset can live in any directory; `data/classification_test` is only the default", and point elsewhere with `split --source-dir <path>`. Add an inline link to `../reference/cli.md`. | | |
| TASK-009 | Step 2 (split): annotate the example to note it reads `--source-dir` (default `data/classification_test`) and that a custom location uses `poetry run split --source-dir <path>`. Cross-link `../reference/cli.md#split`. | | |
| TASK-010 | Step 3 (train): note `--data` selects the pre-split dataset (defaulting from cfg) and that train's starting checkpoint is `--weights` (unlike predict/eval's `--model`). Keep existing `--epochs`/`--device` notes. Cross-link `../reference/cli.md#train`. | | |
| TASK-011 | Step 4 (predict): update the two examples to `--model` (was `--weights`). Rewrite the front matter to explain: predict prints per-image top-k guesses to stdout and writes no files unless `--save` is given; `--source` is an inference source (single image, flat directory, glob, URL, video), distinct from `split --source-dir` (class-foldered dataset); `--model`/`--imgsz`/`--device` share meaning across commands. Cross-link `../reference/cli.md#predict`. | | |
| TASK-012 | Step 4: add a sentence stating saved outputs (when `--save` is passed) land under `runs/`, resolved to an absolute path, so the "where did results go" confusion is resolved. | | |
| TASK-013 | Step 5 (eval): update the example to `--model`. Expand the front matter to contrast eval with predict: eval reads the `<split>/<class>/<image>` layout, uses class folders as ground truth, and reports top1/top5 accuracy plus a confusion matrix, which predict cannot do. Cross-link `../reference/cli.md#eval`. | | |
| TASK-014 | Verify "Where to go next" still reads coherently after per-step links are added; adjust wording if the cli.md link now duplicates awkwardly. | | |

### Implementation Phase 4

- GOAL-004: Update remaining docs and validate.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-015 | `docs/how-to/run-a-smoke-test.md`: change `--weights` to `--model` on the `predict` (line ~49) and `eval` (line ~57) example commands. `--source`/`--data` there are unchanged. | | |
| TASK-016 | `docs/how-to/split-a-dataset.md`: change `poetry run split --source data/my_dataset` (line ~78) to `--source-dir`. Scan the rest of the file for other `--source` split references. | | |
| TASK-017 | Runtime check: `poetry run split --help`, `poetry run predict --help`, `poetry run eval --help`, `poetry run train --help` all succeed and show the renamed/retained flags. | | |
| TASK-018 | Grep the whole repo for residual `--source`/`--weights` (excluding `train`'s legitimate `--weights` and this plan file) to confirm no stale references remain. | | |
| TASK-019 | Commit on branch `fix-cli-consistency-and-tutorial-clarity` referencing Issue #1. Do not push or open a PR until the user requests it. | | |

## 3. Alternatives

- **ALT-001**: Docs-only, keep all flags. Rejected by user decision: they want stricter ultralytics alignment for the checkpoint arg and the split source directory.
- **ALT-002**: Add deprecation aliases (accept both old and new flag names with a warning) instead of a hard break. Not chosen: the repo is early-stage, the issue explicitly permits breaking changes, and aliases add parser complexity. Revisit only if external scripts depend on the old names.
- **ALT-003**: Also rework `train` to a single ultralytics-style `model` argument. Rejected by user scope decision: train's `--model-config` (arch yaml) vs `--weights` (pretrained checkpoint) split is intentional; collapsing it is the most invasive change and out of scope.

## 4. Dependencies

- **DEP-001**: None added. `poetry run <command>` entry points in `pyproject.toml` are unchanged (only flag names inside parsers change).

## 5. Files

- **FILE-001**: `src/flowdetectionyolo/split_dataset.py` — `--source` → `--source-dir` (TASK-001).
- **FILE-002**: `src/flowdetectionyolo/predict_model.py` — `--weights` → `--model` (TASK-002).
- **FILE-003**: `src/flowdetectionyolo/eval_model.py` — `--weights` → `--model` (TASK-003).
- **FILE-004**: `src/flowdetectionyolo/train_model.py` — unchanged (TASK-004, traceability only).
- **FILE-005**: `docs/reference/cli.md` — rename rows + source/data subsection (TASK-005–007).
- **FILE-006**: `docs/tutorials/first-training-run.md` — renames + clarity edits (TASK-008–014).
- **FILE-007**: `docs/how-to/run-a-smoke-test.md` — `--weights` → `--model` (TASK-015).
- **FILE-008**: `docs/how-to/split-a-dataset.md` — `--source` → `--source-dir` (TASK-016).
- **FILE-009**: `plan/feature-cli-consistency-and-tutorial-clarity-1.md` — this plan.

## 6. Testing

- **TEST-001**: `--help` for all four commands runs without error and lists the correct flags (TASK-017).
- **TEST-002**: Repo-wide grep shows no stale `--source`/`--weights` outside train and this plan (TASK-018).
- **TEST-003**: Doc accuracy — flag names, defaults, and example outputs match the edited parsers (`src/flowdetectionyolo/*.py`).
- **TEST-004**: Link check — every added/edited markdown link resolves to an existing file and heading anchor.
- **TEST-005**: Diátaxis coherence — tutorial stays task-sequential; reference stays descriptive.

## 7. Risks & Assumptions

- **RISK-001**: Residual inconsistency — after renaming, `predict`/`eval` use `--model` for the checkpoint while `train` still uses `--weights`. Accepted per ALT-003; documented in cli.md (TASK-007) and the tutorial (TASK-010) so the difference is explicit.
- **RISK-002**: Breaking change — any existing user scripts or shell history using `--source` (split) or `--weights` (predict/eval) will fail with "unrecognized arguments". Acceptable per CON-001; the issue sanctioned breaking changes.
- **RISK-003**: Over-documenting the tutorial could reduce its flow. Mitigation: keep per-step flag notes to one or two sentences; defer detail to cli.md via links.
- **ASSUMPTION-001**: No test suite exists (`tests/` absent), so validation is via `--help` runs, grep, and manual review rather than automated tests.
- **ASSUMPTION-002**: Default values (`from_train_2026-04-28.pt`, `data/classification_test`, etc.) are intentional and unchanged.
- **ASSUMPTION-003**: Example outputs (counts, accuracies) in the tutorial are representative and need no regeneration.

## 8. Related Specifications / Further Reading

- [Issue #1](https://github.com/jrkasprzyk/FlowDetectionYOLO/issues/1)
- ultralytics: `model.predict(source=)`, `model.train(data=)`, `model.val(data=, split=)`, `split_classify_dataset(source_dir=)`
- [docs/reference/cli.md](../docs/reference/cli.md)
- [docs/tutorials/first-training-run.md](../docs/tutorials/first-training-run.md)
- [Diátaxis framework](https://diataxis.fr/)
