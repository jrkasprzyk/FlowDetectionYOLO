import csv
import io
import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from flowdetectionyolo.predict_model import (
    _date_from_filename,
    build_parser,
    print_summary,
    predict_model,
    send_ultralytics_logs_to_stderr,
    summarize_result,
    use_utf8_stdout_on_windows,
    warn_on_collisions,
)


def results_for(*paths):
    return [SimpleNamespace(path=path) for path in paths]


def test_date_from_filename_parses_timestamp_prefix():
    assert (
        _date_from_filename("photos/2024_01_02_0304_CameraA_image.JPG")
        == "2024-01-02 03:04"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "photos/image.JPG",
        "photos/2024_13_02_0304_CameraA_image.JPG",
        "photos/2024_01_02_0304image.JPG",
    ],
)
def test_date_from_filename_returns_empty_for_invalid_prefix(filename):
    assert _date_from_filename(filename) == ""


def test_summarize_result_can_include_all_class_confidences_for_csv():
    result = SimpleNamespace(
        path="image.jpg",
        names={0: "flow", 1: "noflow", 2: "obstructed"},
        probs=SimpleNamespace(
            top5=[0, 1, 2],
            top5conf=[0.8, 0.15, 0.05],
            data=[0.8, 0.15, 0.05],
        ),
    )

    summary = summarize_result(result, top_k=1, include_class_confidences=True)

    assert summary["_class_confidences"] == {0: 0.8, 1: 0.15, 2: 0.05}
    assert len(summary["predictions"]) == 1


def test_csv_output_has_a_confidence_column_per_class(capsys):
    csv_file = io.StringIO(newline="")
    class_names = [(0, "flow"), (1, "noflow"), (2, "obstructed")]
    fieldnames = [
        "path",
        "date",
        *(f"{index}: {name}" for index, name in class_names),
        "predicted class",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    summaries = [
        {
            "path": "photos/2024_01_02_0304_CameraA_image.JPG",
            "_class_confidences": {0: 0.8, 1: 0.15, 2: 0.05},
            "predictions": [
                {
                    "rank": 1,
                    "class_index": 7,
                    "class_name": "flow, fast",
                    "confidence": 0.875,
                }
            ],
        },
        {
            "path": "photos/not_a_timestamp.JPG",
            "predictions": [],
            "error": "No classification probabilities were returned.",
            "_class_confidences": {},
        },
    ]

    print_summary(
        iter(summaries), as_json=False, csv_writer=writer, class_names=class_names
    )

    rows = list(csv.DictReader(io.StringIO(csv_file.getvalue(), newline="")))
    assert len(rows) == 2
    assert list(rows[0]) == fieldnames
    assert rows[0]["path"] == summaries[0]["path"]
    assert rows[0]["date"] == "2024-01-02 03:04"
    assert rows[0]["0: flow"] == "0.8"
    assert rows[0]["1: noflow"] == "0.15"
    assert rows[0]["2: obstructed"] == "0.05"
    assert rows[0]["predicted class"] == "flow"
    assert rows[1]["date"] == ""
    assert rows[1]["0: flow"] == ""
    assert rows[1]["predicted class"] == ""
    assert capsys.readouterr().out


def test_combined_csv_and_json_output_keeps_private_confidences_out_of_json(capsys):
    csv_file = io.StringIO(newline="")
    class_names = [(0, "flow")]
    writer = csv.DictWriter(
        csv_file,
        fieldnames=("path", "date", "0: flow", "predicted class"),
    )
    writer.writeheader()
    summary = {
        "path": "image.jpg",
        "predictions": [{"rank": 1, "class_name": "flow"}],
        "_class_confidences": {0: 1.0},
    }

    print_summary([summary], as_json=True, csv_writer=writer, class_names=class_names)

    assert json.loads(capsys.readouterr().out) == [
        {"path": "image.jpg", "predictions": [{"rank": 1, "class_name": "flow"}]}
    ]


def test_parser_accepts_csv_output_path():
    args = build_parser().parse_args(["--model", "model.pt", "--csv", "results.csv"])
    assert args.csv == "results.csv"


def test_predict_model_writes_dynamic_class_columns(monkeypatch, tmp_path, capsys):
    class FakeModel:
        names = {0: "flow", 1: "noflow", 2: "obstructed"}

        def __init__(self, weights):
            assert weights == "model.pt"

        def predict(self, **kwargs):
            assert kwargs["stream"] is True
            return iter(
                [
                    SimpleNamespace(
                        path="2024_01_02_0304_camera.jpg",
                        names=self.names,
                        probs=SimpleNamespace(
                            top5=[0, 1, 2],
                            top5conf=[0.8, 0.15, 0.05],
                            data=[0.8, 0.15, 0.05],
                        ),
                    )
                ]
            )

    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = FakeModel
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    csv_path = tmp_path / "predictions.csv"

    assert (
        predict_model(
            "model.pt",
            "photos",
            640,
            1,
            False,
            None,
            None,
            False,
            False,
            False,
            csv_path=csv_path,
        )
        == 0
    )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == [
            "path",
            "date",
            "0: flow",
            "1: noflow",
            "2: obstructed",
            "predicted class",
        ]
        row = next(reader)
    assert row["2: obstructed"] == "0.05"
    assert row["predicted class"] == "flow"
    assert capsys.readouterr().out


def run_warnings(capsys, *paths):
    results = results_for(*paths)
    assert list(warn_on_collisions(results)) == results
    return capsys.readouterr()


def test_distinct_names_do_not_warn(capsys):
    captured = run_warnings(capsys, "a/one.jpg", "a/two.jpg")
    assert captured.err == ""


def test_same_name_in_different_folders_warns_on_stderr(capsys):
    captured = run_warnings(capsys, "a/one.jpg", "b/one.jpg")
    assert "b/one.jpg" in captured.err
    assert "a/one.jpg" in captured.err
    assert "1 saved output(s)" in captured.err
    assert captured.out == ""


def test_same_stem_with_different_extensions_warns(capsys):
    captured = run_warnings(capsys, "a/one.png", "a/one.jpg")
    assert "1 saved output(s)" in captured.err


def test_video_frames_sharing_a_path_do_not_warn(capsys):
    captured = run_warnings(capsys, "a/clip.mp4", "a/clip.mp4", "a/clip.mp4")
    assert captured.err == ""


def test_image_and_video_with_same_name_do_not_warn(capsys):
    captured = run_warnings(capsys, "photos/clip.jpg", "videos/clip.mp4")
    assert captured.err == ""


def test_two_videos_with_same_name_warn(capsys):
    captured = run_warnings(capsys, "a/clip.mp4", "b/clip.mov")
    assert "1 saved output(s)" in captured.err


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_case_only_difference_warns_on_case_insensitive_platforms(
    capsys, monkeypatch, platform
):
    monkeypatch.setattr(sys, "platform", platform)
    captured = run_warnings(capsys, "a/One.png", "b/one.jpg")
    assert "1 saved output(s)" in captured.err


def test_case_only_difference_does_not_warn_on_linux(capsys, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    captured = run_warnings(capsys, "a/One.png", "b/one.jpg")
    assert captured.err == ""


def test_each_overwrite_is_counted(capsys):
    captured = run_warnings(capsys, "a/one.jpg", "b/one.jpg", "c/one.jpg")
    assert "2 saved output(s)" in captured.err


def test_results_stream_without_being_collected(capsys):
    def source():
        yield SimpleNamespace(path="a/one.jpg")
        pytest.fail("the next result was requested too early")

    assert next(warn_on_collisions(source())).path == "a/one.jpg"


@pytest.mark.parametrize(
    ("platform", "expected"), [("win32", "utf-8"), ("linux", "cp1252")]
)
def test_stdout_encoding_changes_only_on_windows(monkeypatch, platform, expected):
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "platform", platform)
    use_utf8_stdout_on_windows()
    assert stdout.encoding == expected


@pytest.fixture
def ultralytics_logger():
    logger = logging.getLogger("ultralytics")
    original_handlers = logger.handlers[:]
    logger.handlers[:] = []
    yield logger
    logger.handlers[:] = original_handlers


def test_stdout_log_handler_moves_to_stderr(ultralytics_logger):
    handler = logging.StreamHandler(sys.stdout)
    ultralytics_logger.addHandler(handler)
    send_ultralytics_logs_to_stderr()
    assert handler.stream is sys.stderr


def test_other_log_handlers_are_left_alone(ultralytics_logger, tmp_path):
    handler = logging.FileHandler(tmp_path / "run.log")
    ultralytics_logger.addHandler(handler)
    original_stream = handler.stream
    send_ultralytics_logs_to_stderr()
    assert handler.stream is original_stream
    handler.close()
