import csv
import io
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from flowdetectionyolo.predict_model import (
    _date_from_filename,
    build_parser,
    print_summary,
    send_ultralytics_logs_to_stderr,
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


def test_csv_output_has_one_row_per_image_and_preserves_json_fields(capsys):
    csv_file = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_file, fieldnames=("path", "date", "predictions", "error")
    )
    writer.writeheader()
    summaries = [
        {
            "path": "photos/2024_01_02_0304_CameraA_image.JPG",
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
        },
    ]

    print_summary(iter(summaries), as_json=False, csv_writer=writer)

    rows = list(csv.DictReader(io.StringIO(csv_file.getvalue(), newline="")))
    assert len(rows) == 2
    assert rows[0]["path"] == summaries[0]["path"]
    assert rows[0]["date"] == "2024-01-02 03:04"
    assert json.loads(rows[0]["predictions"]) == summaries[0]["predictions"]
    assert rows[0]["error"] == ""
    assert rows[1]["date"] == ""
    assert json.loads(rows[1]["predictions"]) == []
    assert rows[1]["error"] == summaries[1]["error"]
    assert capsys.readouterr().out


def test_parser_accepts_csv_output_path():
    args = build_parser().parse_args(["--model", "model.pt", "--csv", "results.csv"])
    assert args.csv == "results.csv"


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
