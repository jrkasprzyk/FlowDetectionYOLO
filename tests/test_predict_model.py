import logging
import sys
from types import SimpleNamespace

import pytest

from flowdetectionyolo.predict_model import (
    send_ultralytics_logs_to_stderr,
    warn_on_overwrites,
)


def results_for(*paths):
    return [SimpleNamespace(path=path) for path in paths]


def run_warnings(capsys, *paths):
    results = results_for(*paths)
    assert list(warn_on_overwrites(results)) == results
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


def test_each_overwrite_is_counted(capsys):
    captured = run_warnings(capsys, "a/one.jpg", "b/one.jpg", "c/one.jpg")
    assert "2 saved output(s)" in captured.err


def test_results_stream_without_being_collected(capsys):
    def source():
        yield SimpleNamespace(path="a/one.jpg")
        pytest.fail("the next result was requested too early")

    assert next(warn_on_overwrites(source())).path == "a/one.jpg"


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
