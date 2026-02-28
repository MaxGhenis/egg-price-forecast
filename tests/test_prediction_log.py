"""Tests for log_prediction in export_data (TDD — function not yet implemented)."""

import json
from datetime import datetime
from pathlib import Path

import pytest

# export_data.py lives at the repo root, not inside the egg_price_forecast package.
# pytest is invoked from the project root (per pyproject.toml testpaths), so
# the root is already on sys.path and the import resolves without manipulation.
from export_data import log_prediction


SAMPLE_DATA = {
    "model_version": "lgbm-quantile-v1",
    "current_price": 4.95,
    "raw_p_up": 0.62,
    "calibrated_p_up": 0.58,
    "base_rate": 0.55,
    "point_forecast": 5.10,
    "intervals": {"50": [4.80, 5.40], "90": [4.20, 6.10]},
    "backtest": {"direction_accuracy": 0.63, "brier_score": 0.21},
}


class TestLogPredictionFileCreation:
    """Log file and directory are created on demand."""

    def test_log_file_is_created_when_missing(self, tmp_path):
        """Calling log_prediction should create the log file if it doesn't exist."""
        log_dir = tmp_path / "predictions"
        assert not log_dir.exists()

        returned_path = log_prediction(SAMPLE_DATA, log_dir=log_dir)

        assert returned_path.exists()

    def test_returns_path_to_log_file(self, tmp_path):
        """Return value is the path to the jsonl log file."""
        log_dir = tmp_path / "predictions"
        returned_path = log_prediction(SAMPLE_DATA, log_dir=log_dir)

        assert returned_path == log_dir / "log.jsonl"

    def test_directory_is_created_if_missing(self, tmp_path):
        """log_prediction creates the log directory when it doesn't exist."""
        log_dir = tmp_path / "nested" / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        assert log_dir.is_dir()


class TestLogPredictionAppend:
    """Multiple calls append entries rather than overwriting."""

    def test_two_entries_both_present(self, tmp_path):
        """Writing two entries should result in two lines in the log file."""
        log_dir = tmp_path / "predictions"

        log_prediction(SAMPLE_DATA, log_dir=log_dir)
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        log_file = log_dir / "log.jsonl"
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_existing_entries_are_not_overwritten(self, tmp_path):
        """A second call must not erase the first entry."""
        log_dir = tmp_path / "predictions"

        first_data = {**SAMPLE_DATA, "model_version": "first-run"}
        second_data = {**SAMPLE_DATA, "model_version": "second-run"}

        log_prediction(first_data, log_dir=log_dir)
        log_prediction(second_data, log_dir=log_dir)

        log_file = log_dir / "log.jsonl"
        lines = [l for l in log_file.read_text().splitlines() if l.strip()]
        versions = [json.loads(l)["model_version"] for l in lines]
        assert "first-run" in versions
        assert "second-run" in versions


class TestLogPredictionJsonValidity:
    """Each line must be valid, parseable JSON."""

    def test_each_entry_is_valid_json(self, tmp_path):
        """json.loads should succeed on every line in the log file."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        log_file = log_dir / "log.jsonl"
        for line in log_file.read_text().splitlines():
            if line.strip():
                # Should not raise
                json.loads(line)

    def test_one_json_object_per_line(self, tmp_path):
        """Each non-empty line parses to a dict."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        log_file = log_dir / "log.jsonl"
        for line in log_file.read_text().splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert isinstance(parsed, dict)


class TestLogPredictionRequiredKeys:
    """Logged entries must contain the required metadata and data keys."""

    def test_timestamp_key_present(self, tmp_path):
        """Entry must have a 'timestamp' key."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        assert "timestamp" in entry

    def test_git_commit_key_present(self, tmp_path):
        """Entry must have a 'git_commit' key."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        assert "git_commit" in entry

    def test_model_version_key_present(self, tmp_path):
        """Entry must contain 'model_version' from the data dict."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        assert "model_version" in entry

    def test_all_data_keys_present(self, tmp_path):
        """All keys from the data dict must appear in the log entry."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        for key in SAMPLE_DATA:
            assert key in entry, f"Missing key in log entry: {key}"

    def test_data_values_are_preserved(self, tmp_path):
        """Values from the data dict must be written unchanged."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        assert entry["model_version"] == SAMPLE_DATA["model_version"]
        assert entry["current_price"] == pytest.approx(SAMPLE_DATA["current_price"])


class TestLogPredictionTimestamp:
    """Timestamp must be a valid ISO 8601 datetime string."""

    def test_timestamp_is_valid_iso_format(self, tmp_path):
        """datetime.fromisoformat should parse the timestamp without raising."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        # Should not raise
        dt = datetime.fromisoformat(entry["timestamp"])
        assert isinstance(dt, datetime)

    def test_timestamp_is_recent(self, tmp_path):
        """Timestamp should be within a few seconds of now."""
        from datetime import timezone

        before = datetime.now(timezone.utc)
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)
        after = datetime.now(timezone.utc)

        entry = _read_first_entry(log_dir)
        ts = datetime.fromisoformat(entry["timestamp"])

        assert before <= ts <= after


class TestLogPredictionGitCommit:
    """git_commit field behaviour."""

    def test_git_commit_is_string(self, tmp_path):
        """git_commit must be a non-empty string."""
        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        assert isinstance(entry["git_commit"], str)
        assert len(entry["git_commit"]) > 0

    def test_git_commit_falls_back_to_unknown(self, tmp_path, mocker):
        """When git is unavailable, git_commit should be 'unknown'."""
        mocker.patch("subprocess.check_output", side_effect=Exception("no git"))

        log_dir = tmp_path / "predictions"
        log_prediction(SAMPLE_DATA, log_dir=log_dir)

        entry = _read_first_entry(log_dir)
        assert entry["git_commit"] == "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_first_entry(log_dir: Path) -> dict:
    """Return the first parsed JSON entry from log_dir/log.jsonl."""
    log_file = log_dir / "log.jsonl"
    first_line = next(
        l for l in log_file.read_text().splitlines() if l.strip()
    )
    return json.loads(first_line)
