from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "probe_corporate_event_data.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_events", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_module()


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)

    def query(self, api_name, params, fields):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _spec():
    return probe.ProbeSpec(
        api_name="event",
        fields=("ts_code", "ann_date", "direction"),
        date_field="ann_date",
        key_fields=("ts_code", "ann_date", "direction"),
        critical_fields=("ts_code", "ann_date", "direction"),
        requests=({"period": "a"}, {"period": "b"}, {"period": "c"}),
        row_limit=100,
    )


def test_probe_requires_two_nonempty_periods() -> None:
    client = _Client(
        [
            [{"ts_code": "A", "ann_date": "20190101", "direction": "UP"}],
            [],
            [{"ts_code": "B", "ann_date": "20260101", "direction": "UP"}],
        ]
    )

    result = probe.probe_spec(client, _spec())

    assert result["available"] is True
    assert result["cross_period_ready"] is True
    assert result["critical_fields_complete_in_samples"] is True
    assert result["critical_fields_usable_in_samples"] is True


def test_probe_distinguishes_permission_error_from_empty_period() -> None:
    client = _Client([RuntimeError("permission denied token=hidden"), [], []])

    result = probe.probe_spec(client, _spec())

    assert result["available"] is False
    assert result["nonempty_requests"] == 0
    assert result["critical_fields_complete_in_samples"] is False
    assert result["critical_fields_usable_in_samples"] is False
    assert result["samples"][0]["status"] == "error"


def test_row_summary_reports_duplicate_keys_and_nulls() -> None:
    rows = [
        {"ts_code": "A", "ann_date": "20260101", "direction": "UP"},
        {"ts_code": "A", "ann_date": "20260101", "direction": "UP"},
        {"ts_code": "B", "ann_date": "20260102", "direction": None},
    ]

    result = probe.summarize_rows(_spec(), rows)

    assert result["rows"] == 3
    assert result["symbols"] == 2
    assert result["duplicate_key_rows"] == 1
    assert result["critical_null_rates"]["direction"] == 1 / 3
