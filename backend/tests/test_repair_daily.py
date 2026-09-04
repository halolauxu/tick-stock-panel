"""日 K 修复区间契约。"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from app.jobs import daily_pipeline
from app.services import data_integrity, pipeline_jobs, repair_daily
from app.services.repair_daily import run_repair_daily


def test_repair_daily_honors_fixed_end_date(monkeypatch):
    captured = {}

    def fake_run_now(repo, capset, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(daily_pipeline, "run_now", fake_run_now)
    start = date(2026, 9, 3)

    result = run_repair_daily(object(), object(), start, end_date=start)

    assert result == {"ok": True}
    assert captured["override_start_date"] == start
    assert captured["target_date"] == start


def test_repair_daily_manual_range_still_defaults_to_current_day(monkeypatch):
    captured = {}

    def fake_run_now(repo, capset, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(daily_pipeline, "run_now", fake_run_now)
    start = date(2026, 9, 1)

    result = run_repair_daily(object(), object(), start)

    assert result == {"ok": True}
    assert captured["override_start_date"] == start
    assert captured["target_date"] is None


def test_integrity_repair_fixes_only_confirmed_issue_day(monkeypatch):
    captured = {}

    class FakeStore:
        def create(self):
            return "repair-job", True

        def start(self, job_id):
            return None

        def progress(self, *args, **kwargs):
            return None

        def succeed(self, job_id, result):
            return None

        def fail(self, job_id, error):
            raise AssertionError(error)

    class ImmediateThread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    def fake_repair(repo, capset, start_date, **kwargs):
        captured.update({"start_date": start_date, **kwargs})
        return {"ok": True}

    monkeypatch.setattr(pipeline_jobs, "job_store", FakeStore())
    monkeypatch.setattr(pipeline_jobs, "try_acquire_run_slot", lambda owner="": True)
    monkeypatch.setattr(pipeline_jobs, "release_run_slot", lambda owner=None: None)
    monkeypatch.setattr(repair_daily, "run_repair_daily", fake_repair)
    monkeypatch.setattr("threading.Thread", ImmediateThread)
    target = date(2026, 9, 3)
    state = SimpleNamespace(
        repo=SimpleNamespace(store=SimpleNamespace(data_dir=None)),
        capabilities=SimpleNamespace(has=lambda cap: True),
        quote_service=None,
    )

    job_id, is_new = data_integrity.launch_integrity_repair(
        state,
        target,
        "test",
    )

    assert (job_id, is_new) == ("repair-job", True)
    assert captured["start_date"] == target
    assert captured["end_date"] == target
