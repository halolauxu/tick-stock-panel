"""Launch one persistent Alpha research job from an existing frozen run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.alpha_mining.registry import load_builtin_registry  # noqa: E402
from app.services.alpha_mining_manager import AlphaMiningJobManager  # noqa: E402

TERMINAL_STATUSES = frozenset(
    {"succeeded", "succeeded_with_budget_exhausted", "failed", "cancelled"}
)


def prepare_run(
    source_manifest: Mapping[str, Any],
    engine_ids: Sequence[str],
    engine_versions: Mapping[str, str],
    *,
    max_candidates_per_engine: int,
    max_trials_per_engine: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = dict(source_manifest["request"])
    request["engine_ids"] = list(engine_ids)
    request["max_candidates_per_engine"] = max_candidates_per_engine
    request["max_trials_per_engine"] = max_trials_per_engine

    fingerprint = dict(source_manifest["data_fingerprint"])
    fingerprint["engines"] = [
        {"engine_id": engine_id, "version": engine_versions[engine_id]}
        for engine_id in engine_ids
    ]
    digest_material = dict(fingerprint)
    digest_material.pop("digest", None)
    fingerprint["digest"] = hashlib.sha256(
        json.dumps(
            digest_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return request, fingerprint


def run(args: argparse.Namespace) -> int:
    source_path = (
        args.data_dir
        / "alpha_mining"
        / "runs"
        / args.source_run_id
        / "manifest.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") not in {"succeeded", "succeeded_with_budget_exhausted"}:
        raise ValueError("source Alpha run must be successful and immutable")

    registry, failures = load_builtin_registry()
    if failures:
        raise RuntimeError(f"Alpha engine load failures: {failures}")
    versions = {
        engine.manifest.engine_id: engine.manifest.version
        for engine in registry.list()
    }
    unknown = sorted(set(args.engine) - set(versions))
    if unknown:
        raise ValueError(f"unknown Alpha engines: {unknown}")
    request, fingerprint = prepare_run(
        source,
        args.engine,
        versions,
        max_candidates_per_engine=args.max_candidates_per_engine,
        max_trials_per_engine=args.max_trials_per_engine,
    )

    manager = AlphaMiningJobManager(args.data_dir)
    manifest = manager.start(
        request,
        fingerprint,
        force=args.force,
        source="manual",
    )
    run_id = str(manifest["run_id"])
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": manifest["status"],
                "engine_ids": request["engine_ids"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    last_status = None
    while True:
        current = manager.store.get(run_id)
        if current is None:
            raise RuntimeError("Alpha run disappeared from persistent storage")
        status = str(current["status"])
        if status != last_status:
            print(
                json.dumps(
                    {"run_id": run_id, "status": status},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            last_status = status
        if status in TERMINAL_STATUSES:
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "status": status,
                        "error": current.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0 if status.startswith("succeeded") else 1
        time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/app/data"))
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--engine", action="append", required=True)
    parser.add_argument("--max-candidates-per-engine", type=int, default=8)
    parser.add_argument("--max-trials-per-engine", type=int, default=128)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
