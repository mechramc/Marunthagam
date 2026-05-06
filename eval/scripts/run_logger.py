"""
Marunthagam — Eval run logging.

Creates a per-run directory under eval/logs/<run_id>/ containing:
    - manifest.json   (cmd, args, git sha, model paths, durations, exit status)
    - stdout.log      (mirrored from sys.stdout)
    - stderr.log      (mirrored from sys.stderr)
    - run.log         (combined timestamped log lines)

Eval scripts call:
    with RunLogger(kind="run_eval", args=args_namespace) as run:
        ...
        run.log_event("seed_done", seed=42, weighted_f1=0.81)
        run.attach_result(payload)   # path to the json result the script also wrote
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "eval" / "logs"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=5,
            check=False,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


class _Tee(io.TextIOBase):
    """Mirror writes to multiple streams. Used to fork stdout/stderr to log files."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:  # type: ignore[override]
        for stream in self._streams:
            try:
                stream.write(s)
                stream.flush()
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False


class RunLogger:
    """
    Per-run logging context. Use as a `with` block.

    The directory is created up-front so failures still leave a partial manifest.
    """

    def __init__(
        self,
        kind: str,
        args: Optional[Any] = None,
        extra_manifest: Optional[dict] = None,
    ) -> None:
        self.kind = kind
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{kind}_{self.timestamp}"
        self.run_dir = LOGS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.stdout_path = self.run_dir / "stdout.log"
        self.stderr_path = self.run_dir / "stderr.log"
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"

        self._stdout_file: Optional[TextIO] = None
        self._stderr_file: Optional[TextIO] = None
        self._orig_stdout: Optional[TextIO] = None
        self._orig_stderr: Optional[TextIO] = None
        self._events_file: Optional[TextIO] = None
        self._start_time: float = 0.0

        args_dict: dict[str, Any] = {}
        if args is not None:
            try:
                args_dict = {k: _jsonable(v) for k, v in vars(args).items()}
            except TypeError:
                args_dict = {"_repr": repr(args)}

        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "kind": kind,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "duration_s": None,
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "argv": list(sys.argv),
            "args": args_dict,
            "events": [],
            "result_paths": [],
            "exit_status": None,
        }
        if extra_manifest:
            self.manifest.update(extra_manifest)

    def __enter__(self) -> "RunLogger":
        self._start_time = time.monotonic()
        self._stdout_file = open(self.stdout_path, "w", encoding="utf-8", buffering=1)
        self._stderr_file = open(self.stderr_path, "w", encoding="utf-8", buffering=1)
        self._events_file = open(self.events_path, "w", encoding="utf-8", buffering=1)

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._stdout_file)  # type: ignore[assignment]
        sys.stderr = _Tee(self._orig_stderr, self._stderr_file)  # type: ignore[assignment]

        self._write_manifest()
        self.log_event("run_start", run_id=self.run_id, kind=self.kind)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        end = time.monotonic()
        self.manifest["duration_s"] = round(end - self._start_time, 3)
        self.manifest["ended_at"] = datetime.now(timezone.utc).isoformat()

        if exc_type is None:
            self.manifest["exit_status"] = "ok"
        else:
            self.manifest["exit_status"] = "error"
            self.manifest["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
            }
            self.log_event("run_error", error_type=exc_type.__name__, message=str(exc_val))

        self.log_event("run_end", duration_s=self.manifest["duration_s"])
        self._write_manifest()

        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        for fh in (self._stdout_file, self._stderr_file, self._events_file):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        # Re-raise any exception
        return False

    def log_event(self, name: str, **fields: Any) -> None:
        """Append a structured event to events.jsonl and the manifest event list."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": name,
            **{k: _jsonable(v) for k, v in fields.items()},
        }
        if self._events_file is not None:
            self._events_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._events_file.flush()
        self.manifest["events"].append(record)

    def attach_result(self, result_path: Path | str) -> None:
        """Record that a results JSON / CSV / etc. was produced for this run."""
        self.manifest["result_paths"].append(str(result_path))
        self._write_manifest()

    def merge_manifest(self, **fields: Any) -> None:
        """Merge top-level fields into the manifest."""
        for k, v in fields.items():
            self.manifest[k] = _jsonable(v)
        self._write_manifest()

    def _write_manifest(self) -> None:
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as fh:
                json.dump(self.manifest, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _jsonable(value: Any) -> Any:
    """Best-effort JSON normalisation for argparse Namespaces and Path objects."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)
