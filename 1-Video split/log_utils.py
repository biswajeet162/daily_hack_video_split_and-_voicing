"""Shared console logging for the Video Split pipeline."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

WIDTH = 72
BANNER_CHAR = "="
SECTION_CHAR = "-"


def configure_stdio() -> None:
    """Ensure UTF-8 output and line buffering when run from run.bat."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def banner(title: str, subtitle: str = "") -> None:
    print()
    print(BANNER_CHAR * WIDTH)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(BANNER_CHAR * WIDTH)
    print()


def section(title: str) -> None:
    print()
    print(SECTION_CHAR * WIDTH)
    print(f"  [{_timestamp()}] {title}")
    print(SECTION_CHAR * WIDTH)


def step(current: int, total: int, message: str) -> None:
    print()
    print(f">> STEP {current}/{total}  {message}")
    print(f"   started at {_timestamp()}")


def info(label: str, value: str | Path) -> None:
    print(f"     {label:<18} {value}")


def bullet(text: str, indent: int = 1) -> None:
    pad = "  " * indent
    print(f"{pad}- {text}")


def ok(message: str) -> None:
    print(f"  [OK]  {message}")


def warn(message: str) -> None:
    print(f"  [!!]  {message}")


def fail(message: str) -> None:
    print(f"  [XX]  {message}", file=sys.stderr)


def progress(message: str) -> None:
    print(f"  ...   {message}")


def divider() -> None:
    print(SECTION_CHAR * WIDTH)


def summary(title: str, rows: list[tuple[str, str | Path]]) -> None:
    section(title)
    for label, value in rows:
        info(label, value)


def paths_block(title: str, paths: list[tuple[str, str | Path]]) -> None:
    section(title)
    for label, path in paths:
        info(label, path)


def done_block(
    title: str,
    rows: list[tuple[str, str | int | Path]],
    *,
    success: bool = True,
) -> None:
    print()
    print(BANNER_CHAR * WIDTH)
    status = "COMPLETED" if success else "FINISHED WITH ERRORS"
    print(f"  {status}: {title}")
    print(BANNER_CHAR * WIDTH)
    for label, value in rows:
        info(label, value)
    print()
