"""Shared configuration and tracker paths for the Video Split pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configuration"
TRACKERS_DIR = ROOT / "trackers"

INPUT_DIR = ROOT / "1-input_videos"
OUTPUT_DIR = ROOT / "2-output_videos"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "Transcriptions"
MERGED_DIR = ROOT / "4-output_merged_videos"
VOICEOVER_DIR = ROOT / "voiceover"
VOICEOVER_OUTPUT_DIR = ROOT / "3-output_voiceover_videos"
VOICEOVER_RECORDINGS_DIR = VOICEOVER_DIR / "recordings"
VOICEOVER_FIXED_DIR = VOICEOVER_DIR / "recordings_fixed"
CATEGORIES_PATH = ROOT / "video_categories" / "categories.json"

SPLIT_TRIM_CONFIG = CONFIG_DIR / "split_trim_config.json"
MERGE_TRANSITION_CONFIG = CONFIG_DIR / "merge_transition_config.json"
REMOVE_TEXT_CONFIG = CONFIG_DIR / "remove_text_config.json"

DOWNLOAD_ARCHIVE = TRACKERS_DIR / "download_archive.txt"
SPLIT_PROCESSED = TRACKERS_DIR / "split_processed.json"
SPLIT_FAILED = TRACKERS_DIR / "split_failed.json"
TEXT_REMOVED_PROCESSED = TRACKERS_DIR / "text_removed_processed.json"
TEXT_REMOVED_FAILED = TRACKERS_DIR / "text_removed_failed.json"
TRANSCRIBE_PROCESSED = TRACKERS_DIR / "transcribe_processed.json"
TRANSCRIBE_FAILED = TRACKERS_DIR / "transcribe_failed.json"
CATEGORIZE_PROCESSED = TRACKERS_DIR / "categorize_processed.json"
CATEGORIZE_FAILED = TRACKERS_DIR / "categorize_failed.json"
MERGE_USED_CLIPS = TRACKERS_DIR / "merge_used_clips.json"
VOICEOVER_PROCESSED = TRACKERS_DIR / "voiceover_processed.json"

_FOLDER_RENAMES: list[tuple[str, str]] = [
    ("input_videos", "1-input_videos"),
    ("output_videos", "2-output_videos"),
    ("output_voiceover_videos", "3-output_voiceover_videos"),
    ("output_merged_videos", "4-output_merged_videos"),
]

_CANONICAL_FOLDER_NAMES: tuple[str, ...] = (
    "1-input_videos",
    "2-output_videos",
    "3-output_voiceover_videos",
    "4-output_merged_videos",
)

_LEGACY_PATH_SEGMENTS: list[tuple[str, str]] = [
    ("output_voiceover_videos", "3-output_voiceover_videos"),
    ("output_merged_videos", "4-output_merged_videos"),
    ("output_videos", "2-output_videos"),
    ("input_videos", "1-input_videos"),
]

_LEGACY_MOVES: list[tuple[Path, Path]] = [
    (ROOT / "split_trim_config.json", SPLIT_TRIM_CONFIG),
    (ROOT / "merge_transition_config.json", MERGE_TRANSITION_CONFIG),
    (ROOT / "remove_text_config.json", REMOVE_TEXT_CONFIG),
    (INPUT_DIR / ".download_archive.txt", DOWNLOAD_ARCHIVE),
    (INPUT_DIR / ".split_processed.json", SPLIT_PROCESSED),
    (INPUT_DIR / ".split_failed.json", SPLIT_FAILED),
    (OUTPUT_DIR / ".text_removed_processed.json", TEXT_REMOVED_PROCESSED),
    (OUTPUT_DIR / ".text_removed_failed.json", TEXT_REMOVED_FAILED),
    (OUTPUT_DIR / ".transcribe_processed.json", TRANSCRIBE_PROCESSED),
    (OUTPUT_DIR / ".transcribe_failed.json", TRANSCRIBE_FAILED),
    (OUTPUT_DIR / ".categorize_processed.json", CATEGORIZE_PROCESSED),
    (OUTPUT_DIR / ".categorize_failed.json", CATEGORIZE_FAILED),
    (OUTPUT_DIR / ".merge_used_clips.json", MERGE_USED_CLIPS),
    (ROOT / "input_videos" / ".download_archive.txt", DOWNLOAD_ARCHIVE),
    (ROOT / "input_videos" / ".split_processed.json", SPLIT_PROCESSED),
    (ROOT / "input_videos" / ".split_failed.json", SPLIT_FAILED),
    (ROOT / "output_videos" / ".text_removed_processed.json", TEXT_REMOVED_PROCESSED),
    (ROOT / "output_videos" / ".text_removed_failed.json", TEXT_REMOVED_FAILED),
    (ROOT / "output_videos" / ".transcribe_processed.json", TRANSCRIBE_PROCESSED),
    (ROOT / "output_videos" / ".transcribe_failed.json", TRANSCRIBE_FAILED),
    (ROOT / "output_videos" / ".categorize_processed.json", CATEGORIZE_PROCESSED),
    (ROOT / "output_videos" / ".categorize_failed.json", CATEGORIZE_FAILED),
    (ROOT / "output_videos" / ".merge_used_clips.json", MERGE_USED_CLIPS),
]


def migrate_data_folder_names() -> int:
    """Rename legacy data folders to numbered names (one-time, idempotent)."""
    moved = 0
    for old_name, new_name in _FOLDER_RENAMES:
        old = ROOT / old_name
        new = ROOT / new_name
        if old.exists() and not new.exists():
            shutil.move(str(old), str(new))
            moved += 1
    return moved


def _collapse_repeated_folder_names(text: str) -> tuple[str, bool]:
    import re

    updated = text
    changed = False
    patterns = [
        (re.compile(r"(?:\d+-)+1-input_videos"), "1-input_videos"),
        (re.compile(r"(?:\d+-)+2-output_videos"), "2-output_videos"),
        (re.compile(r"(?:\d+-)+3-output_voiceover_videos"), "3-output_voiceover_videos"),
        (re.compile(r"(?:\d+-)+4-output_merged_videos"), "4-output_merged_videos"),
    ]
    for pattern, canonical in patterns:
        new_text, count = pattern.subn(canonical, updated)
        if count:
            changed = True
            updated = new_text
    return updated, changed


def _replace_legacy_folder_segments(text: str) -> tuple[str, bool]:
    updated = text
    changed = False
    for old, new in _LEGACY_PATH_SEGMENTS:
        for sep in ("\\", "/"):
            old_seg = f"{sep}{old}{sep}"
            new_seg = f"{sep}{new}{sep}"
            if old_seg in updated:
                updated = updated.replace(old_seg, new_seg)
                changed = True
    return updated, changed


def normalize_stored_path(path: Path | str) -> Path:
    """Fix legacy or over-prefixed folder names inside stored absolute paths."""
    text = str(path).replace("/", "\\")
    text, _ = _collapse_repeated_folder_names(text)
    text, _ = _replace_legacy_folder_segments(text)
    return Path(text)


def _replace_path_strings(text: str) -> tuple[str, bool]:
    updated, changed_a = _collapse_repeated_folder_names(text)
    updated, changed_b = _replace_legacy_folder_segments(updated)
    return updated, changed_a or changed_b


def migrate_json_path_references() -> int:
    """Update stored absolute paths inside JSON files after folder renames."""
    updated_files = 0
    skip_dirs = {"voiceover", "configuration", "reference_numbers", "assets", "video_categories"}

    for json_path in ROOT.rglob("*.json"):
        if any(part in skip_dirs for part in json_path.parts):
            continue
        try:
            raw = json_path.read_text(encoding="utf-8")
        except OSError:
            continue
        new_raw, changed = _replace_path_strings(raw)
        if not changed:
            continue
        json_path.write_text(new_raw, encoding="utf-8")
        updated_files += 1
    return updated_files


def migrate_legacy_paths() -> int:
    """Move old root/input/output tracker and config files into configuration/ and trackers/."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TRACKERS_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for old, new in _LEGACY_MOVES:
        if old.exists() and not new.exists():
            shutil.move(str(old), str(new))
            moved += 1
    return moved


migrate_data_folder_names()
migrate_json_path_references()
migrate_legacy_paths()
