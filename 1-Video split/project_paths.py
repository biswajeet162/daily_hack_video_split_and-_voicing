"""Shared configuration and tracker paths for the Video Split pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configuration"
TRACKERS_DIR = ROOT / "trackers"

INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "Transcriptions"
MERGED_DIR = ROOT / "output_merged_videos"
VOICEOVER_DIR = ROOT / "voiceover"
VOICEOVER_OUTPUT_DIR = ROOT / "output_voiceover_videos"
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
]


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


migrate_legacy_paths()
