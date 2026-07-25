"""
[5] Merge split clips by category into one video + combined Hindi dialogue JSON.

Interactive (run.bat option 5):
  - Shows categories with available (not yet merged) clips
  - Prompts for category number and clip count
  - Randomly picks clips, merges with ffmpeg, writes output to output_merged_videos/

Tracks used clips in output_videos/.merge_used_clips.json so the same chunk
is not merged again unless --force is used.

Manual:
  conda run -n utube_env python 05_merge_by_category.py --interactive
  conda run -n utube_env python 05_merge_by_category.py --category clothes_hacks --count 5
  conda run -n utube_env python 05_merge_by_category.py --category kitchen_hacks --count 6 --force
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import log_utils as log

log.configure_stdio()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output_videos"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "Transcriptions"
MERGED_DIR = ROOT / "output_merged_videos"
CATEGORIES_PATH = ROOT / "video_categories" / "categories.json"
MERGE_TRACKER_PATH = OUTPUT_DIR / ".merge_used_clips.json"
TRANSITION_CONFIG_PATH = ROOT / "merge_transition_config.json"
DEFAULT_CLIP_COUNT = 5
DEFAULT_GAP_DURATION_SEC = 1.0
OUTPUT_FPS = 30
OUTPUT_AUDIO_RATE = 48000


def ensure_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if not missing:
        return
    raise RuntimeError(
        f"{', '.join(missing)} not found on PATH. Install ffmpeg and retry."
    )


def load_categories() -> dict[str, dict]:
    data = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    categories = data.get("categories") or []
    return {item["slug"]: item for item in categories if item.get("slug")}


def load_merge_tracker() -> dict:
    if not MERGE_TRACKER_PATH.exists():
        return {"clips": {}, "merges": {}}
    try:
        data = json.loads(MERGE_TRACKER_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"clips": {}, "merges": {}}
    if not isinstance(data, dict):
        return {"clips": {}, "merges": {}}
    data.setdefault("clips", {})
    data.setdefault("merges", {})
    return data


def save_merge_tracker(tracker: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MERGE_TRACKER_PATH.write_text(
        json.dumps(tracker, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def transcript_key(path: Path) -> str:
    try:
        return path.relative_to(TRANSCRIPTIONS_DIR).as_posix()
    except ValueError:
        return path.name


def clip_key(video_path: Path) -> str:
    return video_path.relative_to(OUTPUT_DIR).as_posix()


def load_transcription_payload(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def video_path_from_transcription(transcript_path: Path, payload: dict) -> Path | None:
    source = payload.get("source")
    if source:
        candidate = Path(str(source))
        if candidate.exists():
            return candidate.resolve()

    rel = transcript_path.relative_to(TRANSCRIPTIONS_DIR)
    video_name = rel.name.replace(".transcription.json", ".mp4")
    candidate = OUTPUT_DIR / rel.parent / video_name
    if candidate.exists():
        return candidate.resolve()
    return None


def is_clip_merged(key: str, tracker: dict, *, force: bool) -> bool:
    if force:
        return False
    return key in tracker.get("clips", {})


def iter_clip_records(*, force: bool) -> list[dict]:
    tracker = load_merge_tracker()
    records: list[dict] = []

    for transcript_path in sorted(TRANSCRIPTIONS_DIR.rglob("*.transcription.json")):
        payload = load_transcription_payload(transcript_path)
        if payload is None:
            continue

        categories = payload.get("categories") or []
        if not isinstance(categories, list) or not categories:
            continue

        video_path = video_path_from_transcription(transcript_path, payload)
        if video_path is None or not video_path.exists():
            continue

        key = clip_key(video_path)
        if is_clip_merged(key, tracker, force=force):
            continue

        text = (payload.get("text") or "").strip()
        if not text and payload.get("segments"):
            text = " ".join(
                (seg.get("text") or "").strip()
                for seg in payload["segments"]
                if isinstance(seg, dict)
            ).strip()

        records.append(
            {
                "clip_key": key,
                "video_path": video_path,
                "transcript_path": transcript_path,
                "transcript_key": transcript_key(transcript_path),
                "categories": [str(c) for c in categories],
                "category_names_en": payload.get("category_names_en") or [],
                "category_names_hi": payload.get("category_names_hi") or [],
                "text": text,
                "duration_sec": payload.get("duration_sec"),
            }
        )

    return records


def build_category_availability(
    records: list[dict],
    categories_by_slug: dict[str, dict],
) -> list[dict]:
    counts: dict[str, int] = {}
    for record in records:
        for slug in record["categories"]:
            if slug in categories_by_slug:
                counts[slug] = counts.get(slug, 0) + 1

    rows: list[dict] = []
    for slug, meta in categories_by_slug.items():
        available = counts.get(slug, 0)
        if available <= 0:
            continue
        rows.append(
            {
                "slug": slug,
                "name_en": meta.get("name_en", slug),
                "name_hi": meta.get("name_hi", ""),
                "available_clips": available,
            }
        )

    rows.sort(key=lambda row: (-row["available_clips"], row["slug"]))
    return rows


def clips_for_category(records: list[dict], category_slug: str) -> list[dict]:
    return [r for r in records if category_slug in r["categories"]]


def pick_random_clips(clips: list[dict], count: int) -> list[dict]:
    if count > len(clips):
        raise ValueError(
            f"Requested {count} clips but only {len(clips)} available for this category."
        )
    return random.sample(clips, count)


def merge_id_for(category_slug: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{category_slug}_{stamp}"


def output_paths(merge_id: str) -> tuple[Path, Path]:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    video_path = MERGED_DIR / f"{merge_id}.mp4"
    json_path = MERGED_DIR / f"{merge_id}.json"
    return video_path, json_path


def load_transition_config(config_path: Path = TRANSITION_CONFIG_PATH) -> dict:
    if not config_path.exists():
        log.warn(f"Transition config not found, using defaults: {config_path.name}")
        return {
            "mode": "frame_bridge",
            "gap_duration_sec": DEFAULT_GAP_DURATION_SEC,
            "fade_out_sec": DEFAULT_GAP_DURATION_SEC / 2,
            "fade_in_sec": DEFAULT_GAP_DURATION_SEC / 2,
        }

    data = json.loads(config_path.read_text(encoding="utf-8"))
    gap_duration = float(
        data.get("gap_duration_sec")
        or data.get("transition_duration_sec")
        or DEFAULT_GAP_DURATION_SEC
    )
    if gap_duration <= 0:
        raise ValueError("gap_duration_sec must be positive")

    fade_out = float(data.get("fade_out_sec") or gap_duration / 2)
    fade_in = float(data.get("fade_in_sec") or gap_duration / 2)
    if fade_out <= 0 or fade_in <= 0:
        raise ValueError("fade_out_sec and fade_in_sec must be positive")

    data["mode"] = data.get("mode", "frame_bridge")
    data["gap_duration_sec"] = fade_out + fade_in
    data["fade_out_sec"] = fade_out
    data["fade_in_sec"] = fade_in
    return data


def extract_video_frame(video_path: Path, position: str) -> Path:
    frame_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    frame_file.close()
    frame_path = Path(frame_file.name)

    if position == "first":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,0)",
            "-vframes",
            "1",
            str(frame_path),
        ]
    elif position == "last":
        duration = probe_duration_seconds(video_path)
        seek = max(0.0, duration - 0.04)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{seek:.3f}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            str(frame_path),
        ]
    else:
        raise ValueError(f"Unknown frame position: {position}")

    _run_ffmpeg(cmd)
    if not frame_path.exists() or frame_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract {position} frame from {video_path.name}")
    return frame_path


def generate_frame_bridge_clip(
    output_path: Path,
    prev_video: Path,
    next_video: Path,
    *,
    width: int,
    height: int,
    fade_out_sec: float,
    fade_in_sec: float,
) -> None:
    """1s bridge: last frame of prev fades out, then first frame of next fades in."""
    last_frame = extract_video_frame(prev_video, "last")
    first_frame = extract_video_frame(next_video, "first")
    temp_frames = [last_frame, first_frame]
    total_duration = fade_out_sec + fade_in_sec

    scale_pad = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    try:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-t",
                f"{fade_out_sec:.3f}",
                "-i",
                str(last_frame),
                "-loop",
                "1",
                "-t",
                f"{fade_in_sec:.3f}",
                "-i",
                str(first_frame),
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={OUTPUT_AUDIO_RATE}:cl=stereo",
                "-filter_complex",
                (
                    f"[0:v]{scale_pad},fade=t=out:st=0:d={fade_out_sec:.3f},"
                    f"format=yuv420p[v0];"
                    f"[1:v]{scale_pad},fade=t=in:st=0:d={fade_in_sec:.3f},"
                    f"format=yuv420p[v1];"
                    f"[v0][v1]concat=n=2:v=1:a=0,format=yuv420p[v]"
                ),
                "-map",
                "[v]",
                "-map",
                "2:a",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-r",
                str(OUTPUT_FPS),
                "-t",
                f"{total_duration:.3f}",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    finally:
        for frame_path in temp_frames:
            frame_path.unlink(missing_ok=True)


def concat_video_pieces(pieces: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as list_file:
        for path in pieces:
            escaped = str(path.resolve()).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = list_file.name

    try:
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


def probe_duration_seconds(media_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "format=duration",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout or "{}")
    return float(data.get("format", {}).get("duration", 0) or 0)


def probe_video_size(media_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-show_entries",
            "stream=width,height",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    width = int(stream.get("width") or 1080)
    height = int(stream.get("height") or 1920)
    return width, height


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1200:] or "ffmpeg command failed")


def copy_single_clip(source: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def merge_videos_with_ffmpeg(
    video_paths: list[Path],
    output_path: Path,
    transition_config: dict,
) -> list[dict]:
    if not video_paths:
        raise ValueError("No videos to merge")

    if len(video_paths) == 1:
        copy_single_clip(video_paths[0], output_path)
        return []

    gap_duration = float(transition_config["gap_duration_sec"])
    fade_out_sec = float(transition_config["fade_out_sec"])
    fade_in_sec = float(transition_config["fade_in_sec"])
    width, height = probe_video_size(video_paths[0])
    pieces: list[Path] = []
    bridge_temp_paths: list[Path] = []
    transition_log: list[dict] = []

    for index, clip_path in enumerate(video_paths):
        pieces.append(clip_path)

        if index >= len(video_paths) - 1:
            continue

        next_clip_path = video_paths[index + 1]
        bridge_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        bridge_file.close()
        bridge_path = Path(bridge_file.name)
        bridge_temp_paths.append(bridge_path)

        log.progress(
            f"Bridge {index + 1}/{len(video_paths) - 1} after clip {index + 1}: "
            f"last frame fade-out {fade_out_sec:.2f}s + "
            f"first frame fade-in {fade_in_sec:.2f}s (no overlap)"
        )

        generate_frame_bridge_clip(
            bridge_path,
            clip_path,
            next_clip_path,
            width=width,
            height=height,
            fade_out_sec=fade_out_sec,
            fade_in_sec=fade_in_sec,
        )
        pieces.append(bridge_path)
        transition_log.append(
            {
                "after_clip_order": index + 1,
                "before_clip_order": index + 2,
                "effect_type": "frame_fade_bridge",
                "effect_name_en": "Frame Fade Bridge",
                "effect_name_hi": "फ्रेम फेड ब्रिज",
                "fade_out_sec": fade_out_sec,
                "fade_in_sec": fade_in_sec,
                "bridge_duration_sec": gap_duration,
                "mode": transition_config.get("mode", "frame_bridge"),
            }
        )

    concat_video_pieces(pieces, output_path)

    for bridge_path in bridge_temp_paths:
        bridge_path.unlink(missing_ok=True)

    return transition_log


def build_merge_json(
    *,
    merge_id: str,
    category_slug: str,
    category_meta: dict,
    selected: list[dict],
    output_video: Path,
    transition_config: dict,
    transition_log: list[dict],
) -> dict:
    clip_rows = []
    dialogue_lines: list[str] = []
    clip_durations: list[float] = []

    for index, record in enumerate(selected, 1):
        text = (record.get("text") or "").strip()
        duration = record.get("duration_sec")
        if isinstance(duration, (int, float)):
            clip_durations.append(float(duration))

        clip_rows.append(
            {
                "order": index,
                "clip_key": record["clip_key"],
                "video": str(record["video_path"]),
                "transcription_json": str(record["transcript_path"]),
                "text": text,
                "duration_sec": duration,
                "categories": record.get("categories") or [],
            }
        )
        if text:
            dialogue_lines.append(text)

    gap_duration = float(transition_config.get("gap_duration_sec", 0) or 0)
    gap_added = gap_duration * max(0, len(selected) - 1)
    raw_total = sum(clip_durations)
    merged_total = raw_total + gap_added

    return {
        "merge_id": merge_id,
        "category_slug": category_slug,
        "category_name_en": category_meta.get("name_en", category_slug),
        "category_name_hi": category_meta.get("name_hi", ""),
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(selected),
        "raw_clip_duration_sec": round(raw_total, 3),
        "gap_added_sec": round(gap_added, 3),
        "gap_duration_sec": gap_duration,
        "total_duration_sec": round(merged_total, 3),
        "output_video": str(output_video),
        "transition_config": str(TRANSITION_CONFIG_PATH),
        "merge_mode": transition_config.get("mode", "frame_bridge"),
        "fade_out_sec": transition_config.get("fade_out_sec"),
        "fade_in_sec": transition_config.get("fade_in_sec"),
        "bridge_transitions": transition_log,
        "gap_effects": transition_log,
        "clips": clip_rows,
        "dialogues": [
            {"order": i + 1, "text": line}
            for i, line in enumerate(dialogue_lines)
        ],
        "full_dialogue_hindi": "\n\n".join(dialogue_lines),
    }


def mark_clips_merged(
    tracker: dict,
    merge_id: str,
    category_slug: str,
    selected: list[dict],
    output_video: Path,
    output_json: Path,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    clip_keys = [record["clip_key"] for record in selected]

    for record in selected:
        tracker.setdefault("clips", {})[record["clip_key"]] = {
            "merged_at": now,
            "merge_id": merge_id,
            "category_slug": category_slug,
            "video": str(record["video_path"]),
            "transcription_json": str(record["transcript_path"]),
            "merged_video": str(output_video),
            "merged_json": str(output_json),
        }

    tracker.setdefault("merges", {})[merge_id] = {
        "merged_at": now,
        "category_slug": category_slug,
        "clip_count": len(selected),
        "clip_keys": clip_keys,
        "output_video": str(output_video),
        "output_json": str(output_json),
    }
    save_merge_tracker(tracker)


def run_merge(
    category_slug: str,
    clip_count: int,
    *,
    force: bool = False,
    transition_config_path: Path = TRANSITION_CONFIG_PATH,
) -> int:
    categories_by_slug = load_categories()
    if category_slug not in categories_by_slug:
        raise ValueError(f"Unknown category slug: {category_slug}")

    transition_config = load_transition_config(transition_config_path)
    log.ok(
        f"Loaded transition config: {transition_config_path.name} "
        f"(frame bridge: {transition_config['fade_out_sec']:.2f}s fade-out + "
        f"{transition_config['fade_in_sec']:.2f}s fade-in, no overlap)"
    )

    records = iter_clip_records(force=force)
    available = clips_for_category(records, category_slug)
    if not available:
        raise ValueError(
            f"No available clips for category '{category_slug}'. "
            "All matching clips may already be merged."
        )

    selected = pick_random_clips(available, clip_count)
    video_paths = [record["video_path"] for record in selected]

    merge_id = merge_id_for(category_slug)
    output_video, output_json = output_paths(merge_id)

    log.section(f"Merging {len(selected)} clip(s) with frame-bridge transitions")
    for i, record in enumerate(selected, 1):
        log.bullet(f"[{i}] {record['clip_key']}")
        preview = (record.get("text") or "")[:80]
        if preview:
            log.bullet(f"    {preview}{'...' if len(record.get('text') or '') > 80 else ''}")

    transition_log = merge_videos_with_ffmpeg(
        video_paths,
        output_video,
        transition_config,
    )

    payload = build_merge_json(
        merge_id=merge_id,
        category_slug=category_slug,
        category_meta=categories_by_slug[category_slug],
        selected=selected,
        output_video=output_video,
        transition_config=transition_config,
        transition_log=transition_log,
    )
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tracker = load_merge_tracker()
    mark_clips_merged(
        tracker,
        merge_id,
        category_slug,
        selected,
        output_video,
        output_json,
    )

    log.ok(f"Merged video saved: {output_video.name}")
    log.ok(f"Dialogue JSON saved: {output_json.name}")
    if transition_log:
        log.info(
            "Bridge transitions",
            f"{len(transition_log)} x frame fade bridge "
            f"({transition_config['fade_out_sec']:.2f}s + "
            f"{transition_config['fade_in_sec']:.2f}s)",
        )
        log.info(
            "Duration",
            f"{payload['raw_clip_duration_sec']:.2f}s clips + "
            f"{payload['gap_added_sec']:.2f}s bridges = "
            f"{payload['total_duration_sec']:.2f}s total",
        )
    log.info("Merge tracker", MERGE_TRACKER_PATH)
    log.info("Full Hindi dialogue", payload["full_dialogue_hindi"][:160] + "...")
    return 0


def interactive_merge(*, force: bool = False) -> int:
    categories_by_slug = load_categories()
    records = iter_clip_records(force=force)
    options = build_category_availability(records, categories_by_slug)

    if not options:
        log.fail("No categorized clips are available to merge.")
        log.info("Tip", "Run steps 3 and 4 first, or use --force to reuse clips.")
        return 1

    log.section("Pick a category (only categories with available clips)")
    for i, row in enumerate(options, 1):
        log.bullet(
            f"[{i:2d}] {row['name_en']} ({row['name_hi']}) "
            f"- {row['available_clips']} clip(s) available  [{row['slug']}]"
        )

    print()
    choice_raw = input("Enter category number: ").strip()
    if not re.fullmatch(r"\d+", choice_raw):
        log.fail("Invalid category number.")
        return 1

    choice = int(choice_raw)
    if choice < 1 or choice > len(options):
        log.fail(f"Choose a number between 1 and {len(options)}.")
        return 1

    selected_category = options[choice - 1]
    max_clips = selected_category["available_clips"]

    print()
    count_raw = input(
        f"How many clips to merge? (1-{max_clips}, default {DEFAULT_CLIP_COUNT}): "
    ).strip()
    if not count_raw:
        clip_count = min(DEFAULT_CLIP_COUNT, max_clips)
    elif re.fullmatch(r"\d+", count_raw):
        clip_count = int(count_raw)
    else:
        log.fail("Invalid clip count.")
        return 1

    if clip_count < 1 or clip_count > max_clips:
        log.fail(f"Clip count must be between 1 and {max_clips}.")
        return 1

    log.summary(
        "Merge selection",
        [
            ("Category", selected_category["name_en"]),
            ("Slug", selected_category["slug"]),
            ("Clips to merge", str(clip_count)),
            ("Available clips", str(max_clips)),
            ("Output folder", MERGED_DIR),
            ("Transition config", TRANSITION_CONFIG_PATH),
            ("Used-clips tracker", MERGE_TRACKER_PATH),
        ],
    )

    try:
        return run_merge(
            selected_category["slug"],
            clip_count,
            force=force,
            transition_config_path=TRANSITION_CONFIG_PATH,
        )
    except Exception as exc:
        log.fail(str(exc))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge split clips by category into one video + dialogue JSON"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show category menu and prompt for clip count (default when no args)",
    )
    parser.add_argument(
        "--category",
        help="Category slug, e.g. clothes_hacks",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CLIP_COUNT,
        help=f"Number of clips to merge (default: {DEFAULT_CLIP_COUNT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow reusing clips that were already merged before",
    )
    parser.add_argument(
        "--transition-config",
        type=Path,
        default=TRANSITION_CONFIG_PATH,
        help=f"Transition settings JSON (default: {TRANSITION_CONFIG_PATH.name})",
    )
    args = parser.parse_args()

    log.banner(
        "[5] MERGE CLIPS BY CATEGORY",
        "Random clips -> one merged video + combined Hindi dialogue JSON",
    )

    try:
        ensure_ffmpeg()
        log.ok("ffmpeg and ffprobe found on PATH")
    except RuntimeError as exc:
        log.fail(str(exc))
        return 1

    log.paths_block(
        "Folders and trackers",
        [
            ("Project root", ROOT),
            ("Source clips", OUTPUT_DIR),
            ("Transcriptions", TRANSCRIPTIONS_DIR),
            ("Merged output", MERGED_DIR),
            ("Transition config", args.transition_config),
            ("Used-clips tracker", MERGE_TRACKER_PATH),
            ("Category list", CATEGORIES_PATH),
        ],
    )

    if args.interactive or not args.category:
        return interactive_merge(force=args.force)

    if args.count < 1:
        log.fail("Clip count must be at least 1.")
        return 1

    try:
        return run_merge(
            args.category,
            args.count,
            force=args.force,
            transition_config_path=args.transition_config,
        )
    except Exception as exc:
        log.fail(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
