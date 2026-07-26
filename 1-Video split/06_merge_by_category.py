"""
[6] Merge split clips by category into one video + combined Hindi dialogue JSON.

Interactive (run.bat option 6):
  - Shows categories with available (not yet merged) clips
  - Prompts for category number and clip count
  - Randomly picks clips, merges with ffmpeg, writes output to output_merged_videos/

Tracks used clips in output_videos/.merge_used_clips.json so the same chunk
is not merged again unless --force is used.

Merging uses ffmpeg/ffprobe (subprocess), not MoviePy or OpenCV.

Manual:
  conda run -n utube_env python 06_merge_by_category.py --interactive
  conda run -n utube_env python 06_merge_by_category.py --category clothes_hacks --count 5
  conda run -n utube_env python 06_merge_by_category.py --category kitchen_hacks --count 6 --force
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

from PIL import Image, ImageColor, ImageDraw, ImageFont

import log_utils as log

log.configure_stdio()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output_videos"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "Transcriptions"
MERGED_DIR = ROOT / "output_merged_videos"
CATEGORIES_PATH = ROOT / "video_categories" / "categories.json"
MERGE_TRACKER_PATH = OUTPUT_DIR / ".merge_used_clips.json"
TRANSITION_CONFIG_PATH = ROOT / "merge_transition_config.json"
LABEL_FONT_PATH = ROOT / "assets" / "fonts" / "NotoSansDevanagari-Regular.ttf"
DEFAULT_CLIP_COUNT = 5
DEFAULT_GAP_DURATION_SEC = 1.0
OUTPUT_FPS = 30
DEFAULT_TRANSITION_FPS = 60
OUTPUT_AUDIO_RATE = 48000
FRAME_TRIM_SEC = 1.0 / OUTPUT_FPS
DEFAULT_FADE_OUT_SEC = 0.5
DEFAULT_FADE_IN_SEC = 0.5
MIN_LABEL_FONT_SIZE = 16
DEFAULT_LABEL_OVERLAY = {
    "font_size": 0,
    "line_gap": 10,
    "left_margin": 24,
    "top_margin": 36,
    "color": "yellow",
    "display_mode": "synced",
    "numbering": "reverse",
}
RANDOM_LABEL_COLORS = [
    (255, 255, 0, 255),
    (0, 255, 255, 255),
    (255, 128, 0, 255),
    (255, 105, 180, 255),
    (50, 205, 50, 255),
    (255, 64, 64, 255),
    (173, 216, 230, 255),
    (255, 215, 0, 255),
]


def ffmpeg_encode_args(fps: int) -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-ar",
        str(OUTPUT_AUDIO_RATE),
        "-ac",
        "2",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
    ]


FFMPEG_ENCODE_ARGS = ffmpeg_encode_args(OUTPUT_FPS)


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
                "display_label": (payload.get("display_label") or "").strip(),
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
            "mode": "frame_fade_bridge",
            "gap_duration_sec": DEFAULT_GAP_DURATION_SEC,
            "fade_out_sec": DEFAULT_FADE_OUT_SEC,
            "fade_in_sec": DEFAULT_FADE_IN_SEC,
            "exit_sec": DEFAULT_FADE_OUT_SEC,
            "enter_sec": DEFAULT_FADE_IN_SEC,
            "transition_fps": DEFAULT_TRANSITION_FPS,
            "clip_label_overlay": load_label_overlay_config({}),
        }

    data = json.loads(config_path.read_text(encoding="utf-8"))
    gap_duration = float(
        data.get("gap_duration_sec")
        or data.get("transition_duration_sec")
        or DEFAULT_GAP_DURATION_SEC
    )
    if gap_duration <= 0:
        raise ValueError("gap_duration_sec must be positive")

    exit_sec = float(
        data.get("exit_sec")
        or data.get("fade_out_sec")
        or gap_duration / 2
    )
    enter_sec = float(
        data.get("enter_sec")
        or data.get("fade_in_sec")
        or gap_duration / 2
    )
    if exit_sec <= 0 or enter_sec <= 0:
        raise ValueError("exit_sec/fade_out_sec and enter_sec/fade_in_sec must be positive")

    data["mode"] = data.get("mode", "frame_fade_bridge")
    data["gap_duration_sec"] = exit_sec + enter_sec
    data["fade_out_sec"] = exit_sec
    data["fade_in_sec"] = enter_sec
    data["exit_sec"] = exit_sec
    data["enter_sec"] = enter_sec
    transition_fps = int(data.get("transition_fps") or DEFAULT_TRANSITION_FPS)
    if transition_fps < 30:
        raise ValueError("transition_fps must be at least 30")
    data["transition_fps"] = transition_fps
    data["output_fps"] = transition_fps
    data["clip_label_overlay"] = load_label_overlay_config(data)
    return data


def load_label_overlay_config(data: dict) -> dict:
    overlay = data.get("clip_label_overlay") or {}
    if not isinstance(overlay, dict):
        overlay = {}

    font_size = int(overlay.get("font_size", DEFAULT_LABEL_OVERLAY["font_size"]))
    line_gap = int(overlay.get("line_gap", DEFAULT_LABEL_OVERLAY["line_gap"]))
    left_margin = int(overlay.get("left_margin", DEFAULT_LABEL_OVERLAY["left_margin"]))
    top_margin = int(overlay.get("top_margin", DEFAULT_LABEL_OVERLAY["top_margin"]))
    color = str(overlay.get("color", DEFAULT_LABEL_OVERLAY["color"])).strip()

    if font_size < 0:
        raise ValueError("clip_label_overlay.font_size must be 0 (auto) or positive")
    if 0 < font_size < MIN_LABEL_FONT_SIZE:
        log.warn(
            f"clip_label_overlay.font_size={font_size} is too small; "
            f"using {MIN_LABEL_FONT_SIZE}px minimum for readable labels"
        )
        font_size = MIN_LABEL_FONT_SIZE
    if line_gap < 0:
        raise ValueError("clip_label_overlay.line_gap must be 0 or positive")
    if left_margin < 0 or top_margin < 0:
        raise ValueError("clip_label_overlay margins must be 0 or positive")
    if not color:
        raise ValueError("clip_label_overlay.color must not be empty")

    random_color = color.upper() == "RANDOM"
    if not random_color:
        parse_label_color(color)

    display_mode = str(
        overlay.get("display_mode", DEFAULT_LABEL_OVERLAY["display_mode"])
    ).strip().lower()
    numbering = str(
        overlay.get("numbering", DEFAULT_LABEL_OVERLAY["numbering"])
    ).strip().lower()
    if display_mode not in {"synced", "all"}:
        raise ValueError('clip_label_overlay.display_mode must be "synced" or "all"')
    if numbering not in {"reverse", "normal"}:
        raise ValueError('clip_label_overlay.numbering must be "reverse" or "normal"')

    return {
        "font_size": font_size,
        "line_gap": line_gap,
        "left_margin": left_margin,
        "top_margin": top_margin,
        "color": color,
        "random_color": random_color,
        "display_mode": display_mode,
        "numbering": numbering,
    }


def parse_label_color(color: str) -> tuple[int, int, int, int]:
    try:
        red, green, blue = ImageColor.getrgb(color)
    except ValueError as exc:
        raise ValueError(
            f"Invalid clip_label_overlay.color {color!r}. "
            "Use a color name, #RRGGBB, or RANDOM."
        ) from exc
    return (red, green, blue, 255)


def pick_random_label_color() -> tuple[int, int, int, int]:
    return random.choice(RANDOM_LABEL_COLORS)


def pick_label_color_for_line(
    line_num: int,
    *,
    random_color: bool,
    fixed_color: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if not random_color:
        return fixed_color or (255, 255, 0, 255)
    rng = random.Random(line_num * 9973 + 17)
    return rng.choice(RANDOM_LABEL_COLORS)


def play_order_for_line(line_num: int, total: int, numbering: str) -> int:
    if numbering == "normal":
        return line_num
    return total - line_num + 1


def line_should_show_text(
    line_num: int,
    active_play_order: int,
    total: int,
    numbering: str,
) -> bool:
    if numbering == "reverse":
        return line_num >= active_text_line(active_play_order, total, numbering)
    return line_num <= active_play_order


def bridge_frame_count(duration_sec: float, fps: int) -> int:
    return max(1, round(duration_sec * fps))


def merge_target_size(video_paths: list[Path]) -> tuple[int, int]:
    width = 0
    height = 0
    for path in video_paths:
        clip_w, clip_h = probe_video_size(path)
        width = max(width, clip_w)
        height = max(height, clip_h)
    return width or 1080, height or 1920


def scale_fill_vf(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )


def scale_pad_vf(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )


def fallback_display_label(text: str) -> str:
    words = re.findall(r"[\u0900-\u097F]+", text or "")
    if not words:
        words = (text or "").split()
    cleaned = [word.strip() for word in words if word.strip()]
    if not cleaned:
        return "हैक"
    return " ".join(cleaned[:2])


def _is_valid_ttf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        magic = handle.read(4)
    return magic in {b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1"}


def _font_renders_devanagari(font_path: Path) -> bool:
    if not _is_valid_ttf(font_path):
        return False
    try:
        font = ImageFont.truetype(str(font_path), 48)
    except OSError:
        return False

    sample = "क"
    bbox = font.getbbox(sample)
    if not bbox or bbox[2] - bbox[0] <= 0:
        return False

    image = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), sample, font=font, fill=255)
    ink = image.getbbox()
    return ink is not None and ink[2] - ink[0] >= 12


def resolve_label_font() -> Path:
    candidates = [
        LABEL_FONT_PATH,
        Path(r"C:\Windows\Fonts\NirmalaUI.ttf"),
        Path(r"C:\Windows\Fonts\Nirmala.ttf"),
        Path(r"C:\Windows\Fonts\Mangal.ttf"),
        Path(r"C:\Windows\Fonts\NirmalaUIb.ttf"),
        Path(r"C:\Windows\Fonts\NirmalaS.ttf"),
    ]
    for candidate in candidates:
        if not _font_renders_devanagari(candidate):
            continue
        if candidate != LABEL_FONT_PATH:
            log.warn(
                f"Using system Hindi label font: {candidate.name} "
                f"(bundled font unavailable: {LABEL_FONT_PATH.name})"
            )
        return candidate

    raise RuntimeError(
        "No usable Hindi label font found. "
        f"Install or restore {LABEL_FONT_PATH.name} (Noto Sans Devanagari)."
    )


def active_text_line(play_order: int, total: int, numbering: str) -> int:
    if numbering == "reverse":
        return total - play_order + 1
    return play_order


def render_label_overlay_png(
    *,
    width: int,
    height: int,
    font_path: Path,
    output_path: Path,
    label_config: dict | None = None,
    label_rows: list[dict] | None = None,
    active_play_order: int | None = None,
    show_all_text: bool = False,
) -> None:
    rows = label_rows or []
    total = len(rows)
    if total == 0:
        raise ValueError("label_rows must not be empty")

    config = label_config or DEFAULT_LABEL_OVERLAY.copy()
    configured_size = int(config.get("font_size") or 0)
    fontsize = configured_size if configured_size > 0 else max(28, height // 32)
    line_gap = int(config.get("line_gap", DEFAULT_LABEL_OVERLAY["line_gap"]))
    left_margin = int(config.get("left_margin", DEFAULT_LABEL_OVERLAY["left_margin"]))
    top_margin = int(config.get("top_margin", DEFAULT_LABEL_OVERLAY["top_margin"]))
    random_color = bool(config.get("random_color"))
    fixed_color = None if random_color else parse_label_color(str(config.get("color", "yellow")))
    numbering = str(config.get("numbering", DEFAULT_LABEL_OVERLAY["numbering"]))
    number_color = (255, 255, 255, 255)

    line_height = fontsize + line_gap
    font = ImageFont.truetype(str(font_path), fontsize)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for line_num in range(1, total + 1):
        y = top_margin + (line_num - 1) * line_height
        if show_all_text:
            play_order = play_order_for_line(line_num, total, numbering)
            label = rows[play_order - 1]["label"]
            line_text = f"{line_num}. {label}"
            fill = pick_label_color_for_line(
                line_num,
                random_color=random_color,
                fixed_color=fixed_color,
            )
        elif active_play_order is not None:
            if line_should_show_text(line_num, active_play_order, total, numbering):
                play_order = play_order_for_line(line_num, total, numbering)
                label = rows[play_order - 1]["label"]
                line_text = f"{line_num}. {label}"
                fill = pick_label_color_for_line(
                    line_num,
                    random_color=random_color,
                    fixed_color=fixed_color,
                )
            else:
                line_text = f"{line_num}."
                fill = number_color
        else:
            line_text = f"{line_num}."
            fill = number_color

        draw.text((left_margin, y), line_text, font=font, fill=fill)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def build_merge_label_rows(
    selected: list[dict],
    *,
    numbering: str = "reverse",
) -> list[dict]:
    total = len(selected)
    rows: list[dict] = []
    for play_order, record in enumerate(selected, 1):
        label = (record.get("display_label") or "").strip()
        if not label:
            label = fallback_display_label(record.get("text") or "")
        rows.append(
            {
                "play_order": play_order,
                "line_number": play_order,
                "text_line": active_text_line(play_order, total, numbering),
                "label": label,
            }
        )
    return rows


def create_label_overlay_png(
    *,
    width: int,
    height: int,
    font_path: Path,
    label_config: dict,
    label_rows: list[dict],
    active_play_order: int | None = None,
    show_all_text: bool = False,
) -> Path:
    png_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    png_file.close()
    png_path = Path(png_file.name)
    render_label_overlay_png(
        width=width,
        height=height,
        font_path=font_path,
        output_path=png_path,
        label_config=label_config,
        label_rows=label_rows,
        active_play_order=active_play_order,
        show_all_text=show_all_text,
    )
    return png_path


def build_label_overlay_assets(
    *,
    width: int,
    height: int,
    font_path: Path,
    label_config: dict,
    label_rows: list[dict],
) -> tuple[list[Path], list[tuple[Path | None, Path | None]]]:
    display_mode = label_config.get("display_mode", "synced")
    if display_mode == "all":
        shared_png = create_label_overlay_png(
            width=width,
            height=height,
            font_path=font_path,
            label_config=label_config,
            label_rows=label_rows,
            show_all_text=True,
        )
        clip_pngs = [shared_png] * len(label_rows)
        bridge_pngs = [(shared_png, shared_png)] * max(0, len(label_rows) - 1)
        return clip_pngs, bridge_pngs

    clip_pngs = [
        create_label_overlay_png(
            width=width,
            height=height,
            font_path=font_path,
            label_config=label_config,
            label_rows=label_rows,
            active_play_order=row["play_order"],
        )
        for row in label_rows
    ]
    bridge_pngs = [
        (clip_pngs[index], clip_pngs[index + 1])
        for index in range(len(clip_pngs) - 1)
    ]
    return clip_pngs, bridge_pngs



def compose_video_filter(width: int, height: int, fps: int) -> str:
    return f"{scale_fill_vf(width, height)},fps={fps},format=yuv420p"


def extract_video_frame(
    video_path: Path,
    position: str,
    *,
    width: int,
    height: int,
) -> Path:
    frame_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    frame_file.close()
    frame_path = Path(frame_file.name)
    fill = scale_fill_vf(width, height)

    if position == "first":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"{fill},select=eq(n\\,0)",
            "-vframes",
            "1",
            "-pix_fmt",
            "rgb24",
            str(frame_path),
        ]
    elif position == "last":
        duration = probe_duration_seconds(video_path)
        seek = max(0.0, duration - FRAME_TRIM_SEC * 2)
        attempts = [
            [
                "ffmpeg",
                "-y",
                "-sseof",
                "-0.001",
                "-i",
                str(video_path),
                "-vf",
                fill,
                "-update",
                "1",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                str(frame_path),
            ],
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{seek:.6f}",
                "-i",
                str(video_path),
                "-vf",
                fill,
                "-vframes",
                "1",
                "-pix_fmt",
                "rgb24",
                str(frame_path),
            ],
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"{fill},select='eq(n,n_forced-1)'",
                "-vsync",
                "vfr",
                "-vframes",
                "1",
                "-pix_fmt",
                "rgb24",
                str(frame_path),
            ],
        ]
        last_error = ""
        for cmd in attempts:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0:
                return frame_path
            last_error = result.stderr[-400:]
        raise RuntimeError(
            f"Failed to extract last frame from {video_path.name}: {last_error}"
        )
    else:
        raise ValueError(f"Unknown frame position: {position}")

    if position == "first":
        _run_ffmpeg(cmd)
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to extract first frame from {video_path.name}")
        return frame_path

    raise RuntimeError(f"Unhandled frame position: {position}")


def build_fade_frame_filter(width: int, height: int) -> str:
    return f"format=rgb24,{scale_fill_vf(width, height)}"


def generate_fade_bridge_clip(
    output_path: Path,
    prev_video: Path,
    next_video: Path,
    *,
    width: int,
    height: int,
    fade_out_sec: float,
    fade_in_sec: float,
    fps: int,
    label_png_out: Path | None = None,
    label_png_in: Path | None = None,
) -> None:
    last_frame = extract_video_frame(
        prev_video,
        "last",
        width=width,
        height=height,
    )
    first_frame = extract_video_frame(
        next_video,
        "first",
        width=width,
        height=height,
    )
    temp_frames = [last_frame, first_frame]
    total_duration = fade_out_sec + fade_in_sec
    frame_vf = build_fade_frame_filter(width, height)
    canvas = f"color=c=black:s={width}x{height}:r={fps}"
    fade_out_label = label_png_out
    fade_in_label = label_png_in if label_png_in is not None else label_png_out
    label_input: list[str] = []
    out_label_index = 5
    in_label_index = 5

    if fade_out_label is not None:
        label_input.extend(["-loop", "1", "-i", str(fade_out_label)])
        if fade_in_label is not None and fade_in_label != fade_out_label:
            label_input.extend(["-loop", "1", "-i", str(fade_in_label)])
            in_label_index = 6

    if fade_out_label is not None:
        v0_chain = (
            f"[0:v]{frame_vf}[fg0];"
            f"[2:v][fg0]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1,"
            f"fade=t=out:st=0:d={fade_out_sec:.6f},"
            f"fps={fps},format=yuv420p[v0base];"
            f"[v0base][{out_label_index}:v]overlay=0:0:format=auto,format=yuv420p[v0];"
        )
        v1_chain = (
            f"[1:v]{frame_vf}[fg1];"
            f"[3:v][fg1]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1,"
            f"fade=t=in:st=0:d={fade_in_sec:.6f},"
            f"fps={fps},format=yuv420p[v1base];"
            f"[v1base][{in_label_index}:v]overlay=0:0:format=auto,format=yuv420p[v1];"
        )
    else:
        v0_chain = (
            f"[0:v]{frame_vf}[fg0];"
            f"[2:v][fg0]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1,"
            f"fade=t=out:st=0:d={fade_out_sec:.6f},"
            f"fps={fps},format=yuv420p[v0];"
        )
        v1_chain = (
            f"[1:v]{frame_vf}[fg1];"
            f"[3:v][fg1]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1,"
            f"fade=t=in:st=0:d={fade_in_sec:.6f},"
            f"fps={fps},format=yuv420p[v1];"
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
                "-framerate",
                str(fps),
                "-t",
                f"{fade_out_sec:.3f}",
                "-i",
                str(last_frame),
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{fade_in_sec:.3f}",
                "-i",
                str(first_frame),
                "-f",
                "lavfi",
                "-i",
                f"{canvas}:d={fade_out_sec:.3f}",
                "-f",
                "lavfi",
                "-i",
                f"{canvas}:d={fade_in_sec:.3f}",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={OUTPUT_AUDIO_RATE}:cl=stereo",
                *label_input,
                "-filter_complex",
                f"{v0_chain}{v1_chain}[v0][v1]concat=n=2:v=1:a=0,fps={fps},format=yuv420p[v]",
                "-map",
                "[v]",
                "-map",
                "4:a",
                *ffmpeg_encode_args(fps),
                "-t",
                f"{total_duration:.3f}",
                "-shortest",
                str(output_path),
            ]
        )
    finally:
        for frame_path in temp_frames:
            frame_path.unlink(missing_ok=True)


def normalize_clip_for_merge(
    source: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    fps: int = OUTPUT_FPS,
    label_png: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    source_duration = probe_duration_seconds(source)
    playable = source_duration - trim_start - trim_end
    if playable <= 0:
        raise ValueError(
            f"Clip {source.name} is too short after trimming "
            f"({trim_start:.3f}s start, {trim_end:.3f}s end)."
        )

    base_vf = compose_video_filter(width, height, fps)
    cmd: list[str] = ["ffmpeg", "-y"]
    if trim_start > 0:
        cmd.extend(["-ss", f"{trim_start:.6f}"])
    cmd.extend(["-i", str(source)])

    if label_png is not None:
        cmd.extend(["-loop", "1", "-i", str(label_png)])
        cmd.extend(
            [
                "-filter_complex",
                f"[0:v]{base_vf}[base];[base][1:v]overlay=0:0:format=auto,format=yuv420p[vout]",
                "-map",
                "[vout]",
                "-map",
                "0:a?",
                *ffmpeg_encode_args(fps),
                "-t",
                f"{playable:.6f}",
                str(output_path),
            ]
        )
    else:
        cmd.extend(
            [
                "-vf",
                base_vf,
                *ffmpeg_encode_args(fps),
                "-t",
                f"{playable:.6f}",
                str(output_path),
            ]
        )
    _run_ffmpeg(cmd)


def concat_video_pieces(
    pieces: list[Path],
    output_path: Path,
    *,
    fps: int = OUTPUT_FPS,
) -> None:
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
                *ffmpeg_encode_args(fps),
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


def copy_single_clip(
    source: Path,
    output_path: Path,
    *,
    label_png: Path | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int = OUTPUT_FPS,
) -> None:
    if width is None or height is None:
        width, height = merge_target_size([source])
    normalize_clip_for_merge(
        source,
        output_path,
        width=width,
        height=height,
        fps=fps,
        label_png=label_png,
    )


def merge_videos_with_ffmpeg(
    video_paths: list[Path],
    output_path: Path,
    transition_config: dict,
    selected: list[dict],
) -> list[dict]:
    if not video_paths:
        raise ValueError("No videos to merge")

    if len(video_paths) == 1:
        width, height = merge_target_size(video_paths)
        label_config = transition_config.get("clip_label_overlay") or DEFAULT_LABEL_OVERLAY
        label_rows = build_merge_label_rows(
            selected,
            numbering=str(label_config.get("numbering", "reverse")),
        )
        label_font = resolve_label_font()
        clip_pngs, _bridge_pngs = build_label_overlay_assets(
            width=width,
            height=height,
            font_path=label_font,
            label_config=label_config,
            label_rows=label_rows,
        )
        try:
            copy_single_clip(
                video_paths[0],
                output_path,
                label_png=clip_pngs[0],
                width=width,
                height=height,
                fps=int(transition_config["transition_fps"]),
            )
        finally:
            for png_path in set(clip_pngs):
                png_path.unlink(missing_ok=True)
        return []

    gap_duration = float(transition_config["gap_duration_sec"])
    fade_out_sec = float(transition_config["fade_out_sec"])
    fade_in_sec = float(transition_config["fade_in_sec"])
    transition_fps = int(transition_config["transition_fps"])
    frame_trim_sec = 1.0 / transition_fps
    fade_out_frames = bridge_frame_count(fade_out_sec, transition_fps)
    fade_in_frames = bridge_frame_count(fade_in_sec, transition_fps)
    width, height = merge_target_size(video_paths)
    label_config = transition_config.get("clip_label_overlay") or DEFAULT_LABEL_OVERLAY
    label_font = resolve_label_font()
    label_rows = build_merge_label_rows(
        selected,
        numbering=str(label_config.get("numbering", "reverse")),
    )
    clip_label_pngs, bridge_label_pngs = build_label_overlay_assets(
        width=width,
        height=height,
        font_path=label_font,
        label_config=label_config,
        label_rows=label_rows,
    )

    temp_cleanup: list[Path] = list(dict.fromkeys(clip_label_pngs))
    normalized_clips: list[Path] = []
    transition_log: list[dict] = []

    log.progress(
        f"Normalizing {len(video_paths)} clip(s) to {width}x{height} @ {transition_fps}fps "
        f"(trim {frame_trim_sec:.4f}s at each join to avoid duplicate frames)"
    )
    for index, clip_path in enumerate(video_paths):
        trim_start = frame_trim_sec if index > 0 else 0.0
        trim_end = frame_trim_sec if index < len(video_paths) - 1 else 0.0
        norm_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        norm_file.close()
        norm_path = Path(norm_file.name)
        normalize_clip_for_merge(
            clip_path,
            norm_path,
            width=width,
            height=height,
            trim_start=trim_start,
            trim_end=trim_end,
            fps=transition_fps,
            label_png=clip_label_pngs[index],
        )
        normalized_clips.append(norm_path)
        temp_cleanup.append(norm_path)

    pieces: list[Path] = []
    for index, clip_path in enumerate(normalized_clips):
        pieces.append(clip_path)

        if index >= len(normalized_clips) - 1:
            continue

        bridge_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        bridge_file.close()
        bridge_path = Path(bridge_file.name)
        temp_cleanup.append(bridge_path)

        log.progress(
            f"Fade bridge {index + 1}/{len(normalized_clips) - 1} after clip {index + 1}: "
            f"fade-out {fade_out_sec:.2f}s ({fade_out_frames} frames), "
            f"fade-in {fade_in_sec:.2f}s ({fade_in_frames} frames) @ {transition_fps}fps"
        )

        bridge_out_label, bridge_in_label = bridge_label_pngs[index]
        generate_fade_bridge_clip(
            bridge_path,
            video_paths[index],
            video_paths[index + 1],
            width=width,
            height=height,
            fade_out_sec=fade_out_sec,
            fade_in_sec=fade_in_sec,
            fps=transition_fps,
            label_png_out=bridge_out_label,
            label_png_in=bridge_in_label,
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
                "exit_sec": fade_out_sec,
                "enter_sec": fade_in_sec,
                "fade_out_frames": fade_out_frames,
                "fade_in_frames": fade_in_frames,
                "transition_fps": transition_fps,
                "frame_trim_sec": frame_trim_sec,
                "bridge_duration_sec": gap_duration,
                "fade_out_label": {
                    "play_order": label_rows[index]["play_order"],
                    "text_line": label_rows[index]["text_line"],
                    "label": label_rows[index]["label"],
                },
                "fade_in_label": {
                    "play_order": label_rows[index + 1]["play_order"],
                    "text_line": label_rows[index + 1]["text_line"],
                    "label": label_rows[index + 1]["label"],
                },
                "mode": transition_config.get("mode", "frame_fade_bridge"),
            }
        )

    concat_video_pieces(pieces, output_path, fps=transition_fps)

    for temp_path in temp_cleanup:
        temp_path.unlink(missing_ok=True)

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
                "display_label": record.get("display_label")
                or fallback_display_label(record.get("text") or ""),
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
        "merge_mode": transition_config.get("mode", "frame_fade_bridge"),
        "fade_out_sec": transition_config.get("fade_out_sec"),
        "fade_in_sec": transition_config.get("fade_in_sec"),
        "exit_sec": transition_config.get("exit_sec"),
        "enter_sec": transition_config.get("enter_sec"),
        "transition_fps": transition_config.get("transition_fps"),
        "output_fps": transition_config.get("output_fps"),
        "frame_trim_sec": (
            1.0 / transition_config["transition_fps"]
            if transition_config.get("transition_fps")
            else FRAME_TRIM_SEC
        ),
        "clip_labels": build_merge_label_rows(
            selected,
            numbering=str(
                (transition_config.get("clip_label_overlay") or {}).get(
                    "numbering", "reverse"
                )
            ),
        ),
        "clip_label_overlay": {
            "method": "png_overlay",
            "layout": "inline",
            "display_mode": (transition_config.get("clip_label_overlay") or {}).get(
                "display_mode", "synced"
            ),
            "numbering": (transition_config.get("clip_label_overlay") or {}).get(
                "numbering", "reverse"
            ),
            "font": str(resolve_label_font()),
            **(transition_config.get("clip_label_overlay") or DEFAULT_LABEL_OVERLAY),
        },
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
        f"(fade bridge: {transition_config['fade_out_sec']:.2f}s fade-out + "
        f"{transition_config['fade_in_sec']:.2f}s fade-in @ "
        f"{transition_config['transition_fps']}fps, no overlap)"
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

    log.section(f"Merging {len(selected)} clip(s) with fade-bridge transitions")
    label_config = transition_config.get("clip_label_overlay") or DEFAULT_LABEL_OVERLAY
    label_rows = build_merge_label_rows(
        selected,
        numbering=str(label_config.get("numbering", "reverse")),
    )
    label_preview = " | ".join(
        f"clip {row['play_order']} -> line {row['text_line']}: {row['label']}"
        for row in label_rows
    )
    log.info("Clip labels overlay (cumulative numbered list)", label_preview)
    for i, record in enumerate(selected, 1):
        log.bullet(f"[{i}] {record['clip_key']}")
        preview = (record.get("text") or "")[:80]
        if preview:
            log.bullet(f"    {preview}{'...' if len(record.get('text') or '') > 80 else ''}")

    transition_log = merge_videos_with_ffmpeg(
        video_paths,
        output_video,
        transition_config,
        selected,
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
        fps = int(transition_config["transition_fps"])
        fade_out_frames = bridge_frame_count(
            float(transition_config["fade_out_sec"]),
            fps,
        )
        fade_in_frames = bridge_frame_count(
            float(transition_config["fade_in_sec"]),
            fps,
        )
        log.info(
            "Fade bridges",
            f"{len(transition_log)} x frame fade bridge @ {fps}fps "
            f"({fade_out_frames} fade-out + {fade_in_frames} fade-in frames)",
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
        "[6] MERGE CLIPS BY CATEGORY",
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
