"""
[3] Remove on-screen text (English / Hindi) from split clip MP4s in output_videos/.

Uses EasyOCR to detect text regions and OpenCV inpainting to erase them frame by
frame. Replaces each clip in place while keeping the original audio track.

Tracks completed clips in output_videos/.text_removed_processed.json so reruns
skip videos that were already cleaned.

Preferred: double-click run.bat and choose 3.

Manual:
  conda run -n utube_env python 03_remove_text_from_videos.py
  conda run -n utube_env python 03_remove_text_from_videos.py --latest-only
  conda run -n utube_env python 03_remove_text_from_videos.py --video "output_videos/.../part-01-....mp4"
  conda run -n utube_env python 03_remove_text_from_videos.py --force
  conda run -n utube_env python 03_remove_text_from_videos.py --device cpu
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import easyocr
import numpy as np

import log_utils as log

log.configure_stdio()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output_videos"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "Transcriptions"
CONFIG_PATH = ROOT / "remove_text_config.json"
PROCESSED_TRACKER_PATH = OUTPUT_DIR / ".text_removed_processed.json"
FAILED_TRACKER_PATH = OUTPUT_DIR / ".text_removed_failed.json"
DEFAULT_LANGUAGES = ("en", "hi")
DEFAULT_DETECT_INTERVAL_SEC = 0.5
DEFAULT_BOX_PADDING_PX = 14
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_INPAINT_RADIUS = 5
DEFAULT_VIDEO_CRF = 18
DEFAULT_VIDEO_PRESET = "fast"


@dataclass
class RemoveTextSettings:
    languages: tuple[str, ...]
    detect_interval_sec: float
    box_padding_px: int
    min_confidence: float
    inpaint_radius: int
    ocr_gpu: bool
    video_crf: int
    video_preset: str


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


@dataclass
class TimedBoxes:
    time_sec: float
    boxes: list[tuple[int, int, int, int]]


def ensure_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if not missing:
        return
    raise RuntimeError(
        f"{', '.join(missing)} not found on PATH. Install ffmpeg and rerun from run.bat."
    )


def load_config(config_path: Path) -> RemoveTextSettings:
    if not config_path.exists():
        log.warn(f"Config not found, using defaults: {config_path.name}")
        return RemoveTextSettings(
            languages=DEFAULT_LANGUAGES,
            detect_interval_sec=DEFAULT_DETECT_INTERVAL_SEC,
            box_padding_px=DEFAULT_BOX_PADDING_PX,
            min_confidence=DEFAULT_MIN_CONFIDENCE,
            inpaint_radius=DEFAULT_INPAINT_RADIUS,
            ocr_gpu=True,
            video_crf=DEFAULT_VIDEO_CRF,
            video_preset=DEFAULT_VIDEO_PRESET,
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    languages = tuple(data.get("languages") or DEFAULT_LANGUAGES)
    return RemoveTextSettings(
        languages=languages,
        detect_interval_sec=float(data.get("detect_interval_sec", DEFAULT_DETECT_INTERVAL_SEC)),
        box_padding_px=int(data.get("box_padding_px", DEFAULT_BOX_PADDING_PX)),
        min_confidence=float(data.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
        inpaint_radius=int(data.get("inpaint_radius", DEFAULT_INPAINT_RADIUS)),
        ocr_gpu=bool(data.get("ocr_gpu", True)),
        video_crf=int(data.get("video_crf", DEFAULT_VIDEO_CRF)),
        video_preset=str(data.get("video_preset", DEFAULT_VIDEO_PRESET)),
    )


@lru_cache(maxsize=2)
def get_ocr_reader(languages: tuple[str, ...], use_gpu: bool) -> easyocr.Reader:
    try:
        return easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)
    except Exception:
        if use_gpu:
            log.warn("EasyOCR GPU init failed; falling back to CPU.")
            return easyocr.Reader(list(languages), gpu=False, verbose=False)
        raise


def video_key(video_path: Path) -> str:
    return video_path.relative_to(OUTPUT_DIR).as_posix()


def load_processed_tracker() -> dict:
    if not PROCESSED_TRACKER_PATH.exists():
        return {"videos": {}}

    try:
        data = json.loads(PROCESSED_TRACKER_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warn(f"Invalid tracker file, starting fresh: {PROCESSED_TRACKER_PATH.name}")
        return {"videos": {}}

    if not isinstance(data, dict):
        return {"videos": {}}
    data.setdefault("videos", {})
    return data


def save_processed_tracker(tracker: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TRACKER_PATH.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


def load_failed_tracker() -> dict:
    if not FAILED_TRACKER_PATH.exists():
        return {"videos": {}}

    try:
        data = json.loads(FAILED_TRACKER_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"videos": {}}

    if not isinstance(data, dict):
        return {"videos": {}}
    data.setdefault("videos", {})
    return data


def save_failed_tracker(tracker: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not tracker.get("videos"):
        if FAILED_TRACKER_PATH.exists():
            FAILED_TRACKER_PATH.unlink(missing_ok=True)
        return
    FAILED_TRACKER_PATH.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


def is_video_processed(video_path: Path, tracker: dict) -> bool:
    return video_key(video_path) in tracker.get("videos", {})


def mark_video_processed(
    video_path: Path,
    tracker: dict,
    payload: dict,
    failed_tracker: dict | None = None,
) -> None:
    key = video_key(video_path)
    tracker.setdefault("videos", {})[key] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": str(video_path),
        **payload,
    }
    save_processed_tracker(tracker)

    if failed_tracker is not None:
        clear_video_failed(video_path, failed_tracker)


def mark_video_failed(video_path: Path, tracker: dict, error: str) -> None:
    key = video_key(video_path)
    tracker.setdefault("videos", {})[key] = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "source": str(video_path),
        "error": error,
    }
    save_failed_tracker(tracker)


def clear_video_failed(video_path: Path, tracker: dict) -> None:
    key = video_key(video_path)
    videos = tracker.setdefault("videos", {})
    if key in videos:
        del videos[key]
        save_failed_tracker(tracker)


def list_output_videos() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    clips: list[Path] = []
    for path in OUTPUT_DIR.rglob("*.mp4"):
        if TRANSCRIPTIONS_DIR in path.parents:
            continue
        clips.append(path)
    return sorted(clips, key=lambda p: p.stat().st_mtime)


def resolve_video_arg(video_arg: Path) -> Path:
    path = video_arg
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_arg}")
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Not an MP4 file: {video_arg}")
    try:
        path.relative_to(OUTPUT_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Video must be inside output_videos/: {video_arg}") from exc
    return path


def pick_videos(
    video_arg: Path | None,
    skip_processed: bool,
    tracker: dict,
) -> list[Path]:
    if video_arg is not None:
        return [resolve_video_arg(video_arg)]

    all_videos = list_output_videos()
    if not skip_processed:
        return all_videos

    return [video for video in all_videos if not is_video_processed(video, tracker)]


def print_video_status(
    all_videos: list[Path],
    processed_tracker: dict,
    failed_tracker: dict,
) -> None:
    log.section("Output clip text-removal status")
    log.info("Processed tracker", PROCESSED_TRACKER_PATH)
    log.info("Failed tracker", FAILED_TRACKER_PATH)
    for video in all_videos:
        key = video_key(video)
        if is_video_processed(video, processed_tracker):
            status = "PROCESSED (skip)"
        elif key in failed_tracker.get("videos", {}):
            err = failed_tracker["videos"][key].get("error", "unknown error")
            status = f"PENDING (last attempt failed: {err[:80]})"
        else:
            status = "PENDING (text not removed yet)"
        log.bullet(f"{key}  ->  {status}")


def probe_video_info(video_path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-select_streams",
            "v:0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or [{}]
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)

    fps_text = str(stream.get("r_frame_rate") or "30/1")
    if "/" in fps_text:
        num, den = fps_text.split("/", 1)
        fps = float(num) / float(den or 1)
    else:
        fps = float(fps_text)

    frame_count = int(stream.get("nb_frames") or 0)
    duration_sec = float(stream.get("duration") or 0)
    if frame_count <= 0 and duration_sec > 0 and fps > 0:
        frame_count = max(1, int(round(duration_sec * fps)))

    if frame_count <= 0:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if duration_sec <= 0:
            duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def polygon_to_box(polygon: list, width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    xs = [int(round(p[0])) for p in polygon]
    ys = [int(round(p[1])) for p in polygon]
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(width - 1, max(xs) + padding)
    y2 = min(height - 1, max(ys) + padding)
    return x1, y1, x2, y2


def detect_text_boxes(
    reader: easyocr.Reader,
    frame: np.ndarray,
    min_confidence: float,
    padding: int,
) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    detections = reader.readtext(frame, detail=1, paragraph=False)
    boxes: list[tuple[int, int, int, int]] = []
    for item in detections:
        if len(item) < 3:
            continue
        polygon, _text, confidence = item[0], item[1], float(item[2])
        if confidence < min_confidence:
            continue
        if not polygon:
            continue
        box = polygon_to_box(polygon, width, height, padding)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.append(box)
    return boxes


def merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i, box_a in enumerate(merged):
            if used[i]:
                continue
            x1, y1, x2, y2 = box_a
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                overlaps = not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2)
                if overlaps:
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used[j] = True
                    changed = True
            used[i] = True
            next_boxes.append((x1, y1, x2, y2))
        merged = next_boxes
    return merged


def boxes_to_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def inpaint_frame(frame: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if not np.any(mask):
        return frame
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)


def sample_detection_times(duration_sec: float, interval_sec: float) -> list[float]:
    if duration_sec <= 0:
        return [0.0]
    times = list(np.arange(0.0, duration_sec, interval_sec))
    if not times or times[-1] < max(0.0, duration_sec - 0.05):
        times.append(max(0.0, duration_sec - 0.05))
    return times


def build_timed_boxes(
    reader: easyocr.Reader,
    video_path: Path,
    info: VideoInfo,
    settings: RemoveTextSettings,
) -> list[TimedBoxes]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    timed_boxes: list[TimedBoxes] = []
    sample_times = sample_detection_times(info.duration_sec, settings.detect_interval_sec)

    for sample_time in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        boxes = detect_text_boxes(
            reader,
            frame,
            settings.min_confidence,
            settings.box_padding_px,
        )
        boxes = merge_boxes(boxes)
        timed_boxes.append(TimedBoxes(time_sec=sample_time, boxes=boxes))

    cap.release()
    return timed_boxes


def boxes_for_time(timed_boxes: list[TimedBoxes], time_sec: float) -> list[tuple[int, int, int, int]]:
    if not timed_boxes:
        return []
    chosen = timed_boxes[0]
    for item in timed_boxes:
        if item.time_sec <= time_sec:
            chosen = item
        else:
            break
    return chosen.boxes


def mux_audio(source_video: Path, cleaned_video: Path, output_video: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(cleaned_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:])


def remove_text_from_video(
    video_path: Path,
    reader: easyocr.Reader,
    settings: RemoveTextSettings,
) -> dict:
    info = probe_video_info(video_path)
    timed_boxes = build_timed_boxes(reader, video_path, info, settings)
    total_boxes = sum(len(item.boxes) for item in timed_boxes)

    if total_boxes == 0:
        return {
            "duration_sec": round(info.duration_sec, 3),
            "text_regions_found": 0,
            "frames_inpainted": 0,
            "changed": False,
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    with tempfile.TemporaryDirectory(prefix="remove_text_") as tmp_dir:
        temp_video = Path(tmp_dir) / "cleaned_noaudio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temp_video),
            fourcc,
            info.fps if info.fps > 0 else 30.0,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create temporary video writer")

        frame_index = 0
        inpainted_frames = 0
        fps = info.fps if info.fps > 0 else 30.0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            time_sec = frame_index / fps
            boxes = boxes_for_time(timed_boxes, time_sec)
            if boxes:
                mask = boxes_to_mask(frame.shape[:2], boxes)
                frame = inpaint_frame(frame, mask, settings.inpaint_radius)
                inpainted_frames += 1

            writer.write(frame)
            frame_index += 1

        writer.release()
        cap.release()

        temp_with_audio = Path(tmp_dir) / "cleaned_with_audio.mp4"
        mux_audio(video_path, temp_video, temp_with_audio)

        reencode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_with_audio),
            "-c:v",
            "libx264",
            "-preset",
            settings.video_preset,
            "-crf",
            str(settings.video_crf),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        result = subprocess.run(reencode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-800:])

    unique_regions = len({box for item in timed_boxes for box in item.boxes})
    return {
        "duration_sec": round(info.duration_sec, 3),
        "text_regions_found": unique_regions,
        "frames_inpainted": inpainted_frames,
        "changed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove on-screen English/Hindi text from split MP4 clips"
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Specific MP4 under output_videos/. Default: all unprocessed clips",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Process only the newest unprocessed MP4 in output_videos/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process videos even if they were already cleaned",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"JSON settings file (default: {CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="OCR device preference (default: auto)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.banner(
        "[3] REMOVE ON-SCREEN TEXT FROM CLIPS",
        "EasyOCR (English + Hindi) + OpenCV inpainting",
    )

    try:
        ensure_ffmpeg()
        log.ok("ffmpeg and ffprobe found on PATH")
    except RuntimeError as exc:
        log.fail(str(exc))
        return 1

    settings = load_config(args.config)
    if args.device == "cpu":
        settings = RemoveTextSettings(
            languages=settings.languages,
            detect_interval_sec=settings.detect_interval_sec,
            box_padding_px=settings.box_padding_px,
            min_confidence=settings.min_confidence,
            inpaint_radius=settings.inpaint_radius,
            ocr_gpu=False,
            video_crf=settings.video_crf,
            video_preset=settings.video_preset,
        )
    elif args.device == "cuda":
        settings = RemoveTextSettings(
            languages=settings.languages,
            detect_interval_sec=settings.detect_interval_sec,
            box_padding_px=settings.box_padding_px,
            min_confidence=settings.min_confidence,
            inpaint_radius=settings.inpaint_radius,
            ocr_gpu=True,
            video_crf=settings.video_crf,
            video_preset=settings.video_preset,
        )

    log.paths_block(
        "Folders and trackers",
        [
            ("Project root", ROOT),
            ("Input clips", f"{OUTPUT_DIR}\\**\\*.mp4"),
            ("Text removal config", args.config),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
            ("Failed tracker", FAILED_TRACKER_PATH),
        ],
    )

    skip_processed = not args.force
    tracker = load_processed_tracker()
    failed_tracker = load_failed_tracker()
    all_output_videos = list_output_videos()

    if all_output_videos:
        print_video_status(all_output_videos, tracker, failed_tracker)

    try:
        if args.video:
            videos = pick_videos(args.video, skip_processed=False, tracker=tracker)
        elif args.latest_only:
            pending = pick_videos(None, skip_processed=skip_processed, tracker=tracker)
            videos = pending[-1:] if pending else []
        else:
            videos = pick_videos(None, skip_processed=skip_processed, tracker=tracker)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not videos:
        log.ok("No unprocessed MP4 files found in output_videos/.")
        log.info("Tracking file", PROCESSED_TRACKER_PATH)
        log.info("Tip", "Use --force to clean a video again.")
        return 0

    log.summary(
        "Queue summary",
        [
            ("Clips to clean", str(len(videos))),
            ("OCR languages", ", ".join(settings.languages)),
            ("Detect interval", f"{settings.detect_interval_sec:.2f}s"),
            ("OCR device", "GPU" if settings.ocr_gpu else "CPU"),
        ],
    )
    for i, video in enumerate(videos, 1):
        log.bullet(f"[{i}] {video_key(video)}")

    log.section("Loading EasyOCR reader (first run may download models)")
    log.info("Languages", ", ".join(settings.languages))
    try:
        reader = get_ocr_reader(settings.languages, settings.ocr_gpu)
        log.ok("OCR reader ready")
    except Exception as exc:
        log.fail(f"Failed to load EasyOCR: {exc}")
        return 1

    processed_count = 0
    failed_count = 0
    changed_count = 0

    for i, video in enumerate(videos, 1):
        rel = video_key(video)
        log.step(i, len(videos), f"Removing text: {rel}")
        log.info("Source clip", video)
        try:
            log.progress("Detecting text regions and inpainting frames...")
            payload = remove_text_from_video(video, reader, settings)
            mark_video_processed(video, tracker, payload, failed_tracker)
            processed_count += 1
            if payload.get("changed"):
                changed_count += 1
                log.ok(
                    f"Removed {payload['text_regions_found']} text region(s) "
                    f"across {payload['frames_inpainted']} frame(s)"
                )
            else:
                log.ok("No on-screen text detected; clip left unchanged")
        except Exception as exc:
            failed_count += 1
            msg = str(exc)
            log.fail(f"{rel}: {msg}")
            mark_video_failed(video, failed_tracker, msg)

    log.done_block(
        "Text removal task",
        [
            ("Clips processed", processed_count),
            ("Clips changed", changed_count),
            ("Clips failed", failed_count),
            ("Output folder", OUTPUT_DIR),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
        ],
        success=failed_count == 0,
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
