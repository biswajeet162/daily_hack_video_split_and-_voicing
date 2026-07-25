"""
[2] Split videos by golden number markers (1-5) using computer vision.

Reads videos from input_videos/, matches reference_numbers/1.png .. 5.png,
then exports parts into output_videos/<video_name>/ as part-01-<uuid>.mp4, etc.
Tracks completed splits in input_videos/.split_processed.json so reruns skip
videos that were already split. Existing output folders are synced into that
tracker automatically.

Preferred: double-click run.bat and choose 2 (splits every unprocessed
video in input_videos/, one by one, skipping ones already done).

Manual:
  conda run -n utube_env python 02_split_video.py
  conda run -n utube_env python 02_split_video.py --latest-only
  conda run -n utube_env python 02_split_video.py --video "input_videos/my_video.mp4"
  conda run -n utube_env python 02_split_video.py --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import log_utils as log

log.configure_stdio()

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"
REF_DIR = ROOT / "reference_numbers"
TRIM_CONFIG_PATH = ROOT / "split_trim_config.json"
PROCESSED_TRACKER_PATH = INPUT_DIR / ".split_processed.json"
FAILED_TRACKER_PATH = INPUT_DIR / ".split_failed.json"

NUMBERS = list(range(1, 6))
SAMPLE_FPS = 4.0  # coarse scan: frames analyzed per second
REFINE_FPS = 10.0  # fine scan around each detected marker
MIN_SCORE = 0.06
MIN_SCORE_MARGIN = 0.03  # winner must beat runner-up by at least this
PEAK_WINDOW_SEC = 1.5
MIN_GAP_SEC = 2.0
MARKER_END_DROP_RATIO = 0.45  # card ended when score falls below peak * this


@dataclass
class ReferenceImage:
    number: int
    gray: np.ndarray
    descriptors: np.ndarray | None
    keypoint_count: int
    hist: np.ndarray


@dataclass
class TrimSettings:
    start_trim_sec: float
    end_trim_sec: float
    start_trim_from: str = "marker_end"  # "marker_start" or "marker_end"
    min_score_margin: float | None = None  # override default margin for this number


@dataclass
class MarkerHit:
    number: int
    time_sec: float
    end_sec: float
    score: float


@dataclass
class PartRange:
    part: int
    marker_number: int
    raw_start_sec: float
    raw_end_sec: float
    marker_end_sec: float
    start_sec: float
    end_sec: float
    start_trim_sec: float
    end_trim_sec: float
    start_trim_from: str


def load_references() -> dict[int, ReferenceImage]:
    orb = cv2.ORB_create(nfeatures=2000)
    refs: dict[int, ReferenceImage] = {}

    for n in NUMBERS:
        path = REF_DIR / f"{n}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing reference image: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read reference image: {path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = orb.detectAndCompute(gray, None)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        refs[n] = ReferenceImage(
            number=n,
            gray=gray,
            descriptors=descriptors,
            keypoint_count=max(len(orb.detect(gray, None)), 1),
            hist=hist,
        )

    return refs


def load_trim_config(config_path: Path = TRIM_CONFIG_PATH) -> dict[int, TrimSettings]:
    if not config_path.exists():
        defaults = {n: TrimSettings(0.5, 0.0 if n == 5 else 0.5) for n in NUMBERS}
        log.warn(f"Trim config not found, using defaults: {config_path.name}")
        return defaults

    data = json.loads(config_path.read_text(encoding="utf-8"))
    settings: dict[int, TrimSettings] = {}

    for n in NUMBERS:
        key = str(n)
        if key not in data:
            raise ValueError(f"split_trim_config.json missing entry for number {n}")

        entry = data[key]
        settings[n] = TrimSettings(
            start_trim_sec=float(entry.get("start_trim_sec", 0.5)),
            end_trim_sec=float(entry.get("end_trim_sec", 0.0 if n == 5 else 0.5)),
            start_trim_from=str(entry.get("start_trim_from", "marker_end")),
            min_score_margin=(
                float(entry["min_score_margin"])
                if "min_score_margin" in entry
                else None
            ),
        )

    log.ok(f"Loaded trim config: {config_path.name}")
    for n in NUMBERS:
        t = settings[n]
        margin = t.min_score_margin if t.min_score_margin is not None else MIN_SCORE_MARGIN
        log.bullet(
            f"#{n}: start +{t.start_trim_sec:.2f}s from {t.start_trim_from}, "
            f"end -{t.end_trim_sec:.2f}s, margin {margin:.2f}"
        )

    return settings


def center_crop(frame: np.ndarray, ratio: float = 0.75) -> np.ndarray:
    h, w = frame.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return frame[y0 : y0 + ch, x0 : x0 + cw]


def score_components(frame: np.ndarray, ref: ReferenceImage) -> tuple[float, float, float]:
    crop = center_crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (ref.gray.shape[1], ref.gray.shape[0]))

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    _, frame_des = orb.detectAndCompute(gray, None)

    orb_score = 0.0
    if frame_des is not None and ref.descriptors is not None:
        matches = bf.knnMatch(ref.descriptors, frame_des, k=2)
        good = 0
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good += 1
        orb_score = good / ref.keypoint_count

    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    hist_score = max(0.0, cv2.compareHist(ref.hist, hist, cv2.HISTCMP_CORREL))

    ref_edges = cv2.Canny(ref.gray, 50, 150)
    frame_edges = cv2.Canny(gray, 50, 150)
    template_score = float(
        cv2.matchTemplate(frame_edges, ref_edges, cv2.TM_CCOEFF_NORMED).max()
    )
    template_score = max(0.0, template_score)

    return orb_score, hist_score, template_score


def score_frame(frame: np.ndarray, ref: ReferenceImage) -> float:
    orb_score, hist_score, template_score = score_components(frame, ref)
    # Template edges help separate short cards like #4 from similar-looking scenes.
    return (0.55 * orb_score) + (0.15 * hist_score) + (0.30 * template_score)


def score_margin(scores: dict[int, float], number: int) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return ordered[0] if ordered else 0.0
    best = scores[number]
    runner_up = max(v for n, v in scores.items() if n != number)
    return best - runner_up


def scan_video(video_path: Path, refs: dict[int, ReferenceImage]) -> list[tuple[float, dict[int, float]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / SAMPLE_FPS)))

    timeline: list[tuple[float, dict[int, float]]] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            t = frame_idx / fps
            scores = {n: score_frame(frame, refs[n]) for n in NUMBERS}
            timeline.append((t, scores))

        frame_idx += 1

    cap.release()

    if not timeline:
        raise RuntimeError(f"No frames read from: {video_path}")

    duration = total_frames / fps
    log.progress(f"Scanned {len(timeline)} frame samples across {duration:.1f}s")
    return timeline


def read_frame_at(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def score_at_time(
    cap: cv2.VideoCapture,
    refs: dict[int, ReferenceImage],
    time_sec: float,
) -> dict[int, float]:
    frame = read_frame_at(cap, time_sec)
    if frame is None:
        return {n: 0.0 for n in NUMBERS}
    return {n: score_frame(frame, refs[n]) for n in NUMBERS}


def refine_marker_time(
    video_path: Path,
    refs: dict[int, ReferenceImage],
    number: int,
    coarse_time: float,
    min_margin: float,
) -> tuple[float, float]:
    """Fine-scan around coarse hit to locate the true peak for this number."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return coarse_time, 0.0

    best_time = coarse_time
    best_score = 0.0
    search_start = max(0.0, coarse_time - 2.0)
    search_end = coarse_time + 2.0
    step = 1.0 / REFINE_FPS

    t = search_start
    while t <= search_end:
        scores = score_at_time(cap, refs, t)
        score = scores[number]
        margin = score_margin(scores, number)
        winner = max(scores, key=scores.get)

        if winner == number and score > best_score and margin >= min_margin:
            best_score = score
            best_time = t

        t += step

    cap.release()
    return best_time, best_score


def find_marker_end(
    video_path: Path,
    refs: dict[int, ReferenceImage],
    number: int,
    peak_time: float,
    peak_score: float,
    duration: float,
) -> float:
    """Find when the number card finishes (score drops after the peak)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return peak_time + 0.8

    threshold = max(MIN_SCORE, peak_score * MARKER_END_DROP_RATIO)
    step = 1.0 / REFINE_FPS
    low_streak = 0
    end_time = peak_time

    t = peak_time
    while t <= min(duration, peak_time + 4.0):
        scores = score_at_time(cap, refs, t)
        if scores[number] < threshold:
            low_streak += 1
            if low_streak >= 2:
                end_time = max(peak_time, t - step)
                break
        else:
            low_streak = 0
            end_time = t
        t += step

    cap.release()
    return end_time


def find_markers(
    timeline: list[tuple[float, dict[int, float]]],
    video_path: Path,
    refs: dict[int, ReferenceImage],
    trim_config: dict[int, TrimSettings],
) -> list[MarkerHit]:
    hits: list[MarkerHit] = []
    last_time = -999.0

    for number in NUMBERS:
        min_margin = trim_config[number].min_score_margin
        if min_margin is None:
            min_margin = 0.025 if number == 4 else MIN_SCORE_MARGIN

        candidates: list[tuple[float, float, float]] = []

        def collect_candidates(required_margin: float) -> list[tuple[float, float, float]]:
            found: list[tuple[float, float, float]] = []
            for t, scores in timeline:
                score = scores[number]
                if score < MIN_SCORE:
                    continue

                winner = max(scores, key=scores.get)
                if winner != number:
                    continue

                margin = score_margin(scores, number)
                if margin < required_margin:
                    continue

                if t - last_time < MIN_GAP_SEC:
                    continue

                t0 = max(0.0, t - PEAK_WINDOW_SEC)
                t1 = t + PEAK_WINDOW_SEC
                local = [scores[number] for ts, scores in timeline if t0 <= ts <= t1]
                if score < max(local):
                    continue

                found.append((score, margin, t))
            return found

        candidates = collect_candidates(min_margin)
        if not candidates and min_margin > 0.02:
            relaxed = max(0.02, min_margin * 0.6)
            candidates = collect_candidates(relaxed)
            if candidates:
                log.warn(f"Marker {number}: using relaxed margin {relaxed:.3f}")

        if not candidates:
            raise RuntimeError(
                f"Could not detect number {number} in video. "
                f"Try lowering MIN_SCORE or check reference_numbers/{number}.png"
            )

        _, _, coarse_time = max(candidates, key=lambda x: x[0])
        refined_time, refined_score = refine_marker_time(
            video_path, refs, number, coarse_time, min_margin
        )
        if refined_score <= 0.0:
            refined_time, refined_score = coarse_time, max(c[0] for c in candidates)

        duration = get_video_duration(video_path)
        marker_end = find_marker_end(
            video_path, refs, number, refined_time, refined_score, duration
        )

        hits.append(
            MarkerHit(
                number=number,
                time_sec=refined_time,
                end_sec=marker_end,
                score=refined_score,
            )
        )
        last_time = refined_time
        log.ok(
            f"Marker {number} at {refined_time:.2f}s "
            f"(card ends {marker_end:.2f}s, confidence {refined_score:.3f})"
        )

    return hits


def split_ranges(
    markers: list[MarkerHit],
    duration: float,
    trim_config: dict[int, TrimSettings],
) -> list[PartRange]:
    ranges: list[PartRange] = []

    for part_idx, marker in enumerate(markers):
        part_num = part_idx + 1
        number = marker.number
        raw_start = marker.time_sec
        raw_end = markers[part_idx + 1].time_sec if part_idx + 1 < len(markers) else duration

        trim = trim_config[number]
        trim_anchor = marker.end_sec if trim.start_trim_from == "marker_end" else raw_start
        start = trim_anchor + trim.start_trim_sec
        end = raw_end - trim.end_trim_sec if part_num < len(markers) else raw_end

        # When the next marker is close (short cards like #4), also hide its card end.
        if part_num < len(markers):
            next_marker = markers[part_idx + 1]
            end = min(end, next_marker.time_sec - trim.end_trim_sec)

        # Keep valid, non-empty ranges after trimming
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start < 0.2:
            log.warn(
                f"Part {part_num} too short after trim "
                f"({start:.2f}s -> {end:.2f}s), using minimal bounds"
            )
            start = min(start, raw_end - 0.2)
            end = max(end, start + 0.2)

        ranges.append(
            PartRange(
                part=part_num,
                marker_number=number,
                raw_start_sec=raw_start,
                raw_end_sec=raw_end,
                marker_end_sec=marker.end_sec,
                start_sec=start,
                end_sec=end,
                start_trim_sec=trim.start_trim_sec,
                end_trim_sec=trim.end_trim_sec if part_num < len(markers) else 0.0,
                start_trim_from=trim.start_trim_from,
            )
        )

    return ranges


def part_filename(part_num: int) -> str:
    return f"part-{part_num:02d}-{uuid.uuid4()}.mp4"


def clear_previous_parts(out_dir: Path) -> None:
    """Remove old split parts so reruns replace output instead of keeping stale files."""
    if not out_dir.exists():
        return
    removed = 0
    for pattern in ("part-*.mp4", "part_*.mp4"):
        for old_file in out_dir.glob(pattern):
            old_file.unlink(missing_ok=True)
            removed += 1
    if removed:
        log.progress(f"Removed {removed} previous part file(s) from output folder")


def export_part(video_path: Path, out_path: Path, start: float, end: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    # Accurate frame cuts (stream copy seeks to keyframes and can leave number cards in).
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
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
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:])


def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps


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
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not tracker.get("videos"):
        if FAILED_TRACKER_PATH.exists():
            FAILED_TRACKER_PATH.unlink(missing_ok=True)
        return
    FAILED_TRACKER_PATH.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


def mark_video_failed(video_path: Path, tracker: dict, error: str) -> None:
    tracker.setdefault("videos", {})[video_path.name] = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "source": str(video_path),
        "error": error,
    }
    save_failed_tracker(tracker)


def clear_video_failed(video_path: Path, tracker: dict) -> None:
    videos = tracker.setdefault("videos", {})
    if video_path.name in videos:
        del videos[video_path.name]
        save_failed_tracker(tracker)


def video_status(video_path: Path, processed_tracker: dict) -> str:
    if is_video_processed(video_path, processed_tracker):
        return "processed"
    return "pending"


def print_input_video_status(all_videos: list[Path], processed_tracker: dict, failed_tracker: dict) -> None:
    log.section("Input video status")
    log.info("Processed tracker", PROCESSED_TRACKER_PATH)
    log.info("Failed tracker", FAILED_TRACKER_PATH)
    for video in all_videos:
        if is_video_processed(video, processed_tracker):
            status = "PROCESSED (skip)"
        elif video.name in failed_tracker.get("videos", {}):
            err = failed_tracker["videos"][video.name].get("error", "unknown error")
            status = f"PENDING (last split failed: {err[:80]})"
        else:
            status = "PENDING (not split yet)"
        log.bullet(f"{video.name}  ->  {status}")


def output_manifest_path(video_path: Path) -> Path:
    return OUTPUT_DIR / video_path.stem / "split_manifest.json"


def is_video_processed(video_path: Path, tracker: dict) -> bool:
    if video_path.name in tracker.get("videos", {}):
        return True
    return output_manifest_path(video_path).exists()


def sync_tracker_from_output(tracker: dict, videos: list[Path]) -> dict:
    """Backfill tracker entries from existing output folders/manifests."""
    videos_dict = tracker.setdefault("videos", {})
    changed = False

    for video in videos:
        if video.name in videos_dict:
            continue

        manifest_path = output_manifest_path(video)
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        videos_dict[video.name] = {
            "processed_at": "synced_from_output",
            "source": manifest.get("source", str(video)),
            "output_dir": str(manifest_path.parent),
            "manifest": str(manifest_path),
            "parts_count": len(manifest.get("parts", [])),
            "synced_from_output": True,
        }
        changed = True

    if changed:
        save_processed_tracker(tracker)
        log.ok(f"Synced processed history to {PROCESSED_TRACKER_PATH.name}")

    return tracker


def mark_video_processed(
    video_path: Path,
    tracker: dict,
    parts_count: int,
    out_dir: Path,
    failed_tracker: dict | None = None,
) -> None:
    tracker.setdefault("videos", {})[video_path.name] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": str(video_path),
        "output_dir": str(out_dir),
        "manifest": str(out_dir / "split_manifest.json"),
        "parts_count": parts_count,
    }
    save_processed_tracker(tracker)
    if failed_tracker is not None:
        clear_video_failed(video_path, failed_tracker)


def pick_videos(
    explicit: Path | None,
    *,
    skip_processed: bool = True,
    tracker: dict | None = None,
) -> list[Path]:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return [explicit]

    videos = sorted(INPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {INPUT_DIR}")

    if not skip_processed or tracker is None:
        return videos

    pending: list[Path] = []
    for video in videos:
        if is_video_processed(video, tracker):
            log.bullet(f"SKIP (already split)  {video.name}")
        else:
            pending.append(video)

    skipped = len(videos) - len(pending)
    if skipped:
        log.progress(f"Skipped {skipped} already processed video(s)")

    return pending


def process_video(
    video_path: Path,
    refs: dict[int, ReferenceImage],
    trim_config: dict[int, TrimSettings],
    tracker: dict | None = None,
    failed_tracker: dict | None = None,
    *,
    step_index: int = 1,
    step_total: int = 1,
) -> int:
    log.step(step_index, step_total, f"Splitting video: {video_path.name}")
    log.info("Source file", video_path)
    log.info("Reference images", REF_DIR)
    log.info("Trim config", TRIM_CONFIG_PATH)

    log.section("Step A - Scan video for number markers (1-5)")
    timeline = scan_video(video_path, refs)

    log.section("Step B - Match markers against reference_numbers/")
    markers = find_markers(timeline, video_path, refs, trim_config)
    duration = get_video_duration(video_path)
    ranges = split_ranges(markers, duration, trim_config)

    safe_stem = video_path.stem
    out_dir = OUTPUT_DIR / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_parts(out_dir)
    log.info("Output folder", out_dir)

    manifest = {
        "source": str(video_path),
        "duration_sec": duration,
        "trim_config": str(TRIM_CONFIG_PATH),
        "markers": [
            {
                "number": m.number,
                "time_sec": m.time_sec,
                "end_sec": m.end_sec,
                "score": m.score,
            }
            for m in markers
        ],
        "parts": [],
    }

    log.section(f"Step C - Export {len(ranges)} part(s) with ffmpeg")
    exported_files: list[str] = []

    for part in ranges:
        out_name = part_filename(part.part)
        out_file = out_dir / out_name
        log.bullet(
            f"Part {part.part} -> {out_name}  |  "
            f"{part.start_sec:.2f}s to {part.end_sec:.2f}s  |  "
            f"marker #{part.marker_number} at {part.raw_start_sec:.2f}s"
        )
        export_part(video_path, out_file, part.start_sec, part.end_sec)
        exported_files.append(out_name)
        manifest["parts"].append(
            {
                "part": part.part,
                "filename": out_name,
                "marker_number": part.marker_number,
                "raw_start_sec": part.raw_start_sec,
                "raw_end_sec": part.raw_end_sec,
                "marker_end_sec": part.marker_end_sec,
                "start_trim_sec": part.start_trim_sec,
                "end_trim_sec": part.end_trim_sec,
                "start_trim_from": part.start_trim_from,
                "start_sec": part.start_sec,
                "end_sec": part.end_sec,
                "file": str(out_file),
            }
        )

    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if tracker is not None:
        mark_video_processed(video_path, tracker, len(ranges), out_dir, failed_tracker)
        log.ok(f"Marked as processed in {PROCESSED_TRACKER_PATH.name}")

    log.section("Split output for this video")
    log.info("Output folder", out_dir)
    log.info("Manifest", manifest_path)
    for name in exported_files:
        log.bullet(name)
    log.ok(f"Saved {len(ranges)} part(s)")
    return len(ranges)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split videos by golden number markers using OpenCV matching"
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Specific video file. Default: all/latest mp4 in input_videos/",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Process only the newest unprocessed video in input_videos/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process videos even if they were already split before",
    )
    parser.add_argument(
        "--trim-config",
        type=Path,
        default=TRIM_CONFIG_PATH,
        help=f"JSON trim settings per number (default: {TRIM_CONFIG_PATH.name})",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.banner(
        "[2] SPLIT VIDEOS BY NUMBER MARKERS",
        "Detect golden numbers 1-5 and export parts into output_videos/",
    )
    log.paths_block(
        "Folders and trackers",
        [
            ("Project root", ROOT),
            ("Input folder", INPUT_DIR),
            ("Reference images", REF_DIR),
            ("Output folder", OUTPUT_DIR),
            ("Trim config", args.trim_config),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
            ("Failed tracker", FAILED_TRACKER_PATH),
        ],
    )

    try:
        log.section("Loading reference number images")
        refs = load_references()
        for n in NUMBERS:
            log.bullet(f"reference_numbers/{n}.png")
    except Exception as exc:
        log.fail(f"Reference load failed: {exc}")
        return 1

    try:
        log.section("Loading trim settings")
        trim_config = load_trim_config(args.trim_config)
    except Exception as exc:
        log.fail(f"Trim config load failed: {exc}")
        return 1

    skip_processed = not args.force
    tracker = load_processed_tracker()
    failed_tracker = load_failed_tracker()
    all_input_videos = sorted(INPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if skip_processed:
        tracker = sync_tracker_from_output(tracker, all_input_videos)

    if all_input_videos:
        print_input_video_status(all_input_videos, tracker, failed_tracker)

    try:
        if args.video:
            videos = pick_videos(args.video, skip_processed=False)
        elif args.latest_only:
            pending = pick_videos(None, skip_processed=skip_processed, tracker=tracker)
            videos = pending[-1:] if pending else []
        else:
            videos = pick_videos(None, skip_processed=skip_processed, tracker=tracker)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not videos:
        log.ok("No unprocessed videos found in input_videos/.")
        log.info("Tracking file", PROCESSED_TRACKER_PATH)
        log.info("Tip", "Use --force to split a video again.")
        return 0

    log.summary(
        "Queue summary",
        [
            ("Videos to split", str(len(videos))),
        ],
    )
    for i, video in enumerate(videos, 1):
        log.bullet(f"[{i}] {video.name}")

    total_parts = 0
    processed_count = 0
    failed_count = 0

    for i, video in enumerate(videos, 1):
        try:
            total_parts += process_video(
                video,
                refs,
                trim_config,
                tracker,
                failed_tracker,
                step_index=i,
                step_total=len(videos),
            )
            processed_count += 1
        except Exception as exc:
            failed_count += 1
            msg = str(exc)
            log.fail(f"{video.name}: {msg}")
            mark_video_failed(video, failed_tracker, msg)

    log.done_block(
        "Split task",
        [
            ("Videos split", processed_count),
            ("Videos failed", failed_count),
            ("Parts exported", total_parts),
            ("Output folder", OUTPUT_DIR),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
        ],
        success=failed_count == 0,
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
