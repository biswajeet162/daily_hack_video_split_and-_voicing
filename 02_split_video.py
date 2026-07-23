"""
[2] Split videos by golden number markers (1-5) using computer vision.

Reads videos from input_videos/, matches reference_numbers/1.png .. 5.png,
then exports parts into output_videos/<video_name>/.

Preferred: double-click run.bat and choose 2.

Manual:
  conda run -n utube_env python 02_split_video.py
  conda run -n utube_env python 02_split_video.py --video "input_videos/my_video.mp4"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_videos"
OUTPUT_DIR = ROOT / "output_videos"
REF_DIR = ROOT / "reference_numbers"
TRIM_CONFIG_PATH = ROOT / "split_trim_config.json"

NUMBERS = list(range(1, 6))
SAMPLE_FPS = 4.0  # frames analyzed per second of video
MIN_SCORE = 0.06
PEAK_WINDOW_SEC = 1.5
MIN_GAP_SEC = 2.0


@dataclass
class ReferenceImage:
    number: int
    gray: np.ndarray
    descriptors: np.ndarray | None
    keypoint_count: int
    hist: np.ndarray


@dataclass
class MarkerHit:
    number: int
    time_sec: float
    score: float


@dataclass
class TrimSettings:
    start_trim_sec: float
    end_trim_sec: float


@dataclass
class PartRange:
    part: int
    marker_number: int
    raw_start_sec: float
    raw_end_sec: float
    start_sec: float
    end_sec: float
    start_trim_sec: float
    end_trim_sec: float


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
        print(f"  Trim config not found, using defaults: {config_path.name}")
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
        )

    print(f"  Loaded trim config: {config_path.name}")
    for n in NUMBERS:
        t = settings[n]
        print(f"    #{n}: start +{t.start_trim_sec:.2f}s, end -{t.end_trim_sec:.2f}s")

    return settings


def center_crop(frame: np.ndarray, ratio: float = 0.75) -> np.ndarray:
    h, w = frame.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return frame[y0 : y0 + ch, x0 : x0 + cw]


def score_frame(frame: np.ndarray, ref: ReferenceImage) -> float:
    crop = center_crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (ref.gray.shape[1], ref.gray.shape[0]))

    # ORB feature similarity (robust to lighting / compression)
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

    # Histogram similarity (captures overall look and feel)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    hist_score = max(0.0, cv2.compareHist(ref.hist, hist, cv2.HISTCMP_CORREL))

    # Weighted blend favors ORB but keeps histogram as tie-breaker
    return (0.75 * orb_score) + (0.25 * hist_score)


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
    print(f"  Scanned {len(timeline)} samples across {duration:.1f}s")
    return timeline


def find_markers(timeline: list[tuple[float, dict[int, float]]]) -> list[MarkerHit]:
    hits: list[MarkerHit] = []
    last_time = -999.0

    for number in NUMBERS:
        candidates: list[tuple[float, float]] = []

        for i, (t, scores) in enumerate(timeline):
            score = scores[number]
            if score < MIN_SCORE:
                continue

            winner = max(scores, key=scores.get)
            if winner != number:
                continue

            if t - last_time < MIN_GAP_SEC:
                continue

            # Local peak in a small time window
            t0 = max(0.0, t - PEAK_WINDOW_SEC)
            t1 = t + PEAK_WINDOW_SEC
            local = [
                scores[number]
                for ts, scores in timeline
                if t0 <= ts <= t1
            ]
            if score < max(local):
                continue

            candidates.append((score, t))

        if not candidates:
            raise RuntimeError(
                f"Could not detect number {number} in video. "
                f"Try lowering MIN_SCORE or check reference_numbers/{number}.png"
            )

        best_score, best_time = max(candidates, key=lambda x: x[0])
        hits.append(MarkerHit(number=number, time_sec=best_time, score=best_score))
        last_time = best_time
        print(f"  Found marker {number} at {best_time:.2f}s (confidence {best_score:.3f})")

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
        start = raw_start + trim.start_trim_sec
        end = raw_end - trim.end_trim_sec if part_num < len(markers) else raw_end

        # Keep valid, non-empty ranges after trimming
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start < 0.2:
            print(
                f"  Warning: part {part_num} too short after trim "
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
                start_sec=start,
                end_sec=end,
                start_trim_sec=trim.start_trim_sec,
                end_trim_sec=trim.end_trim_sec if part_num < len(markers) else 0.0,
            )
        )

    return ranges


def export_part(video_path: Path, out_path: Path, start: float, end: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: re-encode if stream copy fails around non-keyframe cuts
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
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


def pick_videos(explicit: Path | None) -> list[Path]:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return [explicit]

    videos = sorted(INPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {INPUT_DIR}")
    return videos


def process_video(
    video_path: Path,
    refs: dict[int, ReferenceImage],
    trim_config: dict[int, TrimSettings],
) -> int:
    print(f"\nProcessing: {video_path.name}")

    timeline = scan_video(video_path, refs)
    markers = find_markers(timeline)
    duration = get_video_duration(video_path)
    ranges = split_ranges(markers, duration, trim_config)

    safe_stem = video_path.stem
    out_dir = OUTPUT_DIR / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(video_path),
        "duration_sec": duration,
        "trim_config": str(TRIM_CONFIG_PATH),
        "markers": [
            {"number": m.number, "time_sec": m.time_sec, "score": m.score}
            for m in markers
        ],
        "parts": [],
    }

    for part in ranges:
        out_file = out_dir / f"part_{part.part:02d}.mp4"
        print(
            f"  Exporting part {part.part}: "
            f"{part.start_sec:.2f}s -> {part.end_sec:.2f}s "
            f"(raw {part.raw_start_sec:.2f}s -> {part.raw_end_sec:.2f}s, "
            f"trim +{part.start_trim_sec:.2f}s / -{part.end_trim_sec:.2f}s)"
        )
        export_part(video_path, out_file, part.start_sec, part.end_sec)
        manifest["parts"].append(
            {
                "part": part.part,
                "marker_number": part.marker_number,
                "raw_start_sec": part.raw_start_sec,
                "raw_end_sec": part.raw_end_sec,
                "start_trim_sec": part.start_trim_sec,
                "end_trim_sec": part.end_trim_sec,
                "start_sec": part.start_sec,
                "end_sec": part.end_sec,
                "file": str(out_file),
            }
        )

    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Saved {len(ranges)} parts to: {out_dir}")
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
        help="Process only the newest video in input_videos/",
    )
    parser.add_argument(
        "--trim-config",
        type=Path,
        default=TRIM_CONFIG_PATH,
        help=f"JSON trim settings per number (default: {TRIM_CONFIG_PATH.name})",
    )
    args = parser.parse_args()

    try:
        refs = load_references()
    except Exception as exc:
        print(f"Reference load failed: {exc}", file=sys.stderr)
        return 1

    try:
        trim_config = load_trim_config(args.trim_config)
    except Exception as exc:
        print(f"Trim config load failed: {exc}", file=sys.stderr)
        return 1

    try:
        if args.video:
            videos = pick_videos(args.video)
        elif args.latest_only:
            videos = pick_videos(None)[:1]
        else:
            videos = pick_videos(None)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total_parts = 0
    for video in videos:
        try:
            total_parts += process_video(video, refs, trim_config)
        except Exception as exc:
            print(f"FAILED {video.name}: {exc}", file=sys.stderr)
            return 1

    print(f"\nDone. Exported {total_parts} part(s) total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
