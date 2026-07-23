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


def split_ranges(markers: list[MarkerHit], duration: float) -> list[tuple[int, float, float]]:
    times = [m.time_sec for m in markers]
    ranges: list[tuple[int, float, float]] = []

    for part_idx in range(len(times)):
        start = times[part_idx]
        end = times[part_idx + 1] if part_idx + 1 < len(times) else duration
        if end - start >= 0.2:
            ranges.append((part_idx + 1, start, end))

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


def process_video(video_path: Path, refs: dict[int, ReferenceImage]) -> int:
    print(f"\nProcessing: {video_path.name}")

    timeline = scan_video(video_path, refs)
    markers = find_markers(timeline)
    duration = get_video_duration(video_path)
    ranges = split_ranges(markers, duration)

    safe_stem = video_path.stem
    out_dir = OUTPUT_DIR / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(video_path),
        "duration_sec": duration,
        "markers": [
            {"number": m.number, "time_sec": m.time_sec, "score": m.score}
            for m in markers
        ],
        "parts": [],
    }

    for part_num, start, end in ranges:
        out_file = out_dir / f"part_{part_num:02d}.mp4"
        print(f"  Exporting part {part_num}: {start:.2f}s -> {end:.2f}s")
        export_part(video_path, out_file, start, end)
        manifest["parts"].append(
            {
                "part": part_num,
                "start_sec": start,
                "end_sec": end,
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
    args = parser.parse_args()

    try:
        refs = load_references()
    except Exception as exc:
        print(f"Reference load failed: {exc}", file=sys.stderr)
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
            total_parts += process_video(video, refs)
        except Exception as exc:
            print(f"FAILED {video.name}: {exc}", file=sys.stderr)
            return 1

    print(f"\nDone. Exported {total_parts} part(s) total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
