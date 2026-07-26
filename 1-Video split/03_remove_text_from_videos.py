"""
[3] Blur on-screen English text from split clip MP4s in output_videos/.

For each frame: detect English text with EasyOCR (GPU), blur those areas, save clip.
Tracks finished chunks in trackers/text_removed_processed.json (one-time per clip).

Preferred: run.bat -> 3
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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import easyocr
import numpy as np

import log_utils as log
import project_paths as paths

log.configure_stdio()

ROOT = paths.ROOT
OUTPUT_DIR = paths.OUTPUT_DIR
TRANSCRIPTIONS_DIR = paths.TRANSCRIPTIONS_DIR
CONFIG_PATH = paths.REMOVE_TEXT_CONFIG
PROCESSED_TRACKER_PATH = paths.TEXT_REMOVED_PROCESSED
FAILED_TRACKER_PATH = paths.TEXT_REMOVED_FAILED


@dataclass
class Settings:
    languages: tuple[str, ...] = ("en",)
    box_padding_px: int = 16
    min_confidence: float = 0.2
    blur_kernel: int = 71
    blur_sigma: float = 15.0
    blur_passes: int = 3
    ocr_gpu: bool = True
    ocr_scale: float = 0.75
    progress_every_frames: int = 30
    video_crf: int = 18
    video_preset: str = "fast"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def load_settings(config_path: Path) -> Settings:
    if not config_path.exists():
        return Settings()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    langs = tuple(data.get("languages") or ["en"])
    return Settings(
        languages=langs,
        box_padding_px=int(data.get("box_padding_px", 16)),
        min_confidence=float(data.get("min_confidence", 0.2)),
        blur_kernel=int(data.get("blur_kernel", 71)),
        blur_sigma=float(data.get("blur_sigma", 15.0)),
        blur_passes=max(1, int(data.get("blur_passes", 3))),
        ocr_gpu=bool(data.get("ocr_gpu", True)),
        ocr_scale=float(data.get("ocr_scale", 0.75)),
        progress_every_frames=int(data.get("progress_every_frames", 30)),
        video_crf=int(data.get("video_crf", 18)),
        video_preset=str(data.get("video_preset", "fast")),
    )


def ocr_language_groups(requested: tuple[str, ...]) -> list[tuple[str, ...]]:
    """EasyOCR: Chinese needs English; Hindi cannot share a reader with Chinese."""
    want = set(requested)
    groups: list[tuple[str, ...]] = []

    if "ch_sim" in want:
        groups.append(("en", "ch_sim"))
        want.discard("ch_sim")

    if "hi" in want:
        groups.append(("en", "hi"))
        want.discard("hi")
        want.discard("en")

    if "en" in want and not groups:
        groups.append(("en",))

    return groups or [("en",)]


@dataclass
class OcrEngine:
    readers: list[easyocr.Reader]
    labels: list[str]
    uses_gpu: bool


def load_ocr_engine(languages: tuple[str, ...], use_gpu: bool) -> OcrEngine:
    groups = ocr_language_groups(languages)
    readers: list[easyocr.Reader] = []
    labels: list[str] = []
    gpu = use_gpu

    for group in groups:
        label = "+".join(group)
        try:
            reader = easyocr.Reader(list(group), gpu=gpu, verbose=False)
        except Exception:
            if gpu:
                log.warn(f"GPU failed for [{label}]; using CPU.")
                reader = easyocr.Reader(list(group), gpu=False, verbose=False)
                gpu = False
            else:
                raise
        readers.append(reader)
        labels.append(label)

    return OcrEngine(readers=readers, labels=labels, uses_gpu=gpu and use_gpu)


def video_key(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR).as_posix()


def load_tracker(path: Path) -> dict:
    if not path.exists():
        return {"videos": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"videos": {}}
    data.setdefault("videos", {})
    return data


def save_tracker(path: Path, data: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def list_clips() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    out = []
    for p in OUTPUT_DIR.rglob("*.mp4"):
        if TRANSCRIPTIONS_DIR not in p.parents:
            out.append(p)
    return sorted(out, key=lambda x: x.stat().st_mtime)


def resolve_video(arg: Path) -> Path:
    path = (ROOT / arg if not arg.is_absolute() else arg).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {arg}")
    path.relative_to(OUTPUT_DIR.resolve())
    return path


def probe_video(path: Path) -> VideoInfo:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-select_streams", "v:0", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = (json.loads(r.stdout).get("streams") or [{}])[0]
    w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
    fps_t = str(stream.get("r_frame_rate") or "30/1")
    fps = float(fps_t.split("/")[0]) / float(fps_t.split("/")[1] or 1) if "/" in fps_t else float(fps_t)
    n = int(stream.get("nb_frames") or 0)
    dur = float(stream.get("duration") or 0)
    if n <= 0 and dur > 0:
        n = max(1, int(round(dur * fps)))
    if n <= 0:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n / fps if fps else 0
        cap.release()
    return VideoInfo(w, h, fps, n, dur)


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60)}m {int(sec % 60)}s" if sec >= 60 else f"{sec:.0f}s"


def find_text_boxes(
    engine: OcrEngine,
    frame: np.ndarray,
    settings: Settings,
) -> list[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    scale = max(0.25, min(settings.ocr_scale, 1.0))
    ocr_frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 0.999 else frame
    oh, ow = ocr_frame.shape[:2]
    sx, sy = w / ow, h / oh
    pad = settings.box_padding_px
    boxes: list[tuple[int, int, int, int]] = []

    for reader in engine.readers:
        for item in reader.readtext(ocr_frame, detail=1, paragraph=False):
            if len(item) < 3:
                continue
            poly, _txt, conf = item[0], item[1], float(item[2])
            if conf < settings.min_confidence or not poly:
                continue
            xs = [int(p[0] * sx) for p in poly]
            ys = [int(p[1] * sy) for p in poly]
            x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
            x2, y2 = min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def blur_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int]], settings: Settings) -> np.ndarray:
    if not boxes:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    k = max(3, settings.blur_kernel | 1)
    sigma = max(0.0, settings.blur_sigma)
    passes = max(1, settings.blur_passes)

    for x1, y1, x2, y2 in boxes:
        # Pad patch so large blur kernel is not clipped by small text boxes.
        margin = k // 2 + 4
        px1, py1 = max(0, x1 - margin), max(0, y1 - margin)
        px2, py2 = min(w - 1, x2 + margin), min(h - 1, y2 + margin)
        patch = out[py1 : py2 + 1, px1 : px2 + 1].copy()
        if patch.size == 0:
            continue

        pk = min(k, patch.shape[1] if patch.shape[1] % 2 else max(3, patch.shape[1] - 1))
        pk = max(3, pk | 1)
        ph = min(k, patch.shape[0] if patch.shape[0] % 2 else max(3, patch.shape[0] - 1))
        ph = max(3, ph | 1)

        blurred = patch
        for _ in range(passes):
            blurred = cv2.GaussianBlur(blurred, (pk, ph), sigma)

        out[py1 : py2 + 1, px1 : px2 + 1] = blurred

    return out


def mux_audio(original: Path, video_only: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_only), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy",
        "-shortest", "-movflags", "+faststart", str(dest),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])


def process_clip(path: Path, engine: OcrEngine, settings: Settings) -> dict:
    info = probe_video(path)
    log.info("Frames", f"{info.frame_count} ({info.duration_sec:.1f}s)")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    t0 = time.perf_counter()
    frames_with_text = 0
    text_hits = 0
    idx = 0

    with tempfile.TemporaryDirectory(prefix="blur_text_") as tmp:
        tmp_vid = Path(tmp) / "v.mp4"
        writer = cv2.VideoWriter(
            str(tmp_vid), cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps or 30.0, (info.width, info.height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot create temp video")

        total = max(1, info.frame_count)
        step = max(1, settings.progress_every_frames)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            boxes = find_text_boxes(engine, frame, settings)
            if boxes:
                frame = blur_boxes(frame, boxes, settings)
                frames_with_text += 1
                text_hits += len(boxes)

            writer.write(frame)
            idx += 1

            if idx == 1 or idx % step == 0 or idx == total:
                elapsed = time.perf_counter() - t0
                eta = (total - idx) / (idx / elapsed) if elapsed > 0 else 0
                log.progress(
                    f"Frame {idx}/{total} ({100 * idx / total:.0f}%)  "
                    f"blurred {frames_with_text} frame(s)  ETA {fmt_time(eta)}"
                )

        writer.release()
        cap.release()

        if frames_with_text == 0:
            return {
                "frame_count": idx,
                "frames_with_text": 0,
                "text_regions": 0,
                "languages": list(settings.languages),
                "blur_kernel": settings.blur_kernel,
                "blur_sigma": settings.blur_sigma,
                "blur_passes": settings.blur_passes,
                "processing_sec": round(time.perf_counter() - t0, 1),
                "changed": False,
            }

        tmp_out = Path(tmp) / "out.mp4"
        mux_audio(path, tmp_vid, tmp_out)
        cmd = [
            "ffmpeg", "-y", "-i", str(tmp_out),
            "-c:v", "libx264", "-preset", settings.video_preset, "-crf", str(settings.video_crf),
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-800:])

    return {
        "frame_count": idx,
        "frames_with_text": frames_with_text,
        "text_regions": text_hits,
        "languages": list(settings.languages),
        "blur_kernel": settings.blur_kernel,
        "blur_sigma": settings.blur_sigma,
        "blur_passes": settings.blur_passes,
        "processing_sec": round(time.perf_counter() - t0, 1),
        "changed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Blur on-screen text on every frame of split clips")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        log.fail("ffmpeg/ffprobe not found on PATH")
        return 1

    settings = load_settings(args.config)
    if args.device == "cpu":
        settings.ocr_gpu = False
    elif args.device == "cuda":
        settings.ocr_gpu = True

    log.banner("[3] BLUR ON-SCREEN TEXT (EVERY FRAME)", "EasyOCR + heavy Gaussian blur")

    tracker = load_tracker(PROCESSED_TRACKER_PATH)
    failed = load_tracker(FAILED_TRACKER_PATH)
    all_clips = list_clips()

    if args.video:
        clips = [resolve_video(args.video)]
    elif args.latest_only:
        pending = [c for c in all_clips if args.force or video_key(c) not in tracker["videos"]]
        clips = pending[-1:] if pending else []
    else:
        clips = all_clips if args.force else [c for c in all_clips if video_key(c) not in tracker["videos"]]

    if not clips:
        log.ok("Nothing to do — all clips already processed.")
        log.info("Tracker", PROCESSED_TRACKER_PATH)
        return 0

    log.summary("Queue", [
        ("Clips", str(len(clips))),
        ("Method", "every frame -> detect text -> blur"),
        ("Languages", ", ".join(settings.languages)),
        ("OCR groups", ", ".join("+".join(g) for g in ocr_language_groups(settings.languages))),
        ("Blur", f"kernel={settings.blur_kernel}, sigma={settings.blur_sigma}, passes={settings.blur_passes}"),
        ("Device", "GPU" if settings.ocr_gpu else "CPU"),
        ("Tracker", PROCESSED_TRACKER_PATH.name),
    ])

    log.section("Loading EasyOCR")
    engine = load_ocr_engine(settings.languages, settings.ocr_gpu)
    for label in engine.labels:
        log.bullet(f"{label} ({'GPU' if engine.uses_gpu else 'CPU'})")
    log.ok("OCR ready")

    ok_n = fail_n = 0
    for i, clip in enumerate(clips, 1):
        key = video_key(clip)
        log.step(i, len(clips), key)
        try:
            result = process_clip(clip, engine, settings)
            tracker["videos"][key] = {
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "source": str(clip),
                **result,
            }
            save_tracker(PROCESSED_TRACKER_PATH, tracker)
            if key in failed["videos"]:
                del failed["videos"][key]
                save_tracker(FAILED_TRACKER_PATH, failed)
            ok_n += 1
            if result["changed"]:
                log.ok(
                    f"Blurred text on {result['frames_with_text']}/{result['frame_count']} frames "
                    f"in {fmt_time(result['processing_sec'])}"
                )
            else:
                log.ok(f"No text found in {result['frame_count']} frames")
        except Exception as exc:
            fail_n += 1
            log.fail(str(exc))
            failed["videos"][key] = {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "source": str(clip),
                "error": str(exc),
            }
            save_tracker(FAILED_TRACKER_PATH, failed)

    log.done_block("Done", [
        ("Processed", ok_n), ("Failed", fail_n), ("Tracker", PROCESSED_TRACKER_PATH),
    ], success=fail_n == 0)
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
