"""
[3] Transcribe split videos in output_videos/ to Hindi using faster-whisper.

Scans output_videos/ recursively for *.mp4, transcribes speech in Hindi,
and writes a JSON file next to each video (<name>.transcription.json).
Tracks completed transcriptions in output_videos/.transcribe_processed.json
so reruns skip videos that were already transcribed.

Uses the same faster-whisper settings as 2-HOLO_GRAM_VOICE_OVER (large-v3,
VAD filter, beam search) for much better Hindi accuracy than openai-whisper base.

Preferred: double-click run.bat and choose 3 (transcribes every unprocessed
.mp4 in output_videos/, one by one).

Manual:
  conda run -n utube_env python 03_transcribe_videos.py
  conda run -n utube_env python 03_transcribe_videos.py --latest-only
  conda run -n utube_env python 03_transcribe_videos.py --video "output_videos/.../part-01-....mp4"
  conda run -n utube_env python 03_transcribe_videos.py --force
  conda run -n utube_env python 03_transcribe_videos.py --model large-v3 --device cpu --compute int8
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

import log_utils as log

log.configure_stdio()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output_videos"
PROCESSED_TRACKER_PATH = OUTPUT_DIR / ".transcribe_processed.json"
FAILED_TRACKER_PATH = OUTPUT_DIR / ".transcribe_failed.json"
DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "hi"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BEAM_SIZE = 5
DEFAULT_VAD_FILTER = True


def ensure_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if not missing:
        return
    raise RuntimeError(
        f"{', '.join(missing)} not found on PATH. faster-whisper needs ffmpeg/ffprobe "
        "to read MP4 audio. Install ffmpeg and make sure it is available in the same shell as run.bat."
    )


@lru_cache(maxsize=2)
def get_whisper_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        if device != "cpu":
            log.warn("CUDA init failed; falling back to CPU int8.")
            return WhisperModel(model_name, device="cpu", compute_type="int8")
        raise

def transcription_path(video_path: Path) -> Path:
    return video_path.with_suffix(".transcription.json")


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
    key = video_key(video_path)
    if key in tracker.get("videos", {}):
        return True
    return transcription_path(video_path).exists()


def sync_tracker_from_output(tracker: dict, videos: list[Path]) -> dict:
    """Backfill tracker entries from existing transcription JSON files."""
    videos_dict = tracker.setdefault("videos", {})
    changed = False

    for video in videos:
        key = video_key(video)
        if key in videos_dict:
            continue

        transcript_path = transcription_path(video)
        if not transcript_path.exists():
            continue

        try:
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        videos_dict[key] = {
            "processed_at": payload.get("processed_at", "synced_from_output"),
            "source": payload.get("source", str(video)),
            "transcription": str(transcript_path),
            "duration_sec": payload.get("duration_sec"),
            "synced_from_output": True,
        }
        changed = True

    if changed:
        save_processed_tracker(tracker)
        log.ok(f"Synced transcription history to {PROCESSED_TRACKER_PATH.name}")

    return tracker


def mark_video_processed(
    video_path: Path,
    tracker: dict,
    transcript_path: Path,
    duration_sec: float,
    failed_tracker: dict | None = None,
) -> None:
    key = video_key(video_path)
    tracker.setdefault("videos", {})[key] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": str(video_path),
        "transcription": str(transcript_path),
        "duration_sec": round(duration_sec, 3),
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
    return sorted(OUTPUT_DIR.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)


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
    log.section("Output clip transcription status")
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
            status = "PENDING (not transcribed yet)"
        log.bullet(f"{key}  ->  {status}")


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


def build_transcription_payload(
    video_path: Path,
    segments: list,
    duration_sec: float,
    model_name: str,
    language: str,
    device: str,
    compute_type: str,
) -> dict:
    segment_rows = []
    texts: list[str] = []

    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            texts.append(text)
        segment_rows.append(
            {
                "start_sec": round(float(seg.start), 3),
                "end_sec": round(float(seg.end), 3),
                "duration_sec": round(float(seg.end) - float(seg.start), 3),
                "text": text,
            }
        )

    full_text = " ".join(texts).strip()
    end_sec = duration_sec
    if segment_rows:
        end_sec = min(max(float(seg.end) for seg in segments), duration_sec)

    return {
        "source": str(video_path),
        "language": language,
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "start_sec": 0.0,
        "end_sec": round(end_sec, 3),
        "duration_sec": round(end_sec, 3),
        "text": full_text,
        "segments": segment_rows,
    }


def transcribe_video(
    video_path: Path,
    model_name: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    vad_filter: bool = DEFAULT_VAD_FILTER,
) -> dict:
    duration_sec = probe_duration_seconds(video_path)
    model = get_whisper_model(model_name, device, compute_type)
    segments_iter, _info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=False,
        vad_filter=vad_filter,
    )
    segments = list(segments_iter)
    return build_transcription_payload(
        video_path,
        segments,
        duration_sec,
        model_name,
        language,
        device,
        compute_type,
    )


def save_transcription(video_path: Path, payload: dict) -> Path:
    out_path = transcription_path(video_path)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe split MP4 clips in output_videos/ to Hindi using faster-whisper"
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Specific MP4 under output_videos/. Default: all unprocessed clips",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Transcribe only the newest unprocessed MP4 in output_videos/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe videos even if a transcription JSON already exists",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Transcription language code (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Inference device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--compute",
        default=DEFAULT_COMPUTE_TYPE,
        help=f"Compute type for faster-whisper (default: {DEFAULT_COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Beam size for decoding (default: {DEFAULT_BEAM_SIZE})",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering (enabled by default)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.banner(
        "[3] TRANSCRIBE SPLIT CLIPS TO HINDI",
        "faster-whisper large-v3 with VAD filter and beam search",
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
            ("Input clips", f"{OUTPUT_DIR}\\**\\*.mp4"),
            ("Output files", f"{OUTPUT_DIR}\\**\\*.transcription.json"),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
            ("Failed tracker", FAILED_TRACKER_PATH),
        ],
    )

    skip_processed = not args.force
    tracker = load_processed_tracker()
    failed_tracker = load_failed_tracker()
    all_output_videos = list_output_videos()

    if skip_processed:
        tracker = sync_tracker_from_output(tracker, all_output_videos)

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
        log.info("Tip", "Use --force to transcribe a video again.")
        return 0

    log.summary(
        "Queue summary",
        [
            ("Clips to transcribe", str(len(videos))),
            ("Language", args.language),
            ("Model", args.model),
            ("Device", args.device),
            ("Compute", args.compute),
            ("Beam size", str(args.beam)),
            ("VAD filter", "off" if args.no_vad else "on"),
        ],
    )
    for i, video in enumerate(videos, 1):
        log.bullet(f"[{i}] {video_key(video)}")

    log.section("Loading faster-whisper model (first run may download weights)")
    log.info("Model", args.model)
    log.info("Device", args.device)
    log.info("Compute type", args.compute)
    try:
        get_whisper_model(args.model, args.device, args.compute)
        log.ok("Model ready")
    except Exception as exc:
        log.fail(f"Failed to load faster-whisper model '{args.model}': {exc}")
        return 1

    processed_count = 0
    failed_count = 0

    for i, video in enumerate(videos, 1):
        rel = video_key(video)
        log.step(i, len(videos), f"Transcribing clip: {rel}")
        log.info("Source clip", video)
        log.info("Output JSON", transcription_path(video))
        try:
            log.progress("Reading audio and running speech recognition...")
            payload = transcribe_video(
                video,
                model_name=args.model,
                language=args.language,
                device=args.device,
                compute_type=args.compute,
                beam_size=args.beam,
                vad_filter=not args.no_vad,
            )
            out_path = save_transcription(video, payload)
            mark_video_processed(
                video,
                tracker,
                out_path,
                payload["duration_sec"],
                failed_tracker,
            )
            preview = payload["text"][:120]
            if len(payload["text"]) > 120:
                preview += "..."
            log.ok(f"Duration {payload['start_sec']:.2f}s -> {payload['end_sec']:.2f}s")
            log.info("Transcript", preview)
            log.info("Saved to", out_path)
            log.ok(f"Marked as processed in {PROCESSED_TRACKER_PATH.name}")
            processed_count += 1
        except Exception as exc:
            msg = str(exc)
            log.fail(msg)
            mark_video_failed(video, failed_tracker, msg)
            failed_count += 1

    log.done_block(
        "Transcription task",
        [
            ("Transcribed", processed_count),
            ("Failed", failed_count),
            ("Total attempted", len(videos)),
            ("Output folder", OUTPUT_DIR),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
        ],
        success=failed_count == 0,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
