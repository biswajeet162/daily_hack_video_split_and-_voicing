import json
import subprocess
from pathlib import Path
from typing import Callable, Optional


VIDEO_FILE = Path("story.mp4")
TRANSCRIPT_FILE = Path("transcript.json")
OUTPUT_DIR = Path("split_videos")


def run_ffmpeg(video_file: Path, start_sec: float, duration_sec: float, output_file: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_file),
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    subprocess.run(cmd, check=True)


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
    dur = float(data.get("format", {}).get("duration", 0) or 0)
    if dur <= 0:
        raise ValueError(f"Unable to read duration for: {media_path}")
    return dur


def split_video_by_chunk_size(
    video_file: Path,
    chunk_seconds: int,
    output_dir: Path,
    min_chunk_seconds: float = 0.1,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if not video_file.exists():
        raise FileNotFoundError(f"Missing video file: {video_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration_seconds(video_file)
    total = int((duration + chunk_seconds - 1e-9) // chunk_seconds)
    if total <= 0:
        raise ValueError("Video duration is too short")

    kept = 0
    for idx in range(total):
        chunk_id = idx + 1
        start = idx * chunk_seconds
        end = min(duration, (idx + 1) * chunk_seconds)
        dur = max(0.1, end - start)
        if dur < min_chunk_seconds:
            print(
                f"- skip chunk {chunk_id:03d}: {start:.2f}s -> {end:.2f}s "
                f"(duration {dur:.2f}s < {min_chunk_seconds:.2f}s)"
            )
            if on_progress:
                on_progress(idx + 1, total)
            continue
        out = output_dir / f"chunk_{chunk_id:03d}.mp4"
        run_ffmpeg(video_file, start, dur, out)
        kept += 1
        if on_progress:
            on_progress(chunk_id, total)
    return kept


def split_video_chunks(
    video_file: Path,
    transcript_file: Path,
    output_dir: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    if not video_file.exists():
        raise FileNotFoundError(f"Missing video file: {video_file}")

    if not transcript_file.exists():
        raise FileNotFoundError(f"Missing transcript file: {transcript_file}")

    with transcript_file.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("transcript.json must contain a non-empty list")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating chunk videos in: {output_dir.resolve()}")
    total = len(chunks)
    for idx, item in enumerate(chunks, start=1):
        chunk_id = int(item["chunk_id"])
        start = float(item["start"])
        end = float(item["end"])
        duration = max(0.1, end - start)

        output_file = output_dir / f"chunk_{chunk_id:03d}.mp4"
        print(f"- chunk {chunk_id:03d}: {start:.2f}s -> {end:.2f}s")
        run_ffmpeg(video_file, start, duration, output_file)
        if on_progress:
            on_progress(idx, total)

    print("Done. Chunk videos created successfully.")
    return total


def main() -> None:
    split_video_chunks(VIDEO_FILE, TRANSCRIPT_FILE, OUTPUT_DIR)


if __name__ == "__main__":
    main()
