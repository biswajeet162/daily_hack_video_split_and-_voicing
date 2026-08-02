import json
import shutil
import threading
import subprocess
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from split_video_chunks import (
    split_video_by_chunk_size,
    split_video_chunks,
    probe_duration_seconds,
)
from transcriber import transcribe_clip_text


PORT = 8000
VIDEO_FILE = Path("story.mp4")
TRANSCRIPT_FILE = Path("transcript.json")
SESSION_FILE = Path("session.json")
OUTPUT_DIR = Path("split_videos")
UPLOADS_DIR = Path("uploads")
RECORDINGS_DIR = Path("recordings")
RECORDINGS_FIXED_DIR = Path("recordings_fixed")
TRANSCRIPTIONS_DIR = Path("transcription_folder")
MERGED_PARTS_DIR = Path("merged_parts")
FINAL_OUTPUT = Path("final_output.mp4")
FINAL_WITH_BG = Path("final_output_bg.mp4")
BG_MUSIC_FILE = UPLOADS_DIR / "bg_music_track"
BG_MUSIC_VOLUME = 0.22
MIN_CHUNK_SECONDS = 2.0

# Record extra audio and then trim to exact chunk duration on server.
# Must match frontend values in app.js if you use pre/post buffering there.
PRE_BUFFER_SEC = 0.15
POST_BUFFER_SEC = 0.14
FIXED_SAMPLE_RATE = 48000
progress_lock = threading.Lock()
split_state = {
    "running": False,
    "current": 0,
    "total": 0,
    "done": False,
    "error": None,
}
process_state = {
    "running": False,
    "stage": "idle",
    "message": "Waiting",
    "current": 0,
    "total": 0,
    "done": False,
    "error": None,
    "job_id": 0,
}
active_video_file = VIDEO_FILE
session_mode = "split"  # "split" | "chunks"
_process_job_counter = 0


def _set_state(**kwargs):
    with progress_lock:
        split_state.update(kwargs)


def _get_state():
    with progress_lock:
        return dict(split_state)


def _set_process_state(**kwargs):
    with progress_lock:
        process_state.update(kwargs)


def _get_process_state():
    with progress_lock:
        return dict(process_state)


def _begin_process_job(*, stage: str, message: str, total: int = 0) -> int:
    """Mark a new job as running BEFORE the worker thread starts (avoids stale done=True races)."""
    global _process_job_counter
    with progress_lock:
        _process_job_counter += 1
        job_id = _process_job_counter
        process_state.update(
            {
                "running": True,
                "done": False,
                "error": None,
                "stage": stage,
                "message": message,
                "current": 0,
                "total": total,
                "job_id": job_id,
            }
        )
        return job_id


def _set_session_mode(mode: str) -> None:
    global session_mode
    if mode not in {"split", "chunks"}:
        raise ValueError(f"Invalid session mode: {mode}")
    session_mode = mode
    SESSION_FILE.write_text(json.dumps({"mode": mode}, indent=2), encoding="utf-8")


def _get_session_mode() -> str:
    if SESSION_FILE.exists():
        try:
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            mode = data.get("mode")
            if mode in {"split", "chunks"}:
                return mode
        except Exception:  # noqa: BLE001
            pass
    return session_mode


def _read_transcript() -> list:
    if not TRANSCRIPT_FILE.exists():
        raise FileNotFoundError(f"Missing transcript file: {TRANSCRIPT_FILE}")
    with TRANSCRIPT_FILE.open("r", encoding="utf-8") as f:
        transcript = json.load(f)
    if not isinstance(transcript, list) or not transcript:
        raise ValueError("transcript.json must contain a non-empty list")
    return transcript


def _write_transcript(transcript: list) -> None:
    with TRANSCRIPT_FILE.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)


def _write_chunk_transcript_files(item: dict) -> None:
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    seq = int(item["chunk_id"])
    text = item.get("text") or ""
    (TRANSCRIPTIONS_DIR / f"chunk_{seq:03d}.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (TRANSCRIPTIONS_DIR / f"chunk_{seq:03d}.txt").write_text(text, encoding="utf-8")


def _clear_work_dirs(*, keep_uploads: bool = False) -> None:
    for folder in (OUTPUT_DIR, TRANSCRIPTIONS_DIR, RECORDINGS_DIR, RECORDINGS_FIXED_DIR, MERGED_PARTS_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    if TRANSCRIPT_FILE.exists():
        TRANSCRIPT_FILE.unlink()
    if FINAL_OUTPUT.exists():
        FINAL_OUTPUT.unlink()
    if not keep_uploads and UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR)


def _chunk_video_path(chunk_id: int) -> Path:
    return (OUTPUT_DIR / f"chunk_{chunk_id:03d}.mp4").resolve()


def _recording_paths(chunk_id: int) -> tuple[Path, Path]:
    return (
        RECORDINGS_DIR / f"chunk_{chunk_id:03d}.webm",
        RECORDINGS_FIXED_DIR / f"chunk_{chunk_id:03d}.wav",
    )


def _run_split_job():
    def on_progress(current, total):
        _set_state(current=current, total=total)

    try:
        _set_state(running=True, done=False, error=None, current=0, total=0)
        count = split_video_chunks(VIDEO_FILE, TRANSCRIPT_FILE, OUTPUT_DIR, on_progress=on_progress)
        _set_state(running=False, done=True, total=count)
    except Exception as exc:  # noqa: BLE001
        _set_state(running=False, done=False, error=str(exc))


def _run_process_job(video_file: Path, chunk_size: int):
    def on_split_progress(current, total):
        _set_process_state(stage="splitting", message=f"Splitting video: {current}/{total}", current=current, total=total)

    try:
        _set_session_mode("split")
        _set_process_state(
            running=True,
            done=False,
            error=None,
            stage="splitting",
            message="Splitting selected video into chunks...",
            current=0,
            total=0,
        )

        # 1) Split video into chunks purely by chunk_size
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        kept = split_video_by_chunk_size(
            video_file,
            chunk_size,
            OUTPUT_DIR,
            min_chunk_seconds=MIN_CHUNK_SECONDS,
            on_progress=on_split_progress,
        )

        # 2) Transcribe each chunk sequentially and persist per-chunk transcript file
        if TRANSCRIPTIONS_DIR.exists():
            shutil.rmtree(TRANSCRIPTIONS_DIR)
        TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

        video_duration = probe_duration_seconds(video_file)
        transcript = []
        chunk_files = sorted(OUTPUT_DIR.glob("chunk_*.mp4"))
        total = len(chunk_files)
        _set_process_state(stage="transcribing", message=f"Transcribing chunks: 0/{total}", current=0, total=total)
        for seq, chunk_file in enumerate(chunk_files, start=1):
            # Preserve original timeline based on source chunk id in filename.
            # Example: chunk_004.mp4 means [18,24] if chunk_size=6.
            src_chunk_id = int(chunk_file.stem.split("_")[1])
            start = (src_chunk_id - 1) * chunk_size
            end = min(video_duration, src_chunk_id * chunk_size)
            text = transcribe_clip_text(str(chunk_file))

            item = {"chunk_id": seq, "start": start, "end": end, "text": text}
            _write_chunk_transcript_files(item)
            transcript.append(item)

            _set_process_state(
                stage="transcribing",
                message=f"Transcribing chunks: {seq}/{total}",
                current=seq,
                total=total,
            )

        # 3) Write transcript.json used by the frontend
        _write_transcript(transcript)

        _set_process_state(
            running=False,
            done=True,
            stage="done",
            message=(
                f"Pipeline complete. Generated {kept} kept video chunks "
                f"(ignored < {MIN_CHUNK_SECONDS:.0f}s) and {total} transcripts."
            ),
            current=total,
            total=total,
        )
    except Exception as exc:  # noqa: BLE001
        _set_process_state(
            running=False,
            done=False,
            stage="error",
            message="Pipeline failed.",
            error=str(exc),
        )


def _run_load_chunks_job(files: list[tuple[str, Path]]):
    try:
        _set_session_mode("chunks")
        _set_process_state(
            running=True,
            done=False,
            error=None,
            stage="loading",
            message="Loading selected video chunks...",
            current=0,
            total=len(files),
        )

        # Preserve staged files while clearing previous session outputs.
        staging = UPLOADS_DIR / "chunks_staging"
        staging_backup = None
        if staging.exists():
            staging_backup = UPLOADS_DIR / "_chunks_staging_keep"
            if staging_backup.exists():
                shutil.rmtree(staging_backup)
            shutil.move(str(staging), str(staging_backup))

        _clear_work_dirs(keep_uploads=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

        if staging_backup is not None:
            shutil.move(str(staging_backup), str(staging))

        transcript = []
        total = len(files)
        for seq, (name, staged_path) in enumerate(files, start=1):
            if not staged_path.exists():
                raise FileNotFoundError(f"Missing staged file: {staged_path}")

            ext = Path(name).suffix.lower() or ".mp4"
            if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                raise ValueError(f"Unsupported video extension for {name}")

            final_path = _chunk_video_path(seq)
            if ext == ".mp4":
                shutil.copy2(str(staged_path), str(final_path))
            else:
                temp_src = OUTPUT_DIR / f"_src_{seq:03d}{ext}"
                shutil.copy2(str(staged_path), str(temp_src))
                _run_ffmpeg([
                    "ffmpeg", "-y", "-i", str(temp_src),
                    "-c", "copy", "-movflags", "+faststart",
                    str(final_path),
                ])
                temp_src.unlink(missing_ok=True)

            duration = probe_duration_seconds(final_path)
            item = {
                "chunk_id": seq,
                "start": 0.0,
                "end": round(duration, 3),
                "text": "",
                "source_name": Path(name).name,
            }
            _write_chunk_transcript_files(item)
            transcript.append(item)

            _set_process_state(
                stage="loading",
                message=f"Loaded chunks: {seq}/{total}",
                current=seq,
                total=total,
            )

        _write_transcript(transcript)
        _set_process_state(
            running=False,
            done=True,
            stage="done",
            message=f"Loaded {total} video chunks. Click a chunk to transcribe and record.",
            current=total,
            total=total,
        )
    except Exception as exc:  # noqa: BLE001
        _set_process_state(
            running=False,
            done=False,
            stage="error",
            message="Failed to load video chunks.",
            error=str(exc),
        )


def _run_ffmpeg(cmd, *, capture_stderr: bool = False) -> str:
    if capture_stderr:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or "").strip() or f"ffmpeg failed (code={p.returncode})")
        return (p.stderr or "").strip()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ""


def _probe_duration_seconds(media_path: Path) -> float:
    """Return duration in seconds (best-effort)."""
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


def _normalize_recording_to_fixed_wav(chunk_id: int, source_audio: Path, chunk_dur: float) -> Path:
    pre_samples = int(round(PRE_BUFFER_SEC * FIXED_SAMPLE_RATE))
    post_samples = int(round(POST_BUFFER_SEC * FIXED_SAMPLE_RATE))
    dur_samples = int(round(chunk_dur * FIXED_SAMPLE_RATE))
    total_needed = pre_samples + dur_samples + post_samples

    RECORDINGS_FIXED_DIR.mkdir(parents=True, exist_ok=True)
    fixed_wav = RECORDINGS_FIXED_DIR / f"chunk_{chunk_id:03d}.wav"
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        str(source_audio),
        "-vn",
        "-filter:a",
        (
            f"aresample={FIXED_SAMPLE_RATE},"
            f"apad=whole_len={total_needed},"
            f"atrim=start_sample={pre_samples}:end_sample={pre_samples + dur_samples},"
            f"asetpts=PTS-STARTPTS,"
            f"apad=whole_len={dur_samples},"
            f"atrim=end_sample={dur_samples},"
            f"asetpts=PTS-STARTPTS"
        ),
        "-ar",
        str(FIXED_SAMPLE_RATE),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(fixed_wav),
    ], capture_stderr=True)
    return fixed_wav


def _find_chunk_recording(chunk_id: int) -> Path | None:
    fixed_audio = RECORDINGS_FIXED_DIR / f"chunk_{chunk_id:03d}.wav"
    raw_audio = RECORDINGS_DIR / f"chunk_{chunk_id:03d}.webm"
    if fixed_audio.exists():
        return fixed_audio
    if raw_audio.exists():
        return raw_audio
    return None


def _prepare_fixed_wav_from_audio(source_audio: Path, wav_file: Path, chunk_dur: float) -> None:
    dur_samples = int(round(max(0.1, chunk_dur) * FIXED_SAMPLE_RATE))
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(source_audio),
        "-filter:a",
        (
            f"aresample={FIXED_SAMPLE_RATE},"
            f"apad=whole_len={dur_samples},"
            f"atrim=end_sample={dur_samples},"
            f"asetpts=PTS-STARTPTS"
        ),
        "-ar",
        str(FIXED_SAMPLE_RATE),
        "-ac",
        "2",
        str(wav_file),
    ])


def _extract_original_audio_segment(base_video: Path, start: float, end: float, wav_file: Path) -> None:
    chunk_dur = max(0.1, end - start)
    dur_samples = int(round(chunk_dur * FIXED_SAMPLE_RATE))
    # Prefer original audio from the source clip timeline; fall back to silence.
    try:
        _run_ffmpeg([
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.4f}",
            "-t",
            f"{chunk_dur:.4f}",
            "-i",
            str(base_video),
            "-vn",
            "-filter:a",
            (
                f"aresample={FIXED_SAMPLE_RATE},"
                f"apad=whole_len={dur_samples},"
                f"atrim=end_sample={dur_samples},"
                f"asetpts=PTS-STARTPTS"
            ),
            "-ar",
            str(FIXED_SAMPLE_RATE),
            "-ac",
            "2",
            str(wav_file),
        ], capture_stderr=True)
    except Exception:
        _run_ffmpeg([
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={FIXED_SAMPLE_RATE}:cl=stereo",
            "-t",
            f"{chunk_dur:.4f}",
            "-filter:a",
            (
                f"apad=whole_len={dur_samples},"
                f"atrim=end_sample={dur_samples},"
                f"asetpts=PTS-STARTPTS"
            ),
            "-ar",
            str(FIXED_SAMPLE_RATE),
            "-ac",
            "2",
            str(wav_file),
        ])


def _merge_final_video_split():
    transcript = _read_transcript()

    # Use the uploaded/active video as the base
    base_video = active_video_file
    if not base_video.exists():
        raise FileNotFoundError(f"Missing source video: {base_video}")

    if MERGED_PARTS_DIR.exists():
        shutil.rmtree(MERGED_PARTS_DIR)
    MERGED_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: For each chunk use VO if recorded, otherwise original audio from that range.
    wav_files = []
    for item in transcript:
        chunk_id = int(item["chunk_id"])
        start = float(item["start"])
        end = float(item["end"])
        chunk_dur = max(0.1, end - start)

        wav_file = MERGED_PARTS_DIR / f"chunk_{chunk_id:03d}.wav"
        source_audio = _find_chunk_recording(chunk_id)
        if source_audio is not None:
            _prepare_fixed_wav_from_audio(source_audio, wav_file, chunk_dur)
        else:
            _extract_original_audio_segment(base_video, start, end, wav_file)
        wav_files.append((item, wav_file))

    # Step 2: Concatenate the fixed-length chunks.
    input_args = []
    filter_parts = []
    concat_inputs = []

    for idx, (item, wav_file) in enumerate(wav_files):
        start = float(item["start"])
        end = float(item["end"])
        chunk_dur = max(0.1, end - start)

        input_args.extend(["-i", str(wav_file)])

        dur_samples = int(round(chunk_dur * FIXED_SAMPLE_RATE))
        filter_parts.append(f"[{idx}:a]atrim=end_sample={dur_samples},asetpts=PTS-STARTPTS[seg{idx}]")
        concat_inputs.append(f"[seg{idx}]")

    full_filter = ";".join(filter_parts) + ";" + "".join(concat_inputs) + f"concat=n={len(wav_files)}:v=0:a=1[fullaud]"

    # Use PCM WAV here to avoid AAC encoder delay / priming trimming.
    merged_audio = MERGED_PARTS_DIR / "merged_audio.wav"
    _run_ffmpeg([
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", full_filter,
        "-map", "[fullaud]",
        "-c:a", "pcm_s16le",
        str(merged_audio),
    ])

    # Step 3: Mux merged audio + original video (strip original audio)
    video_dur = _probe_duration_seconds(base_video)
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(base_video),
        "-i", str(merged_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a",
        "192k",
        "-t",
        f"{video_dur:.4f}",
        str(FINAL_OUTPUT),
    ])


def _encode_part_with_vo(video_path: Path, wav_file: Path, part_file: Path) -> None:
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(wav_file),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(part_file),
    ])


def _encode_part_keep_original(video_path: Path, part_file: Path, chunk_dur: float) -> None:
    """Keep original video + original voice; ensure audio exists for concat."""
    try:
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(part_file),
        ], capture_stderr=True)
    except Exception:
        # No usable audio track — attach silence so concat still works.
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-f", "lavfi",
            "-i", f"anullsrc=r={FIXED_SAMPLE_RATE}:cl=stereo",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", f"{chunk_dur:.4f}",
            "-movflags", "+faststart",
            str(part_file),
        ])


def _merge_final_video_chunks():
    """Option 2: VO where recorded, original audio otherwise, then concatenate."""
    transcript = _read_transcript()

    missing_videos = []
    for item in transcript:
        chunk_id = int(item["chunk_id"])
        video_path = _chunk_video_path(chunk_id)
        if not video_path.exists():
            missing_videos.append(chunk_id)
    if missing_videos:
        raise ValueError(f"Missing videos for chunk ids: {missing_videos}")

    if MERGED_PARTS_DIR.exists():
        shutil.rmtree(MERGED_PARTS_DIR)
    MERGED_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    part_files = []
    for item in transcript:
        chunk_id = int(item["chunk_id"])
        video_path = _chunk_video_path(chunk_id)
        start = float(item["start"])
        end = float(item["end"])
        chunk_dur = max(0.1, end - start)
        part_file = MERGED_PARTS_DIR / f"part_{chunk_id:03d}.mp4"

        source_audio = _find_chunk_recording(chunk_id)
        if source_audio is not None:
            wav_file = MERGED_PARTS_DIR / f"chunk_{chunk_id:03d}.wav"
            _prepare_fixed_wav_from_audio(source_audio, wav_file, chunk_dur)
            _encode_part_with_vo(video_path, wav_file, part_file)
        else:
            _encode_part_keep_original(video_path, part_file, chunk_dur)
        part_files.append(part_file)

    # Concat demuxer list (relative paths are more reliable on Windows)
    list_file = MERGED_PARTS_DIR / "concat_list.txt"
    list_file.write_text(
        "".join(f"file '{p.name}'\n" for p in part_files),
        encoding="utf-8",
    )
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(FINAL_OUTPUT),
    ])


def _apply_background_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    volume: float = BG_MUSIC_VOLUME,
) -> None:
    """Mix looping background music under the video's existing audio."""
    if not video_path.exists():
        raise FileNotFoundError(f"Missing merged video: {video_path}")
    if not music_path.exists():
        raise FileNotFoundError(f"Missing background music: {music_path}")

    bg_volume = max(0.01, min(1.0, float(volume)))
    video_dur = _probe_duration_seconds(video_path)
    # Loop music to cover full video length, keep voice dominant, trim to video duration.
    _run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-stream_loop",
        "-1",
        "-i",
        str(music_path),
        "-filter_complex",
        (
            f"[0:a]volume=1.0[voice];"
            f"[1:a]volume={bg_volume:.4f}[bg];"
            f"[voice][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        ),
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{video_dur:.4f}",
        "-movflags",
        "+faststart",
        str(output_path),
    ], capture_stderr=True)


def _merge_final_video():
    mode = _get_session_mode()
    if mode == "chunks":
        _merge_final_video_chunks()
    else:
        _merge_final_video_split()
    if FINAL_WITH_BG.exists():
        FINAL_WITH_BG.unlink()


def _reorder_chunks(order: list[int]) -> list:
    transcript = _read_transcript()
    by_id = {int(item["chunk_id"]): item for item in transcript}
    if sorted(order) != sorted(by_id.keys()):
        raise ValueError("order must contain each existing chunk_id exactly once")

    # Move media to temp names first to avoid overwrite collisions.
    temp_dir = OUTPUT_DIR / "_reorder_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    moved = {}
    for old_id in by_id:
        src_video = _chunk_video_path(old_id)
        raw_audio, fixed_audio = _recording_paths(old_id)
        bundle = {"video": None, "raw": None, "fixed": None, "item": by_id[old_id]}
        if src_video.exists():
            dst = temp_dir / f"video_{old_id:03d}.mp4"
            shutil.move(str(src_video), str(dst))
            bundle["video"] = dst
        if raw_audio.exists():
            dst = temp_dir / f"raw_{old_id:03d}.webm"
            shutil.move(str(raw_audio), str(dst))
            bundle["raw"] = dst
        if fixed_audio.exists():
            dst = temp_dir / f"fixed_{old_id:03d}.wav"
            shutil.move(str(fixed_audio), str(dst))
            bundle["fixed"] = dst
        moved[old_id] = bundle

    new_transcript = []
    for new_id, old_id in enumerate(order, start=1):
        bundle = moved[old_id]
        item = dict(bundle["item"])
        item["chunk_id"] = new_id
        if bundle["video"] is not None:
            shutil.move(str(bundle["video"]), str(_chunk_video_path(new_id)))
        if bundle["raw"] is not None:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(bundle["raw"]), str(_recording_paths(new_id)[0]))
        if bundle["fixed"] is not None:
            RECORDINGS_FIXED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(bundle["fixed"]), str(_recording_paths(new_id)[1]))
        _write_chunk_transcript_files(item)
        new_transcript.append(item)

    shutil.rmtree(temp_dir, ignore_errors=True)
    _write_transcript(new_transcript)
    return new_transcript


def _json_response(handler: SimpleHTTPRequestHandler, payload: dict, status=HTTPStatus.OK):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    content_len = int(handler.headers.get("Content-Length", "0"))
    if content_len <= 0:
        return {}
    raw = handler.rfile.read(content_len)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _handle_transcribe_chunk(handler: SimpleHTTPRequestHandler, query: dict) -> None:
    try:
        chunk_id_values = query.get("chunk_id", [])
        if not chunk_id_values:
            raise ValueError("chunk_id is required")
        chunk_id = int(chunk_id_values[0])

        transcript = _read_transcript()
        match = next((x for x in transcript if int(x["chunk_id"]) == chunk_id), None)
        if not match:
            raise ValueError(
                f"chunk_id {chunk_id} not found in transcript. "
                "Reload your videos, then click the part again."
            )

        video_path = _chunk_video_path(chunk_id)
        if not video_path.exists():
            available = sorted(p.name for p in OUTPUT_DIR.glob("chunk_*.mp4")) if OUTPUT_DIR.exists() else []
            raise FileNotFoundError(
                f"Missing video for part {chunk_id}: {video_path.name}. "
                f"Available: {available or 'none'}. "
                "Please Load Selected Videos again."
            )

        force = (query.get("force", ["0"])[0] == "1")
        if match.get("text") and not force:
            _json_response(handler, {"ok": True, "chunk": match, "cached": True})
            return

        text = transcribe_clip_text(str(video_path))
        match["text"] = text
        duration = probe_duration_seconds(video_path)
        if _get_session_mode() == "chunks":
            match["start"] = 0.0
            match["end"] = round(duration, 3)
        _write_chunk_transcript_files(match)
        _write_transcript(transcript)
        _json_response(handler, {"ok": True, "chunk": match, "cached": False})
    except Exception as exc:  # noqa: BLE001
        _json_response(handler, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


class AppHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Permissions-Policy", "microphone=(self)")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/split-progress":
            _json_response(self, _get_state())
            return
        if parsed.path == "/api/process-progress":
            _json_response(self, _get_process_state())
            return
        if parsed.path == "/api/session":
            mode = _get_session_mode()
            payload = {"ok": True, "mode": mode}
            if TRANSCRIPT_FILE.exists():
                try:
                    payload["transcript"] = _read_transcript()
                except Exception:  # noqa: BLE001
                    payload["transcript"] = []
            else:
                payload["transcript"] = []
            _json_response(self, payload)
            return
        if parsed.path == "/api/transcribe-chunk":
            _handle_transcribe_chunk(self, parse_qs(parsed.query))
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/upload-video":
            query = parse_qs(parsed.query)
            name_values = query.get("name", [])
            file_name = name_values[0] if name_values else "story.mp4"
            ext = Path(file_name).suffix.lower() or ".mp4"
            if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                self.send_error(HTTPStatus.BAD_REQUEST, "Unsupported video extension")
                return

            content_len = int(self.headers.get("Content-Length", "0"))
            if content_len <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "Empty video body")
                return

            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            target_file = UPLOADS_DIR / f"source{ext}"
            body = self.rfile.read(content_len)
            target_file.write_bytes(body)

            global active_video_file
            active_video_file = target_file
            _set_session_mode("split")

            _json_response(self, {"ok": True, "video_file": str(target_file)})
            return

        if parsed.path == "/api/process-video":
            try:
                query = parse_qs(parsed.query)
                chunk_values = query.get("chunk_size", [])
                chunk_size = int(chunk_values[0]) if chunk_values else 6
                if chunk_size <= 0:
                    raise ValueError("chunk_size must be positive")

                if _get_process_state().get("running"):
                    payload = {
                        "ok": False,
                        "started": False,
                        "error": "Pipeline already running. Wait for it to finish.",
                    }
                    _json_response(self, payload, status=HTTPStatus.CONFLICT)
                    return

                job_id = _begin_process_job(
                    stage="queued",
                    message="Starting video split pipeline...",
                )
                worker = threading.Thread(
                    target=_run_process_job,
                    args=(active_video_file, chunk_size),
                    daemon=True,
                )
                worker.start()
                _json_response(self, {"ok": True, "started": True, "job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/load-chunks":
            try:
                if _get_process_state().get("running"):
                    _json_response(
                        self,
                        {
                            "ok": False,
                            "started": False,
                            "error": "Pipeline already running. Wait for it to finish.",
                        },
                        status=HTTPStatus.CONFLICT,
                    )
                    return

                data = _read_json_body(self)
                files_meta = data.get("files") or []
                if not isinstance(files_meta, list) or not files_meta:
                    raise ValueError("files list is required")

                # Frontend uploads files one-by-one into uploads/chunks_staging first,
                # then calls this with the staged filenames in desired order.
                staging = UPLOADS_DIR / "chunks_staging"
                if not staging.exists():
                    raise FileNotFoundError("No staged chunk uploads found. Upload chunks first.")

                files: list[tuple[str, Path]] = []
                for meta in files_meta:
                    name = meta.get("name") if isinstance(meta, dict) else None
                    staged_name = meta.get("staged") if isinstance(meta, dict) else None
                    if not name or not staged_name:
                        raise ValueError("Each file needs name and staged fields")
                    staged_path = (staging / staged_name).resolve()
                    if not staged_path.exists():
                        raise FileNotFoundError(f"Missing staged file: {staged_name}")
                    files.append((name, staged_path))

                job_id = _begin_process_job(
                    stage="queued",
                    message=f"Starting load of {len(files)} video chunks...",
                    total=len(files),
                )
                worker = threading.Thread(target=_run_load_chunks_job, args=(files,), daemon=True)
                worker.start()
                _json_response(self, {"ok": True, "started": True, "count": len(files), "job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/upload-chunk-file":
            try:
                query = parse_qs(parsed.query)
                name_values = query.get("name", [])
                index_values = query.get("index", [])
                file_name = name_values[0] if name_values else "chunk.mp4"
                index = int(index_values[0]) if index_values else 0
                ext = Path(file_name).suffix.lower() or ".mp4"
                if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                    raise ValueError("Unsupported video extension")

                content_len = int(self.headers.get("Content-Length", "0"))
                if content_len <= 0:
                    raise ValueError("Empty video body")

                staging = UPLOADS_DIR / "chunks_staging"
                staging.mkdir(parents=True, exist_ok=True)
                staged_name = f"{index:04d}_{Path(file_name).name}"
                target = staging / staged_name
                target.write_bytes(self.rfile.read(content_len))

                _json_response(self, {"ok": True, "staged": staged_name, "name": Path(file_name).name})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/clear-chunk-staging":
            staging = UPLOADS_DIR / "chunks_staging"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=True)
            _json_response(self, {"ok": True})
            return

        if parsed.path == "/api/transcribe-chunk":
            _handle_transcribe_chunk(self, parse_qs(parsed.query))
            return

        if parsed.path == "/api/reorder-chunks":
            try:
                data = _read_json_body(self)
                order = data.get("order")
                if not isinstance(order, list) or not order:
                    raise ValueError("order must be a non-empty list of chunk_ids")
                order_ids = [int(x) for x in order]
                new_transcript = _reorder_chunks(order_ids)
                _json_response(self, {"ok": True, "transcript": new_transcript})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/upload-recording":
            query = parse_qs(parsed.query)
            chunk_id_values = query.get("chunk_id", [])
            if not chunk_id_values:
                self.send_error(HTTPStatus.BAD_REQUEST, "chunk_id is required")
                return

            try:
                chunk_id = int(chunk_id_values[0])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "chunk_id must be integer")
                return

            content_len = int(self.headers.get("Content-Length", "0"))
            if content_len <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "Empty audio body")
                return

            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            RECORDINGS_FIXED_DIR.mkdir(parents=True, exist_ok=True)
            body = self.rfile.read(content_len)
            target_file = RECORDINGS_DIR / f"chunk_{chunk_id:03d}.webm"
            target_file.write_bytes(body)

            # Normalize to exact transcript duration and save as fixed WAV.
            try:
                transcript = _read_transcript()
                match = next((x for x in transcript if int(x["chunk_id"]) == chunk_id), None)
                if not match:
                    raise ValueError(f"chunk_id {chunk_id} not found in transcript.json")
                chunk_dur = max(0.1, float(match["end"]) - float(match["start"]))
                fixed_wav = _normalize_recording_to_fixed_wav(chunk_id, target_file, chunk_dur)
                fixed_url = f"/{RECORDINGS_FIXED_DIR.as_posix()}/chunk_{chunk_id:03d}.wav"
                normalize_error = None
            except Exception as exc:  # noqa: BLE001
                fixed_url = None
                normalize_error = str(exc)

            # Measure duration of the fixed WAV for debugging/confirmation.
            fixed_duration = None
            if fixed_url:
                try:
                    fixed_duration = _probe_duration_seconds(RECORDINGS_FIXED_DIR / f"chunk_{chunk_id:03d}.wav")
                except Exception:
                    fixed_duration = None

            payload = {
                "ok": True,
                "chunk_id": chunk_id,
                "fixed_url": fixed_url,
                "fixed_duration": fixed_duration,
                "normalize_error": normalize_error,
            }
            _json_response(self, payload)
            return

        if parsed.path == "/api/merge-final":
            try:
                _merge_final_video()
                _json_response(self, {"ok": True, "output_url": f"/{FINAL_OUTPUT.name}"})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/upload-bg-music":
            try:
                query = parse_qs(parsed.query)
                name_values = query.get("name", [])
                file_name = name_values[0] if name_values else "bg_music.mp3"
                ext = Path(file_name).suffix.lower() or ".mp3"
                if ext not in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"}:
                    raise ValueError("Unsupported audio extension")

                content_len = int(self.headers.get("Content-Length", "0"))
                if content_len <= 0:
                    raise ValueError("Empty audio body")

                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                # Clear previous bg track variants
                for old in UPLOADS_DIR.glob("bg_music_track*"):
                    old.unlink(missing_ok=True)

                target = Path(f"{BG_MUSIC_FILE}{ext}")
                target.write_bytes(self.rfile.read(content_len))
                _json_response(self, {
                    "ok": True,
                    "music_file": str(target),
                    "name": Path(file_name).name,
                })
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/prepare-final-download":
            try:
                query = parse_qs(parsed.query)
                use_bg = (query.get("bg", ["0"])[0] == "1")
                if not FINAL_OUTPUT.exists():
                    raise FileNotFoundError("No merged video yet. Click Merge first.")

                if not use_bg:
                    _json_response(self, {
                        "ok": True,
                        "output_url": f"/{FINAL_OUTPUT.name}?t={int(FINAL_OUTPUT.stat().st_mtime)}",
                        "filename": "final_output.mp4",
                        "bg": False,
                    })
                    return

                music_candidates = sorted(UPLOADS_DIR.glob("bg_music_track.*"))
                if not music_candidates:
                    raise FileNotFoundError("BG Music is on, but no music file was selected.")

                level_values = query.get("level", [])
                try:
                    level_pct = float(level_values[0]) if level_values else (BG_MUSIC_VOLUME * 100.0)
                except ValueError as exc:
                    raise ValueError("BG music level must be a number") from exc
                if level_pct < 1 or level_pct > 100:
                    raise ValueError("BG music level must be between 1 and 100")

                music_path = music_candidates[0]
                _apply_background_music(
                    FINAL_OUTPUT,
                    music_path,
                    FINAL_WITH_BG,
                    volume=level_pct / 100.0,
                )
                _json_response(self, {
                    "ok": True,
                    "output_url": f"/{FINAL_WITH_BG.name}?t={int(FINAL_WITH_BG.stat().st_mtime)}",
                    "filename": "final_output_bg.mp4",
                    "bg": True,
                    "level": level_pct,
                })
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path != "/api/split-video":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        try:
            state = _get_state()
            if not state["running"]:
                worker = threading.Thread(target=_run_split_job, daemon=True)
                worker.start()
            payload = {
                "ok": True,
                "started": True,
                "output_dir": str(OUTPUT_DIR),
            }
            _json_response(self, payload)
        except Exception as exc:  # noqa: BLE001
            _json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def run() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), AppHandler)
    print(f"Serving at http://localhost:{PORT}")
    print("Option 1: Select video -> set chunk size -> process -> record -> merge.")
    print("Option 2: Select video chunks -> load -> click to transcribe/record -> reorder -> merge.")
    server.serve_forever()


if __name__ == "__main__":
    run()
