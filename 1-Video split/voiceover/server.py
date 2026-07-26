"""Voice-over web server for split clips (Video Split step 6)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import (  # noqa: E402
    OUTPUT_DIR,
    TRANSCRIPTIONS_DIR,
    VOICEOVER_DIR,
    VOICEOVER_FIXED_DIR,
    VOICEOVER_OUTPUT_DIR,
    VOICEOVER_PROCESSED,
    VOICEOVER_RECORDINGS_DIR,
)

PORT = 8001
PRE_BUFFER_SEC = 0.15
POST_BUFFER_SEC = 0.14
FIXED_SAMPLE_RATE = 48000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_tracker() -> dict:
    if VOICEOVER_PROCESSED.exists():
        return json.loads(VOICEOVER_PROCESSED.read_text(encoding="utf-8"))
    return {"clips": {}}


def _save_tracker(data: dict) -> None:
    VOICEOVER_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    VOICEOVER_PROCESSED.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_ffmpeg(cmd: list[str], *, capture_stderr: bool = False) -> str:
    if capture_stderr:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "").strip() or f"ffmpeg failed (code={proc.returncode})")
        return (proc.stderr or "").strip()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ""


def _probe_duration_seconds(media_path: Path) -> float:
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
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:
        raise ValueError(f"Unable to read duration for: {media_path}")
    return duration


def _clip_key_to_id(clip_key: str) -> str:
    return quote(clip_key.replace("\\", "/"), safe="")


def _clip_id_to_key(clip_id: str) -> str:
    return unquote(clip_id).replace("\\", "/")


def _safe_clip_paths(clip_key: str) -> tuple[Path, Path]:
    normalized = clip_key.replace("\\", "/").lstrip("/")
    video_path = (OUTPUT_DIR / normalized).resolve()
    output_root = OUTPUT_DIR.resolve()
    if not str(video_path).startswith(str(output_root)):
        raise ValueError("Invalid clip path")
    if not video_path.exists() or video_path.suffix.lower() != ".mp4":
        raise FileNotFoundError(f"Clip video not found: {clip_key}")
    return video_path, output_root


def _transcription_path_for_clip(clip_key: str) -> Path:
    rel = Path(clip_key)
    return TRANSCRIPTIONS_DIR / rel.parent / f"{rel.stem}.transcription.json"


def _load_transcription(clip_key: str) -> dict:
    path = _transcription_path_for_clip(clip_key)
    if not path.exists():
        raise FileNotFoundError(f"Missing transcription JSON for: {clip_key}")
    return json.loads(path.read_text(encoding="utf-8"))


def _recording_paths(clip_key: str) -> tuple[Path, Path]:
    clip_id = _clip_key_to_id(clip_key)
    raw = VOICEOVER_RECORDINGS_DIR / f"{clip_id}.webm"
    fixed = VOICEOVER_FIXED_DIR / f"{clip_id}.wav"
    return raw, fixed


def _output_video_path(clip_key: str) -> Path:
    return VOICEOVER_OUTPUT_DIR / Path(clip_key.replace("\\", "/"))


def scan_clips() -> list[dict]:
    tracker = _load_tracker()
    done_map = tracker.get("clips", {})
    clips: list[dict] = []

    if not TRANSCRIPTIONS_DIR.exists():
        return clips

    for json_path in sorted(TRANSCRIPTIONS_DIR.rglob("*.transcription.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        source_text = data.get("source") or ""
        if not source_text:
            continue

        video_path = Path(source_text)
        if not video_path.exists():
            continue

        try:
            clip_key = video_path.relative_to(OUTPUT_DIR).as_posix()
        except ValueError:
            continue

        categories = data.get("categories") or []
        display_label = (data.get("display_label") or "").strip()
        if not categories or not display_label:
            continue

        duration = float(data.get("duration_sec") or _probe_duration_seconds(video_path))
        done_info = done_map.get(clip_key, {})
        clip_id = _clip_key_to_id(clip_key)
        _, fixed_wav = _recording_paths(clip_key)
        has_recording = fixed_wav.exists() or (VOICEOVER_RECORDINGS_DIR / f"{clip_id}.webm").exists()

        clips.append(
            {
                "clip_id": clip_id,
                "clip_key": clip_key,
                "video_name": video_path.parent.name,
                "part_name": video_path.name,
                "text": data.get("text") or "",
                "display_label": display_label,
                "categories": categories,
                "category_names_en": data.get("category_names_en") or [],
                "category_names_hi": data.get("category_names_hi") or [],
                "duration_sec": round(duration, 2),
                "done": bool(done_info.get("output_video")),
                "has_recording": has_recording,
                "output_video": done_info.get("output_video"),
            }
        )

    clips.sort(key=lambda item: (item["video_name"].lower(), item["part_name"].lower()))
    return clips


def _normalize_recording(clip_key: str, raw_webm: Path) -> Path:
    data = _load_transcription(clip_key)
    chunk_dur = max(0.1, float(data.get("duration_sec") or 0.1))
    pre_samples = int(round(PRE_BUFFER_SEC * FIXED_SAMPLE_RATE))
    post_samples = int(round(POST_BUFFER_SEC * FIXED_SAMPLE_RATE))
    dur_samples = int(round(chunk_dur * FIXED_SAMPLE_RATE))
    total_needed = pre_samples + dur_samples + post_samples

    VOICEOVER_FIXED_DIR.mkdir(parents=True, exist_ok=True)
    clip_id = _clip_key_to_id(clip_key)
    fixed_wav = VOICEOVER_FIXED_DIR / f"{clip_id}.wav"

    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(raw_webm),
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
        ],
        capture_stderr=True,
    )
    return fixed_wav


def _mux_voiceover_clip(clip_key: str) -> Path:
    video_path, _ = _safe_clip_paths(clip_key)
    raw_webm, fixed_wav = _recording_paths(clip_key)
    audio_path = fixed_wav if fixed_wav.exists() else raw_webm
    if not audio_path.exists():
        raise FileNotFoundError("Record this clip before marking it done.")

    if not fixed_wav.exists() and raw_webm.exists():
        audio_path = _normalize_recording(clip_key, raw_webm)

    output_path = _output_video_path(clip_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_dur = _probe_duration_seconds(video_path)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{video_dur:.4f}",
            "-shortest",
            str(output_path),
        ]
    )

    sidecar = output_path.with_suffix(".voiceover.json")
    transcript = _load_transcription(clip_key)
    sidecar.write_text(
        json.dumps(
            {
                "clip_key": clip_key,
                "source_video": str(video_path),
                "output_video": str(output_path),
                "text": transcript.get("text") or "",
                "display_label": transcript.get("display_label") or "",
                "categories": transcript.get("categories") or [],
                "duration_sec": transcript.get("duration_sec"),
                "voiceover_audio": str(audio_path),
                "completed_at": _utc_now(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tracker = _load_tracker()
    tracker.setdefault("clips", {})[clip_key] = {
        "completed_at": _utc_now(),
        "output_video": str(output_path),
        "sidecar_json": str(sidecar),
        "voiceover_audio": str(audio_path),
    }
    _save_tracker(tracker)
    return output_path


class VoiceoverHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Permissions-Policy", "microphone=(self)")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/clips":
            clips = scan_clips()
            done_count = sum(1 for clip in clips if clip["done"])
            self._send_json({"ok": True, "clips": clips, "done_count": done_count, "total": len(clips)})
            return

        if parsed.path == "/api/video":
            query = parse_qs(parsed.query)
            clip_values = query.get("clip_id", [])
            if not clip_values:
                self.send_error(HTTPStatus.BAD_REQUEST, "clip_id is required")
                return
            try:
                clip_key = _clip_id_to_key(clip_values[0])
                video_path, _ = _safe_clip_paths(clip_key)
            except (ValueError, FileNotFoundError) as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                return

            try:
                data = video_path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Unable to read clip video")
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/output-video":
            query = parse_qs(parsed.query)
            clip_values = query.get("clip_id", [])
            if not clip_values:
                self.send_error(HTTPStatus.BAD_REQUEST, "clip_id is required")
                return
            try:
                clip_key = _clip_id_to_key(clip_values[0])
                output_path = _output_video_path(clip_key)
                if not output_path.exists():
                    raise FileNotFoundError("Voice-over output not found. Mark clip as Done first.")
                output_root = VOICEOVER_OUTPUT_DIR.resolve()
                resolved = output_path.resolve()
                if not str(resolved).startswith(str(output_root)):
                    raise ValueError("Invalid output path")
            except (ValueError, FileNotFoundError) as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                return

            data = resolved.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path.startswith("/recordings_fixed/") or parsed.path.startswith("/recordings/"):
            rel = parsed.path.lstrip("/")
            file_path = (VOICEOVER_DIR / rel).resolve()
            voiceover_root = VOICEOVER_DIR.resolve()
            if not str(file_path).startswith(str(voiceover_root)) or not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Recording not found")
                return
            if file_path.suffix.lower() == ".wav":
                content_type = "audio/wav"
            else:
                content_type = "audio/webm"
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        clip_values = query.get("clip_id", [])
        if not clip_values and parsed.path != "/api/clips":
            self._send_json({"ok": False, "error": "clip_id is required"}, HTTPStatus.BAD_REQUEST)
            return

        clip_id = clip_values[0] if clip_values else ""
        clip_key = _clip_id_to_key(clip_id) if clip_id else ""

        if parsed.path == "/api/upload-recording":
            content_len = int(self.headers.get("Content-Length", "0"))
            if content_len <= 0:
                self._send_json({"ok": False, "error": "Empty audio body"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                _safe_clip_paths(clip_key)
            except (ValueError, FileNotFoundError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            VOICEOVER_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            raw_path = VOICEOVER_RECORDINGS_DIR / f"{clip_id}.webm"
            raw_path.write_bytes(self.rfile.read(content_len))

            fixed_url = None
            fixed_duration = None
            normalize_error = None
            try:
                fixed_wav = _normalize_recording(clip_key, raw_path)
                fixed_url = f"/recordings_fixed/{clip_id}.wav"
                fixed_duration = _probe_duration_seconds(fixed_wav)
            except Exception as exc:  # noqa: BLE001
                normalize_error = str(exc)

            self._send_json(
                {
                    "ok": True,
                    "clip_id": clip_id,
                    "fixed_url": fixed_url,
                    "fixed_duration": fixed_duration,
                    "normalize_error": normalize_error,
                }
            )
            return

        if parsed.path == "/api/finish-clip":
            try:
                output_path = _mux_voiceover_clip(clip_key)
                rel_output = output_path.relative_to(VOICEOVER_OUTPUT_DIR.resolve()).as_posix()
                self._send_json(
                    {
                        "ok": True,
                        "clip_id": clip_id,
                        "output_video": rel_output,
                        "output_url": f"/api/output-video?clip_id={clip_id}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/clear-recording":
            raw_path, fixed_path = _recording_paths(clip_key)
            for path in (raw_path, fixed_path):
                if path.exists():
                    path.unlink()
            tracker = _load_tracker()
            if clip_key in tracker.get("clips", {}):
                del tracker["clips"][clip_key]
                _save_tracker(tracker)
            output_path = _output_video_path(clip_key)
            sidecar = output_path.with_suffix(".voiceover.json")
            for path in (output_path, sidecar):
                if path.exists():
                    path.unlink()
            self._send_json({"ok": True, "clip_id": clip_id})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def run() -> None:
    VOICEOVER_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    VOICEOVER_FIXED_DIR.mkdir(parents=True, exist_ok=True)
    VOICEOVER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), VoiceoverHandler)
    print(f"Voice-over server running at http://localhost:{PORT}")
    print(f"Clips source : {OUTPUT_DIR}")
    print(f"Voice output : {VOICEOVER_OUTPUT_DIR}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    run()
