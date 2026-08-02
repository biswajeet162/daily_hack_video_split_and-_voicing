import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from faster_whisper import WhisperModel


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@lru_cache(maxsize=2)
def _get_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        if device != "cpu":
            print("CUDA init failed; falling back to CPU int8.")
            return WhisperModel(model_name, device="cpu", compute_type="int8")
        raise


def transcribe_clip_text(
    video_path: str,
    *,
    language: str = "hi",
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> str:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    model = _get_model(model_name, device, compute_type)
    segments, _info = model.transcribe(
        str(video_file),
        language=language,
        beam_size=beam_size,
        word_timestamps=False,
        vad_filter=vad_filter,
    )
    texts: List[str] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            texts.append(t)
    return " ".join(texts).strip()


def transcribe_video(video_path: str, chunk_seconds: int) -> List[Dict]:
    """
    Backward-compatible API (used by older server code).
    For new pipeline we transcribe per clip using transcribe_clip_text().
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    # Simple fallback: single clip transcription into 1 bucket.
    text = transcribe_clip_text(str(video_file))
    return [{"chunk_id": 1, "start": 0, "end": chunk_seconds, "text": text}]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe a single clip to text.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="clip_transcript.txt")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute", default="float16")
    parser.add_argument("--beam", type=int, default=5)
    args = parser.parse_args(argv)

    text = transcribe_clip_text(
        args.video,
        language=args.lang,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute,
        beam_size=args.beam,
    )
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"Wrote transcript to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
