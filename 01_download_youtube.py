"""
[1] Download YouTube videos (high quality + audio) into input_videos/.

Preferred: double-click run.bat and choose 1 (uses links.txt automatically).

Manual (conda env utube_env):
  conda run -n utube_env python 01_download_youtube.py --file links.txt
  conda run -n utube_env python 01_download_youtube.py "https://youtu.be/XXXX"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "input_videos"

# Prefer best MP4 video + best M4A audio, merge to MP4.
# Falls back to best single file if separate streams aren't available.
FORMAT = (
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo+bestaudio/"
    "best[ext=mp4]/"
    "best"
)

URL_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)[\w\-?=&%.]+",
    re.IGNORECASE,
)
VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)([\w-]{6,})",
    re.IGNORECASE,
)
DOWNLOAD_ARCHIVE_PATH = OUTPUT_DIR / ".download_archive.txt"


def extract_urls(text: str) -> list[str]:
    """Pull YouTube URLs from free-form text (one per line or mixed)."""
    found = URL_RE.findall(text)
    # Also accept bare lines that look like youtu URLs without http
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "youtu" in line.lower() and line not in found:
            if not line.startswith("http"):
                line = "https://" + line
            if URL_RE.match(line) or "youtube.com" in line or "youtu.be" in line:
                found.append(line)
    # Preserve order, drop duplicates
    seen: set[str] = set()
    urls: list[str] = []
    for u in found:
        if u not in seen:
            seen.add(u)
            urls.append(u)
    return urls


def read_links_file(path: Path) -> list[str]:
    return extract_urls(path.read_text(encoding="utf-8"))


def extract_video_id(url: str) -> str | None:
    match = VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def load_download_archive() -> set[str]:
    if not DOWNLOAD_ARCHIVE_PATH.exists():
        return set()

    ids: set[str] = set()
    for line in DOWNLOAD_ARCHIVE_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            ids.add(parts[1])
    return ids


def prompt_links() -> list[str]:
    print("Paste YouTube links (one per line). Empty line when done:\n")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return extract_urls("\n".join(lines))


def build_ydl_opts() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "format": FORMAT,
        "outtmpl": str(OUTPUT_DIR / "%(title)s [%(id)s].%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "ignoreerrors": False,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "writethumbnail": False,
        "writesubtitles": False,
        "quiet": False,
        "noprogress": False,
        # Avoid re-downloading the same video id
        "download_archive": str(DOWNLOAD_ARCHIVE_PATH),
    }


def download(urls: list[str]) -> int:
    if not urls:
        print("No YouTube links provided.")
        return 1

    print(f"Saving to: {OUTPUT_DIR}")
    print(f"Links in file: {len(urls)}")

    archive = load_download_archive()
    pending: list[str] = []
    skipped = 0
    for url in urls:
        video_id = extract_video_id(url)
        if video_id and video_id in archive:
            skipped += 1
            print(f"  Skip (already downloaded): {url}")
        else:
            pending.append(url)

    if skipped:
        print(f"Skipped {skipped} already downloaded link(s)")
    print(f"Videos to download: {len(pending)}\n")

    if not pending:
        print("Nothing new to download.")
        return 0

    opts = build_ydl_opts()
    failed = 0
    downloaded = 0

    with yt_dlp.YoutubeDL(opts) as ydl:
        for i, url in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {url}")
            try:
                ydl.download([url])
                downloaded += 1
            except Exception as exc:
                failed += 1
                print(f"  FAILED: {exc}", file=sys.stderr)
            print()

    print(
        f"Done. Downloaded {downloaded}, skipped {skipped}, failed {failed}."
    )
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download high-quality YouTube videos (with audio) into input_videos/"
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="One or more YouTube video URLs",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Text file with YouTube links (one per line; # comments allowed)",
    )
    args = parser.parse_args()

    urls: list[str] = []
    if args.file:
        if not args.file.exists():
            print(f"File not found: {args.file}", file=sys.stderr)
            return 1
        urls.extend(read_links_file(args.file))
    if args.urls:
        urls.extend(extract_urls("\n".join(args.urls)))
    if not urls:
        urls = prompt_links()

    # Dedupe again after combining sources
    seen: set[str] = set()
    unique: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    return download(unique)


if __name__ == "__main__":
    raise SystemExit(main())
