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

import log_utils as log

log.configure_stdio()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "input_videos"
LINKS_FILE = ROOT / "links.txt"

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
        "download_archive": str(DOWNLOAD_ARCHIVE_PATH),
        "progress_hooks": [_download_progress_hook],
    }


def _download_progress_hook(status: dict) -> None:
    if status.get("status") == "downloading":
        pct = (status.get("_percent_str") or "?").strip()
        speed = (status.get("_speed_str") or "?").strip()
        eta = (status.get("_eta_str") or "?").strip()
        log.progress(f"Downloading {pct}  |  speed {speed}  |  ETA {eta}")
    elif status.get("status") == "finished":
        filename = status.get("filename") or status.get("info_dict", {}).get("_filename", "")
        if filename:
            log.ok(f"Download finished -> {Path(filename).name}")


def download(urls: list[str], links_file: Path | None = None) -> int:
    if not urls:
        log.fail("No YouTube links provided.")
        return 1

    log.banner(
        "[1] DOWNLOAD YOUTUBE VIDEOS",
        "High-quality MP4 with audio into input_videos/",
    )
    log.paths_block(
        "Folders and trackers",
        [
            ("Project root", ROOT),
            ("Links file", links_file or "(from command line / prompt)"),
            ("Output folder", OUTPUT_DIR),
            ("Skip tracker", DOWNLOAD_ARCHIVE_PATH),
        ],
    )

    archive = load_download_archive()
    pending: list[str] = []
    skipped = 0

    log.section(f"Checking {len(urls)} link(s) against download archive")
    for url in urls:
        video_id = extract_video_id(url)
        if video_id and video_id in archive:
            skipped += 1
            log.bullet(f"SKIP (already downloaded)  {url}")
        else:
            pending.append(url)
            log.bullet(f"PENDING  {url}")

    log.summary(
        "Queue summary",
        [
            ("Total links", str(len(urls))),
            ("Already downloaded", str(skipped)),
            ("To download now", str(len(pending))),
        ],
    )

    if not pending:
        log.ok("Nothing new to download. All links are already in input_videos/.")
        log.done_block(
            "Download task",
            [
                ("Downloaded", 0),
                ("Skipped", skipped),
                ("Failed", 0),
                ("Output folder", OUTPUT_DIR),
            ],
        )
        return 0

    opts = build_ydl_opts()
    failed = 0
    downloaded = 0
    saved_files: list[str] = []

    with yt_dlp.YoutubeDL(opts) as ydl:
        for i, url in enumerate(pending, 1):
            log.step(i, len(pending), f"Downloading video")
            log.info("URL", url)
            try:
                info = ydl.extract_info(url, download=False)
                title = info.get("title", "unknown title") if info else "unknown title"
                video_id = info.get("id", "?") if info else "?"
                log.info("Title", title)
                log.info("Video ID", video_id)
                log.info("Target folder", OUTPUT_DIR)

                ydl.download([url])
                downloaded += 1

                matches = sorted(OUTPUT_DIR.glob(f"*[{video_id}].mp4"))
                if matches:
                    out_file = matches[-1]
                    saved_files.append(out_file.name)
                    log.ok(f"Saved -> {out_file}")
                else:
                    log.ok("Saved into input_videos/ (filename from YouTube title)")
            except Exception as exc:
                failed += 1
                log.fail(f"Download failed: {exc}")

    log.done_block(
        "Download task",
        [
            ("Downloaded", downloaded),
            ("Skipped", skipped),
            ("Failed", failed),
            ("Output folder", OUTPUT_DIR),
            ("Tracker file", DOWNLOAD_ARCHIVE_PATH),
        ],
        success=failed == 0,
    )
    if saved_files:
        log.section("Files saved this run")
        for name in saved_files:
            log.bullet(name)

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

    return download(unique, links_file=args.file)


if __name__ == "__main__":
    raise SystemExit(main())
