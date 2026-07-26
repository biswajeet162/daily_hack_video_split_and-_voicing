"""
[5] Categorize Hindi transcriptions using local Ollama.

Reads *.transcription.json under output_videos/Transcriptions/, sends the Hindi
text to Ollama with the 100-category list from video_categories/categories.json,
and writes 1-5 category slugs back into the same JSON file.

Tracks completed categorizations in output_videos/.categorize_processed.json
so reruns skip files already categorized (use --force to redo).

Default model: llama3.1:8b (best Hindi + JSON among locally installed models).

Manual:
  conda run -n utube_env python 05_categorize_transcriptions.py
  conda run -n utube_env python 05_categorize_transcriptions.py --force
  conda run -n utube_env python 05_categorize_transcriptions.py --file "output_videos/Transcriptions/...json"
  conda run -n utube_env python 05_categorize_transcriptions.py --model deepseek-r1:14b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import log_utils as log

log.configure_stdio()

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output_videos"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "Transcriptions"
CATEGORIES_PATH = ROOT / "video_categories" / "categories.json"
PROCESSED_TRACKER_PATH = OUTPUT_DIR / ".categorize_processed.json"
FAILED_TRACKER_PATH = OUTPUT_DIR / ".categorize_failed.json"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
MIN_CATEGORIES = 1
MAX_CATEGORIES = 5
MAX_DISPLAY_LABEL_WORDS = 2
FALLBACK_CATEGORY = "life_hacks_general"


def load_categories() -> tuple[dict, list[dict]]:
    if not CATEGORIES_PATH.exists():
        raise FileNotFoundError(f"Category list not found: {CATEGORIES_PATH}")

    data = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    categories = data.get("categories") or []
    if not categories:
        raise ValueError(f"No categories defined in {CATEGORIES_PATH}")

    by_slug = {item["slug"]: item for item in categories if item.get("slug")}
    return by_slug, categories


def load_tracker(path: Path) -> dict:
    if not path.exists():
        return {"files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"files": {}}
    if not isinstance(data, dict):
        return {"files": {}}
    data.setdefault("files", {})
    return data


def save_tracker(path: Path, tracker: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if path == FAILED_TRACKER_PATH and not tracker.get("files"):
        if path.exists():
            path.unlink(missing_ok=True)
        return
    path.write_text(json.dumps(tracker, indent=2, ensure_ascii=False), encoding="utf-8")


def transcript_key(path: Path) -> str:
    try:
        return path.relative_to(TRANSCRIPTIONS_DIR).as_posix()
    except ValueError:
        return path.name


def list_transcription_files() -> list[Path]:
    if not TRANSCRIPTIONS_DIR.exists():
        return []
    return sorted(TRANSCRIPTIONS_DIR.rglob("*.transcription.json"))


def is_already_categorized(payload: dict) -> bool:
    categories = payload.get("categories")
    label = (payload.get("display_label") or "").strip()
    return (
        isinstance(categories, list)
        and len(categories) > 0
        and bool(label)
    )


def load_transcription_payload(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def is_file_processed(path: Path, tracker: dict) -> bool:
    payload = load_transcription_payload(path)
    return payload is not None and is_already_categorized(payload)


def sync_tracker_from_transcriptions(tracker: dict, files: list[Path]) -> dict:
    """Backfill tracker entries from JSON files that already have categories."""
    files_dict = tracker.setdefault("files", {})
    changed = False

    for path in files:
        key = transcript_key(path)
        if key in files_dict:
            continue

        payload = load_transcription_payload(path)
        if payload is None or not is_already_categorized(payload):
            continue

        files_dict[key] = {
            "processed_at": payload.get("categorized_at", "synced_from_output"),
            "file": str(path),
            "categories": payload.get("categories") or [],
            "display_label": payload.get("display_label") or "",
            "synced_from_output": True,
        }
        changed = True

    if changed:
        save_tracker(PROCESSED_TRACKER_PATH, tracker)
        log.ok(f"Synced categorization history to {PROCESSED_TRACKER_PATH.name}")

    return tracker


def build_category_prompt_lines(categories: list[dict]) -> str:
    lines: list[str] = []
    for item in categories:
        slug = item.get("slug", "")
        name_en = item.get("name_en", "")
        name_hi = item.get("name_hi", "")
        lines.append(f"- {slug} | {name_en} | {name_hi}")
    return "\n".join(lines)


def check_ollama(ollama_url: str, model: str) -> None:
    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {ollama_url}. Start Ollama and try again."
        ) from exc

    available = {item.get("name") for item in payload.get("models", [])}
    if model not in available:
        names = ", ".join(sorted(n for n in available if n))
        raise RuntimeError(
            f"Ollama model '{model}' is not installed. Available: {names}"
        )


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("Empty response from Ollama")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Could not parse JSON from Ollama response: {text[:300]}")


def call_ollama(
    *,
    ollama_url: str,
    model: str,
    hindi_text: str,
    category_lines: str,
    timeout_sec: int,
) -> dict:
    system_prompt = (
        "You classify short Hindi life-hack / DIY video transcripts into categories. "
        "Return ONLY valid JSON with this exact shape:\n"
        '{"categories": ["slug_one", "slug_two"], "display_label": "short label"}\n'
        f"Pick between {MIN_CATEGORIES} and {MAX_CATEGORIES} category slugs from the allowed list. "
        "Use only slugs from the list. Prefer the most specific matches "
        "(example: fruit_hacks for fruit peeling, cement_concrete_diy for cement, "
        "plant_propagation for growing plants, clothes_hacks for fabric/clothing).\n"
        f"display_label must be 1-{MAX_DISPLAY_LABEL_WORDS} Hindi words that describe what "
        "happens in this clip (example: 'लहसुन', 'मूंग अंकुर', 'सीमेंट'). No punctuation."
    )
    user_prompt = (
        "Hindi transcript:\n"
        f"{hindi_text.strip()}\n\n"
        "Allowed categories (slug | English | Hindi):\n"
        f"{category_lines}\n\n"
        "Return JSON only."
    )

    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1},
    }

    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    message = payload.get("message") or {}
    content = message.get("content") or ""
    return extract_json_object(content)


def normalize_categories(
    raw_categories: object,
    categories_by_slug: dict[str, dict],
) -> list[str]:
    if not isinstance(raw_categories, list):
        return []

    picked: list[str] = []
    for item in raw_categories:
        slug = str(item).strip()
        if slug in categories_by_slug and slug not in picked:
            picked.append(slug)
        if len(picked) >= MAX_CATEGORIES:
            break

    if not picked and FALLBACK_CATEGORY in categories_by_slug:
        picked = [FALLBACK_CATEGORY]

    return picked[:MAX_CATEGORIES]


def fallback_display_label(hindi_text: str) -> str:
    words = re.findall(r"[\u0900-\u097F]+", hindi_text or "")
    if not words:
        words = (hindi_text or "").split()
    cleaned = [word.strip() for word in words if word.strip()]
    if not cleaned:
        return "हैक"
    return " ".join(cleaned[:MAX_DISPLAY_LABEL_WORDS])


def normalize_display_label(raw_label: object, hindi_text: str) -> str:
    label = re.sub(r"\s+", " ", str(raw_label or "").strip())
    label = label.strip(" .,:;!?\"'")
    words = label.split()
    if words:
        return " ".join(words[:MAX_DISPLAY_LABEL_WORDS])
    return fallback_display_label(hindi_text)


def categorize_text(
    hindi_text: str,
    categories_by_slug: dict[str, dict],
    category_list: list[dict],
    *,
    ollama_url: str,
    model: str,
    timeout_sec: int,
) -> tuple[list[str], str]:
    if not hindi_text.strip():
        raise ValueError("Transcription text is empty")

    category_lines = build_category_prompt_lines(category_list)
    result = call_ollama(
        ollama_url=ollama_url,
        model=model,
        hindi_text=hindi_text,
        category_lines=category_lines,
        timeout_sec=timeout_sec,
    )
    slugs = normalize_categories(result.get("categories"), categories_by_slug)
    display_label = normalize_display_label(result.get("display_label"), hindi_text)
    return slugs, display_label


def apply_categories_to_payload(
    payload: dict,
    slugs: list[str],
    categories_by_slug: dict[str, dict],
    *,
    model: str,
    ollama_url: str,
    display_label: str,
) -> dict:
    payload["categories"] = slugs
    payload["display_label"] = display_label
    payload["category_names_en"] = [
        categories_by_slug[slug]["name_en"] for slug in slugs
    ]
    payload["category_names_hi"] = [
        categories_by_slug[slug]["name_hi"] for slug in slugs
    ]
    payload["categorized_at"] = datetime.now(timezone.utc).isoformat()
    payload["categorizer"] = {
        "provider": "ollama",
        "model": model,
        "ollama_url": ollama_url,
        "min_categories": MIN_CATEGORIES,
        "max_categories": MAX_CATEGORIES,
        "category_list": str(CATEGORIES_PATH),
    }
    return payload


def mark_processed(
    tracker: dict,
    key: str,
    path: Path,
    slugs: list[str],
    display_label: str,
) -> None:
    tracker.setdefault("files", {})[key] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "file": str(path),
        "categories": slugs,
        "display_label": display_label,
    }
    save_tracker(PROCESSED_TRACKER_PATH, tracker)


def mark_failed(tracker: dict, key: str, path: Path, error: str) -> None:
    tracker.setdefault("files", {})[key] = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "file": str(path),
        "error": error,
    }
    save_tracker(FAILED_TRACKER_PATH, tracker)


def clear_failed(failed_tracker: dict, key: str) -> None:
    files = failed_tracker.setdefault("files", {})
    if key in files:
        del files[key]
        save_tracker(FAILED_TRACKER_PATH, failed_tracker)


def resolve_file_arg(path_arg: Path) -> Path:
    path = path_arg
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Transcription file not found: {path_arg}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Not a JSON file: {path_arg}")
    return path


def pick_files(
    file_arg: Path | None,
    *,
    skip_processed: bool,
    processed_tracker: dict,
    failed_tracker: dict,
) -> list[Path]:
    if file_arg is not None:
        path = resolve_file_arg(file_arg)
        if skip_processed and is_file_processed(path, processed_tracker):
            log.bullet(f"SKIP (already categorized)  {transcript_key(path)}")
            return []
        return [path]

    files = list_transcription_files()
    if not skip_processed:
        return files

    pending: list[Path] = []
    skipped = 0
    for path in files:
        if is_file_processed(path, processed_tracker):
            skipped += 1
            continue
        pending.append(path)

    if skipped:
        log.progress(f"Skipped {skipped} already categorized file(s)")

    return pending


def print_status(
    all_files: list[Path],
    processed_tracker: dict,
    failed_tracker: dict,
) -> None:
    log.section("Transcription categorization status")
    log.info("Transcriptions folder", TRANSCRIPTIONS_DIR)
    log.info("Category list", CATEGORIES_PATH)
    log.info("Processed tracker", PROCESSED_TRACKER_PATH)
    log.info("Failed tracker", FAILED_TRACKER_PATH)

    for path in all_files:
        key = transcript_key(path)
        payload = load_transcription_payload(path)

        if payload is None:
            log.bullet(f"{key}  ->  INVALID JSON")
            continue

        if is_file_processed(path, processed_tracker):
            cats = ", ".join(payload.get("categories") or [])
            label = payload.get("display_label") or "-"
            log.bullet(f"{key}  ->  PROCESSED (skip)  [{cats}]  label={label}")
        elif key in failed_tracker.get("files", {}):
            err = failed_tracker["files"][key].get("error", "unknown error")
            log.bullet(f"{key}  ->  PENDING (last attempt failed: {err[:80]})")
        else:
            log.bullet(f"{key}  ->  PENDING (not categorized yet)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Categorize Hindi transcription JSON files using local Ollama"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Specific transcription JSON under output_videos/Transcriptions/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-categorize even if categories already exist",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Ollama request timeout in seconds (default: 180)",
    )
    args = parser.parse_args()

    log.banner(
        "[5] CATEGORIZE TRANSCRIPTIONS WITH OLLAMA",
        f"Model: {args.model}  |  Hindi text -> 1-{MAX_CATEGORIES} categories",
    )

    try:
        categories_by_slug, category_list = load_categories()
    except Exception as exc:
        log.fail(str(exc))
        return 1

    log.paths_block(
        "Folders and trackers",
        [
            ("Project root", ROOT),
            ("Transcriptions", TRANSCRIPTIONS_DIR),
            ("Category list", CATEGORIES_PATH),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
            ("Failed tracker", FAILED_TRACKER_PATH),
            ("Ollama URL", args.ollama_url),
            ("Ollama model", args.model),
        ],
    )
    log.info("Category count", str(len(category_list)))

    try:
        check_ollama(args.ollama_url, args.model)
        log.ok(f"Ollama is running and model '{args.model}' is available")
    except Exception as exc:
        log.fail(str(exc))
        return 1

    processed_tracker = load_tracker(PROCESSED_TRACKER_PATH)
    failed_tracker = load_tracker(FAILED_TRACKER_PATH)
    all_files = list_transcription_files()
    skip_processed = not args.force

    if skip_processed:
        processed_tracker = sync_tracker_from_transcriptions(processed_tracker, all_files)

    if all_files:
        print_status(all_files, processed_tracker, failed_tracker)

    try:
        files = pick_files(
            args.file,
            skip_processed=skip_processed,
            processed_tracker=processed_tracker,
            failed_tracker=failed_tracker,
        )
    except Exception as exc:
        log.fail(str(exc))
        return 1

    if not files:
        log.ok("No transcription JSON files need categorization.")
        log.info("Tracking file", PROCESSED_TRACKER_PATH)
        log.info("Tip", "Use --force to categorize again.")
        return 0

    log.summary(
        "Queue summary",
        [
            ("Files to categorize", str(len(files))),
            ("Model", args.model),
            ("Min categories", str(MIN_CATEGORIES)),
            ("Max categories", str(MAX_CATEGORIES)),
        ],
    )
    for i, path in enumerate(files, 1):
        log.bullet(f"[{i}] {transcript_key(path)}")

    processed_count = 0
    failed_count = 0

    for i, path in enumerate(files, 1):
        key = transcript_key(path)
        log.step(i, len(files), f"Categorizing: {key}")
        log.info("JSON file", path)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            hindi_text = (payload.get("text") or "").strip()
            if not hindi_text and payload.get("segments"):
                hindi_text = " ".join(
                    (seg.get("text") or "").strip()
                    for seg in payload["segments"]
                    if isinstance(seg, dict)
                ).strip()

            log.info("Hindi text", hindi_text[:120] + ("..." if len(hindi_text) > 120 else ""))
            slugs, display_label = categorize_text(
                hindi_text,
                categories_by_slug,
                category_list,
                ollama_url=args.ollama_url,
                model=args.model,
                timeout_sec=args.timeout,
            )
            payload = apply_categories_to_payload(
                payload,
                slugs,
                categories_by_slug,
                model=args.model,
                ollama_url=args.ollama_url,
                display_label=display_label,
            )
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            mark_processed(processed_tracker, key, path, slugs, display_label)
            clear_failed(failed_tracker, key)

            names = ", ".join(payload.get("category_names_en") or slugs)
            log.ok(f"Categories: {names} | Label: {display_label}")
            log.info("Updated file", path)
            processed_count += 1
        except Exception as exc:
            msg = str(exc)
            log.fail(msg)
            mark_failed(failed_tracker, key, path, msg)
            failed_count += 1

    log.done_block(
        "Categorization task",
        [
            ("Categorized", processed_count),
            ("Failed", failed_count),
            ("Total attempted", len(files)),
            ("Model used", args.model),
            ("Transcriptions folder", TRANSCRIPTIONS_DIR),
            ("Processed tracker", PROCESSED_TRACKER_PATH),
        ],
        success=failed_count == 0,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
