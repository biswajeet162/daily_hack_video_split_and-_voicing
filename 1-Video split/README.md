# Video Split Project

Download YouTube shorts, split them by on-screen number markers (1–5), blur on-screen text, transcribe each clip to Hindi, categorize, and merge by topic.

Double-click **`run.bat`** in this folder (or the root **`run.bat`**) to open the task menu.

Choose **`A`** (or type `all`) to run steps **1 → 5**, then **7** (auto merge). Step **6** voice-over is manual in the browser.

## Project layout

```
1-Video split/
├── configuration/          ← step settings (edit these)
│   ├── split_trim_config.json
│   ├── remove_text_config.json
│   └── merge_transition_config.json
├── trackers/               ← one-time progress logs (auto-managed)
│   ├── download_archive.txt
│   ├── split_processed.json
│   ├── text_removed_processed.json
│   ├── transcribe_processed.json
│   ├── categorize_processed.json
│   ├── voiceover_processed.json
│   └── merge_used_clips.json
├── 1-input_videos/
├── 2-output_videos/
├── 3-output_voiceover_videos/
├── 4-output_merged_videos/
└── 01_… 02_… 03_… scripts
```

## Setup (one time)

```powershell
conda activate utube_env
pip install -r requirements.txt
```

Step **[3] Blur text** — edit `configuration/remove_text_config.json` (blur strength, languages).

Step **[4] Transcribe** uses **faster-whisper** `large-v3`. Re-run with `--force` if needed:

```powershell
conda run -n utube_env python 04_transcribe_videos.py --force
```

Re-blur clips after config changes:

```powershell
conda run -n utube_env python 03_remove_text_from_videos.py --force
```
