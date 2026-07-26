# Video Split Project

Download YouTube shorts, split them by on-screen number markers (1–5), remove burned-in text, transcribe each clip to Hindi, categorize, and merge by topic.

Double-click **`run.bat`** in this folder (or the root **`run.bat`**) to open the task menu.

## Setup (one time)

```powershell
conda activate utube_env
pip install -r requirements.txt
```

Step **[3] Remove text** scans **every frame** with **EasyOCR on GPU** (English, Hindi, Chinese), inpaints detected text, and tracks finished chunks in `output_videos/.text_removed_processed.json` so each clip is only processed once.

Step **[4] Transcribe** uses **faster-whisper** with the **large-v3** model (same engine as `2-HOLO_GRAM_VOICE_OVER`). GPU is used when available; it falls back to CPU automatically.

Re-transcribe clips that used the old openai-whisper `base` model:

```powershell
conda run -n utube_env python 04_transcribe_videos.py --force
```

Use `--device cpu --compute int8` if you do not have a CUDA GPU.

Re-run text removal on already-cleaned clips:

```powershell
conda run -n utube_env python 03_remove_text_from_videos.py --force
```
