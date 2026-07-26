# Video Split Project

Download YouTube shorts, split them by on-screen number markers (1–5), and transcribe each clip to Hindi.

Double-click **`run.bat`** in this folder (or the root **`run.bat`**) to open the task menu.

## Setup (one time)

```powershell
conda activate utube_env
pip install -r requirements.txt
```

Step **[3] Transcribe** uses **faster-whisper** with the **large-v3** model (same engine as `2-HOLO_GRAM_VOICE_OVER`). GPU is used when available; it falls back to CPU automatically.

Re-transcribe clips that used the old openai-whisper `base` model:

```powershell
conda run -n utube_env python 03_transcribe_videos.py --force
```

Use `--device cpu --compute int8` if you do not have a CUDA GPU.



"font_size": 80,

mulporl of 16