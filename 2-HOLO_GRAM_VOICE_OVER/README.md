# Voice Over Recorder

Web app to record voice-over for video chunks and merge them into a final video.

## Prerequisites

- **Python 3.10+**
- **Conda** (with `whisperx` environment)
- **ffmpeg** and **ffprobe** installed and available in PATH

Check ffmpeg:

```powershell
ffmpeg -version
ffprobe -version
```

## Setup (one time)

Activate your conda environment and install dependencies:

```powershell
conda activate whisperx
pip install faster-whisper
```

## Run the project

From the repo root, double-click:

```
hologram_run.bat
```

Or manually:

```powershell
cd D:\YOUTUBES\DAILY_HACKS_VIDEOS\2-HOLO_GRAM_VOICE_OVER
conda activate whisperx
python server.py
```

Open:

```
http://localhost:8000
```

> **Important:** Use `python server.py`, not `python -m http.server`.

## Option 1: Split One Video

1. Choose **Option 1: Split One Video**
2. **Select Video** – choose your `.mp4` file
3. Set **Chunk Size (sec)** – e.g. 6, 8, or 10
4. Click **Process Video**
   - Splits video into `split_videos/`
   - Transcribes each chunk into `transcription_folder/`
   - Creates `transcript.json`
   - Chunks shorter than 2 seconds are ignored
5. Click a chunk on the left – video and text appear on the right
6. Click **Start Recording** – speak for that chunk
7. After all chunks are recorded, click **Merge**
8. Download **final_output.mp4**

## Option 2: Load Video Chunks

Use this when you already have pre-split clip files and do not want to set a chunk size.

1. Choose **Option 2: Load Video Chunks**
2. **Select Video Chunks** – pick multiple `.mp4` files
3. Click **Load Chunks**
4. Click each part on the left:
   - Video loads
   - Transcription runs for that part
   - Record your voice-over
   - Part is marked **Done** when recording is saved
5. Use **↑ / ↓** on each left-side card to reorder parts
6. Click **Merge** when every part is Done
7. Download **final_output.mp4**

## Output folders

| Folder | Description |
|--------|-------------|
| `uploads/` | Uploaded source video / staged chunk uploads |
| `split_videos/` | Video chunks / loaded parts |
| `transcription_folder/` | Per-chunk transcript files |
| `recordings/` | Raw recorded audio |
| `recordings_fixed/` | Fixed-length WAV per chunk |
| `merged_parts/` | Temporary merge parts |
| `transcript.json` | Combined transcript for the UI |
| `session.json` | Active mode (`split` or `chunks`) |
| `final_output.mp4` | Final merged video |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `501 Unsupported method ('POST')` | Use `python server.py`, not `python -m http.server` |
| Transcription fails | Ensure `whisperx` env is active; try freeing GPU memory |
| `ffmpeg not found` | Install ffmpeg and add it to PATH |
| Microphone not working | Allow mic permission for `localhost:8000` in the browser |
