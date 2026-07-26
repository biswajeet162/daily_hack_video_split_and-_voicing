"""Step 6: Launch the voice-over browser UI for split clips."""

from __future__ import annotations

import threading
import time
import webbrowser

from voiceover.server import PORT, run


def _open_browser() -> None:
    time.sleep(1.0)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    run()
