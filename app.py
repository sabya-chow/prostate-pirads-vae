from __future__ import annotations

import sys
from pathlib import Path


# Hugging Face Spaces expects an app.py at the repository root. The actual
# Gradio app lives in app/app.py, so we load that folder first.
APP_DIR = Path(__file__).resolve().parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import demo  # noqa: E402


if __name__ == "__main__":
    demo.launch()
