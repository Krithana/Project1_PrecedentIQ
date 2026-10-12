import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    app_path = PROJECT_ROOT / "src" / "precedentiq" / "streamlit_app.py"
    print(f"Launching PrecedentIQ Research Terminal: {app_path}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false"
    ])
