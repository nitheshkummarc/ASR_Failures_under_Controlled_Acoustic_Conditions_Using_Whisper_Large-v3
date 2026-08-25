import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.service:app", host="127.0.0.1", port=8000, reload=True)
