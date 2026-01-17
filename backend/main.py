from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import base64
import numpy as np

from point_cloud import load_glb, load_ply, PointCloud

app = FastAPI(title="Point Cloud Labeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store current point cloud in memory
current_pc: PointCloud | None = None
DATA_DIR = Path(__file__).parent.parent / "data" / "real"


class LoadRequest(BaseModel):
    path: str
    num_samples: int = 500000


class LoadResponse(BaseModel):
    num_points: int
    points: str  # base64 encoded Float32Array
    colors: str | None  # base64 encoded Uint8Array
    labels: str  # base64 encoded Int32Array


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/load", response_model=LoadResponse)
async def load_file(req: LoadRequest):
    global current_pc

    # Validate path doesn't escape DATA_DIR
    file_path = (DATA_DIR / req.path).resolve()
    if not str(file_path).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(400, "Invalid path: path traversal not allowed")

    if not file_path.exists():
        raise HTTPException(404, f"File not found: {req.path}")

    suffix = file_path.suffix.lower()
    try:
        if suffix == '.glb':
            current_pc = load_glb(file_path, req.num_samples)
        elif suffix == '.ply':
            current_pc = load_ply(file_path)
        else:
            raise HTTPException(400, f"Unsupported file type: {suffix}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load file: {str(e)}")

    return LoadResponse(
        num_points=len(current_pc),
        points=base64.b64encode(current_pc.points.tobytes()).decode(),
        colors=base64.b64encode(current_pc.colors.tobytes()).decode() if current_pc.colors is not None else None,
        labels=base64.b64encode(current_pc.labels.tobytes()).decode(),
    )
