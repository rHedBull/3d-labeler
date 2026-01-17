from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List
import base64
import numpy as np

from point_cloud import load_glb, load_ply, save_ply, PointCloud
from supervoxels import compute_supervoxels

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
current_supervoxels: tuple[np.ndarray, np.ndarray] | None = None
DATA_DIR = Path(__file__).parent.parent / "data" / "real"


class LoadRequest(BaseModel):
    path: str
    num_samples: int = 500000


class LoadResponse(BaseModel):
    num_points: int
    points: str  # base64 encoded Float32Array
    colors: str | None  # base64 encoded Uint8Array
    labels: str  # base64 encoded Int32Array


class SaveRequest(BaseModel):
    labels: str  # base64 encoded Int32Array
    instance_ids: str  # base64 encoded Int32Array
    scene_name: str


class SaveResponse(BaseModel):
    success: bool
    num_points: int
    path: str


class SupervoxelRequest(BaseModel):
    resolution: float = 0.1


class SupervoxelResponse(BaseModel):
    num_supervoxels: int
    supervoxel_ids: str  # base64 encoded Int32Array
    centroids: str  # base64 encoded Float32Array


class SceneInfo(BaseModel):
    name: str
    has_source: bool
    has_ground_truth: bool
    source_type: str | None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/files", response_model=List[SceneInfo])
async def list_files():
    scenes = []

    if not DATA_DIR.exists():
        return scenes

    for scene_dir in sorted(DATA_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue

        source_glb = scene_dir / "source.glb"
        source_ply = scene_dir / "source.ply"
        gt_ply = scene_dir / "ground_truth.ply"

        has_source = source_glb.exists() or source_ply.exists()
        source_type = None
        if source_glb.exists():
            source_type = "glb"
        elif source_ply.exists():
            source_type = "ply"

        scenes.append(SceneInfo(
            name=scene_dir.name,
            has_source=has_source,
            has_ground_truth=gt_ply.exists(),
            source_type=source_type,
        ))

    return scenes


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


@app.post("/save", response_model=SaveResponse)
async def save_file(req: SaveRequest):
    global current_pc

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    # Validate scene_name doesn't escape DATA_DIR
    output_dir = (DATA_DIR / req.scene_name).resolve()
    if not str(output_dir).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(400, "Invalid scene_name: path traversal not allowed")

    try:
        # Decode labels
        labels = np.frombuffer(base64.b64decode(req.labels), dtype=np.int32)
        instance_ids = np.frombuffer(base64.b64decode(req.instance_ids), dtype=np.int32)

        if len(labels) != len(current_pc):
            raise HTTPException(400, f"Label count mismatch: {len(labels)} vs {len(current_pc)}")

        current_pc.labels = labels
        current_pc.instance_ids = instance_ids

        # Save to ground_truth.ply
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "ground_truth.ply"

        num_saved = save_ply(output_path, current_pc)

        return SaveResponse(
            success=True,
            num_points=num_saved,
            path=str(output_path.relative_to(DATA_DIR.parent.parent)),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")


@app.post("/compute-supervoxels", response_model=SupervoxelResponse)
async def compute_supervoxels_endpoint(req: SupervoxelRequest):
    global current_pc, current_supervoxels

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    try:
        sv_ids, centroids = compute_supervoxels(current_pc.points, req.resolution)
        current_supervoxels = (sv_ids, centroids)

        return SupervoxelResponse(
            num_supervoxels=len(centroids),
            supervoxel_ids=base64.b64encode(sv_ids.tobytes()).decode(),
            centroids=base64.b64encode(centroids.tobytes()).decode(),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to compute supervoxels: {str(e)}")
