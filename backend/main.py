from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Any
import base64
import json
import numpy as np

from point_cloud import load_glb, load_ply, save_ply, PointCloud
from supervoxels import compute_supervoxels
from clustering import region_grow
from fitting import fit_cylinders_in_region, fit_boxes_in_region

app = FastAPI(title="Point Cloud Labeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5175"],
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
    instance_ids: str | None  # base64 encoded Int32Array


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
    exclude_mask: str | None = None  # base64 encoded Int32Array - points with value > 0 are excluded


class SupervoxelHull(BaseModel):
    vertices: List[List[float]]  # Nx3 vertices of the hull
    faces: List[List[int]]  # Triangle faces (indices into vertices)


class SupervoxelResponse(BaseModel):
    num_supervoxels: int
    supervoxel_ids: str  # base64 encoded Int32Array
    centroids: str  # base64 encoded Float32Array
    hulls: List[SupervoxelHull]  # Convex hull for each supervoxel


class ClusterRequest(BaseModel):
    seed_index: int
    normal_threshold: float = 15.0
    distance_threshold: float = 0.05
    max_points: int = 50000


class ClusterResponse(BaseModel):
    indices: str  # base64 encoded Int32Array
    num_points: int


class ExtractRequest(BaseModel):
    indices: str  # base64 encoded Int32Array
    scene_name: str
    filename: str  # output filename (without .ply extension)


class ExtractResponse(BaseModel):
    success: bool
    num_points: int
    path: str


class SceneInfo(BaseModel):
    name: str
    has_source: bool
    has_ground_truth: bool
    source_type: str | None


class CylinderFitRequest(BaseModel):
    center: List[float]  # [x, y, z]
    axis: List[float]    # [x, y, z] normalized
    radius: float
    height: float
    tolerance: float = 0.02
    min_inliers: int = 500


class CylinderCandidate(BaseModel):
    id: int
    center: List[float]
    axis: List[float]
    radius: float
    height: float
    point_indices: str  # base64 encoded Int32Array


class CylinderFitResponse(BaseModel):
    candidates: List[CylinderCandidate]


class BoxFitRequest(BaseModel):
    corner1: List[float]
    corner2: List[float]
    corner3: List[float]
    height: float
    tolerance: float = 0.02
    min_inliers: int = 500


class BoxCandidate(BaseModel):
    id: int
    center: List[float]
    size: List[float]
    rotation: List[float]  # euler angles xyz
    point_indices: str  # base64 encoded Int32Array


class BoxFitResponse(BaseModel):
    candidates: List[BoxCandidate]


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
        instance_ids=base64.b64encode(current_pc.instance_ids.tobytes()).decode() if current_pc.instance_ids is not None else None,
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


@app.post("/extract", response_model=ExtractResponse)
async def extract_points(req: ExtractRequest):
    global current_pc

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    # Validate scene_name doesn't escape DATA_DIR
    output_dir = (DATA_DIR / req.scene_name).resolve()
    if not str(output_dir).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(400, "Invalid scene_name: path traversal not allowed")

    # Validate filename
    if not req.filename or '/' in req.filename or '\\' in req.filename:
        raise HTTPException(400, "Invalid filename")

    try:
        # Decode indices
        indices = np.frombuffer(base64.b64decode(req.indices), dtype=np.int32)

        if len(indices) == 0:
            raise HTTPException(400, "No points selected")

        # Validate indices
        if np.any(indices < 0) or np.any(indices >= len(current_pc)):
            raise HTTPException(400, "Invalid point indices")

        # Extract selected points
        extracted_pc = PointCloud(
            points=current_pc.points[indices],
            colors=current_pc.colors[indices] if current_pc.colors is not None else None,
            labels=current_pc.labels[indices] if current_pc.labels is not None else None,
            instance_ids=current_pc.instance_ids[indices] if current_pc.instance_ids is not None else None,
        )

        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{req.filename}.ply"

        num_saved = save_ply(output_path, extracted_pc)

        return ExtractResponse(
            success=True,
            num_points=num_saved,
            path=str(output_path.relative_to(DATA_DIR.parent.parent)),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to extract points: {str(e)}")


@app.post("/compute-supervoxels", response_model=SupervoxelResponse)
async def compute_supervoxels_endpoint(req: SupervoxelRequest):
    global current_pc, current_supervoxels

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    try:
        # Decode exclude mask if provided (labels array - exclude points with label > 0)
        exclude_mask = None
        if req.exclude_mask:
            exclude_mask = np.frombuffer(base64.b64decode(req.exclude_mask), dtype=np.int32)

        sv_ids, centroids, hulls = compute_supervoxels(
            current_pc.points,
            req.resolution,
            compute_hulls=True,
            exclude_mask=exclude_mask,
        )
        current_supervoxels = (sv_ids, centroids, hulls)

        # Convert hulls to response format
        hull_responses = [
            SupervoxelHull(vertices=h['vertices'], faces=h['faces'])
            for h in hulls
        ]

        return SupervoxelResponse(
            num_supervoxels=len(centroids),
            supervoxel_ids=base64.b64encode(sv_ids.tobytes()).decode(),
            centroids=base64.b64encode(centroids.tobytes()).decode(),
            hulls=hull_responses,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to compute supervoxels: {str(e)}")


@app.post("/cluster", response_model=ClusterResponse)
async def compute_cluster(req: ClusterRequest):
    global current_pc

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    if req.seed_index < 0 or req.seed_index >= len(current_pc):
        raise HTTPException(400, f"Invalid seed index: {req.seed_index}")

    try:
        indices = region_grow(
            current_pc.points,
            req.seed_index,
            normal_threshold_deg=req.normal_threshold,
            distance_threshold=req.distance_threshold,
            max_points=req.max_points,
        )

        indices_array = np.array(indices, dtype=np.int32)

        return ClusterResponse(
            indices=base64.b64encode(indices_array.tobytes()).decode(),
            num_points=len(indices),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to compute cluster: {str(e)}")


@app.post("/fit-cylinders", response_model=CylinderFitResponse)
async def fit_cylinders(req: CylinderFitRequest):
    global current_pc

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    try:
        candidates = fit_cylinders_in_region(
            current_pc.points.reshape(-1, 3),
            np.array(req.center),
            np.array(req.axis),
            req.radius,
            req.height,
            req.tolerance,
            req.min_inliers,
        )

        response_candidates = []
        for i, c in enumerate(candidates):
            indices = np.array(c['point_indices'], dtype=np.int32)
            response_candidates.append(CylinderCandidate(
                id=i,
                center=c['center'],
                axis=c['axis'],
                radius=c['radius'],
                height=c['height'],
                point_indices=base64.b64encode(indices.tobytes()).decode(),
            ))

        return CylinderFitResponse(candidates=response_candidates)
    except Exception as e:
        raise HTTPException(500, f"Failed to fit cylinders: {str(e)}")


@app.post("/fit-boxes", response_model=BoxFitResponse)
async def fit_boxes(req: BoxFitRequest):
    global current_pc

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    try:
        candidates = fit_boxes_in_region(
            current_pc.points.reshape(-1, 3),
            np.array(req.corner1),
            np.array(req.corner2),
            np.array(req.corner3),
            req.height,
            req.tolerance,
            req.min_inliers,
        )

        response_candidates = []
        for i, c in enumerate(candidates):
            indices = np.array(c['point_indices'], dtype=np.int32)
            response_candidates.append(BoxCandidate(
                id=i,
                center=c['center'],
                size=c['size'],
                rotation=c['rotation'],
                point_indices=base64.b64encode(indices.tobytes()).decode(),
            ))

        return BoxFitResponse(candidates=response_candidates)
    except Exception as e:
        raise HTTPException(500, f"Failed to fit boxes: {str(e)}")
