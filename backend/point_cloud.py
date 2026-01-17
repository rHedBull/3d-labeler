import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from pathlib import Path
from typing import Optional
import struct

class PointCloud:
    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        instance_ids: Optional[np.ndarray] = None,
    ):
        self.points = points.astype(np.float32)
        self.colors = colors.astype(np.uint8) if colors is not None else None
        self.labels = labels.astype(np.int32) if labels is not None else np.zeros(len(points), dtype=np.int32)
        self.instance_ids = instance_ids.astype(np.int32) if instance_ids is not None else np.zeros(len(points), dtype=np.int32)

    def __len__(self):
        return len(self.points)


def load_glb(path: Path, num_samples: int = 500000) -> PointCloud:
    """Load GLB mesh and sample points from surface."""
    mesh = trimesh.load(str(path), force='mesh')

    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes in scene
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid meshes found in GLB file")

    # Sample points from surface
    points, face_indices = mesh.sample(num_samples, return_index=True)

    # Get colors from vertex colors or face colors
    colors = None
    if mesh.visual.kind == 'vertex':
        # Interpolate vertex colors
        colors = mesh.visual.vertex_colors[mesh.faces[face_indices]].mean(axis=1)[:, :3]
    elif mesh.visual.kind == 'face':
        colors = mesh.visual.face_colors[face_indices][:, :3]

    if colors is None:
        colors = np.full((len(points), 3), 128, dtype=np.uint8)

    return PointCloud(points=points, colors=colors.astype(np.uint8))


def load_ply(path: Path) -> PointCloud:
    """Load PLY file with optional labels and instance_ids."""
    plydata = PlyData.read(str(path))
    vertex = plydata['vertex']

    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    # Colors
    colors = None
    if 'red' in vertex.data.dtype.names:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T

    # Labels
    labels = None
    if 'label' in vertex.data.dtype.names:
        labels = np.array(vertex['label'])

    # Instance IDs
    instance_ids = None
    if 'instance_id' in vertex.data.dtype.names:
        instance_ids = np.array(vertex['instance_id'])

    return PointCloud(points=points, colors=colors, labels=labels, instance_ids=instance_ids)


def save_ply(path: Path, pc: PointCloud) -> int:
    """Save point cloud to PLY with labels and instance_ids."""
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'i4'), ('instance_id', 'i4'),
    ]

    data = np.zeros(len(pc), dtype=dtype)
    data['x'] = pc.points[:, 0]
    data['y'] = pc.points[:, 1]
    data['z'] = pc.points[:, 2]

    if pc.colors is not None:
        data['red'] = pc.colors[:, 0]
        data['green'] = pc.colors[:, 1]
        data['blue'] = pc.colors[:, 2]
    else:
        data['red'] = data['green'] = data['blue'] = 128

    data['label'] = pc.labels
    data['instance_id'] = pc.instance_ids

    vertex = PlyElement.describe(data, 'vertex')
    PlyData([vertex], text=False).write(str(path))

    return len(pc)
