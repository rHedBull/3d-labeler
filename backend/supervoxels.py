import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional


def compute_supervoxels(
    points: np.ndarray,
    resolution: float = 0.1,
    compute_hulls: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[dict]]]:
    """
    Compute supervoxels using voxel downsampling and clustering.

    Args:
        points: Nx3 array of point positions
        resolution: voxel size for downsampling
        compute_hulls: whether to compute bounding boxes for each supervoxel

    Returns:
        supervoxel_ids: Int32Array of supervoxel ID per point
        centroids: Float32Array of supervoxel centroids (N x 3)
        hulls: List of hull dicts with 'vertices' and 'faces' for each supervoxel (as boxes)
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Voxel downsampling to get supervoxel centers
    downsampled, _, indices = pcd.voxel_down_sample_and_trace(
        voxel_size=resolution,
        min_bound=pcd.get_min_bound() - resolution,
        max_bound=pcd.get_max_bound() + resolution,
    )

    # Create supervoxel IDs - each point gets the ID of its voxel
    supervoxel_ids = np.zeros(len(points), dtype=np.int32)
    point_groups: List[List[int]] = [[] for _ in range(len(indices))]

    for sv_id, point_indices in enumerate(indices):
        for idx in point_indices:
            supervoxel_ids[idx] = sv_id
            point_groups[sv_id].append(idx)

    centroids = np.asarray(downsampled.points).astype(np.float32)

    # Compute bounding boxes for each supervoxel
    hulls = None
    if compute_hulls:
        hulls = []
        for sv_id, group_indices in enumerate(point_groups):
            sv_points = points[group_indices]

            if len(group_indices) == 1:
                # Single point - create small box around it
                center = sv_points[0]
                half = resolution * 0.4
                min_pt = center - half
                max_pt = center + half
            else:
                # Compute bounding box with small padding
                min_pt = sv_points.min(axis=0) - resolution * 0.05
                max_pt = sv_points.max(axis=0) + resolution * 0.05

            # Create box vertices (8 corners)
            vertices = [
                [min_pt[0], min_pt[1], min_pt[2]],  # 0
                [max_pt[0], min_pt[1], min_pt[2]],  # 1
                [max_pt[0], max_pt[1], min_pt[2]],  # 2
                [min_pt[0], max_pt[1], min_pt[2]],  # 3
                [min_pt[0], min_pt[1], max_pt[2]],  # 4
                [max_pt[0], min_pt[1], max_pt[2]],  # 5
                [max_pt[0], max_pt[1], max_pt[2]],  # 6
                [min_pt[0], max_pt[1], max_pt[2]],  # 7
            ]

            # Create box faces (12 triangles, 2 per face)
            faces = [
                # Bottom face
                [0, 2, 1], [0, 3, 2],
                # Top face
                [4, 5, 6], [4, 6, 7],
                # Front face
                [0, 1, 5], [0, 5, 4],
                # Back face
                [2, 3, 7], [2, 7, 6],
                # Left face
                [0, 4, 7], [0, 7, 3],
                # Right face
                [1, 2, 6], [1, 6, 5],
            ]

            hulls.append({
                'vertices': vertices,
                'faces': faces,
            })

    return supervoxel_ids, centroids, hulls
