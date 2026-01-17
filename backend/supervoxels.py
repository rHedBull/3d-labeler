import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional


def compute_supervoxels(
    points: np.ndarray,
    resolution: float = 0.1,
    compute_hulls: bool = True,
    exclude_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[dict]]]:
    """
    Compute supervoxels using voxel downsampling and clustering.

    Args:
        points: Nx3 array of point positions
        resolution: voxel size for downsampling
        compute_hulls: whether to compute bounding boxes for each supervoxel
        exclude_mask: Int32Array where points with value > 0 are excluded from computation

    Returns:
        supervoxel_ids: Int32Array of supervoxel ID per point (-1 for excluded points)
        centroids: Float32Array of supervoxel centroids (N x 3)
        hulls: List of hull dicts with 'vertices' and 'faces' for each supervoxel (as boxes)
    """
    num_points = len(points)

    # Determine which points to include
    if exclude_mask is not None:
        include_mask = exclude_mask <= 0
        included_indices = np.where(include_mask)[0]
        filtered_points = points[included_indices]
    else:
        included_indices = np.arange(num_points)
        filtered_points = points

    # Handle case where all points are excluded
    if len(filtered_points) == 0:
        return (
            np.full(num_points, -1, dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            [] if compute_hulls else None,
        )

    # Create Open3D point cloud with only included points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Voxel downsampling to get supervoxel centers
    downsampled, _, indices = pcd.voxel_down_sample_and_trace(
        voxel_size=resolution,
        min_bound=pcd.get_min_bound() - resolution,
        max_bound=pcd.get_max_bound() + resolution,
    )

    # Create supervoxel IDs - map back to original point indices
    # Initialize with -1 for excluded points
    supervoxel_ids = np.full(num_points, -1, dtype=np.int32)
    point_groups: List[List[int]] = [[] for _ in range(len(indices))]

    for sv_id, filtered_indices in enumerate(indices):
        for filtered_idx in filtered_indices:
            # Map filtered index back to original index
            original_idx = included_indices[filtered_idx]
            supervoxel_ids[original_idx] = sv_id
            point_groups[sv_id].append(original_idx)

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
