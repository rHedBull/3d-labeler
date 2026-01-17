import numpy as np
import open3d as o3d
from typing import Tuple


def compute_supervoxels(
    points: np.ndarray,
    resolution: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute supervoxels using voxel downsampling and clustering.

    Args:
        points: Nx3 array of point positions
        resolution: voxel size for downsampling

    Returns:
        supervoxel_ids: Int32Array of supervoxel ID per point
        centroids: Float32Array of supervoxel centroids (N x 3)
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
    for sv_id, point_indices in enumerate(indices):
        for idx in point_indices:
            supervoxel_ids[idx] = sv_id

    centroids = np.asarray(downsampled.points).astype(np.float32)

    return supervoxel_ids, centroids
