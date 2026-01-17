#!/usr/bin/env python3
"""Generate a synthetic point cloud for testing the labeling app."""

import numpy as np
from pathlib import Path

def generate_cylinder(center, radius, height, axis='z', n_points=5000):
    """Generate points on a cylinder surface."""
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    h = np.random.uniform(-height/2, height/2, n_points)

    if axis == 'z':
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2] + h
    elif axis == 'x':
        x = center[0] + h
        y = center[1] + radius * np.cos(theta)
        z = center[2] + radius * np.sin(theta)
    elif axis == 'y':
        x = center[0] + radius * np.cos(theta)
        y = center[1] + h
        z = center[2] + radius * np.sin(theta)

    return np.column_stack([x, y, z])

def generate_sphere(center, radius, n_points=3000):
    """Generate points on a sphere surface."""
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    theta = np.arccos(np.random.uniform(-1, 1, n_points))

    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)

    return np.column_stack([x, y, z])

def generate_torus(center, major_radius, minor_radius, n_points=4000):
    """Generate points on a torus (elbow) surface."""
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi/2, n_points)  # Quarter torus for elbow

    x = center[0] + (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y = center[1] + (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z = center[2] + minor_radius * np.sin(theta)

    return np.column_stack([x, y, z])

def generate_box(center, size, n_points=3000):
    """Generate points on a box surface (for structural elements)."""
    points = []
    pts_per_face = n_points // 6

    # 6 faces
    for axis in range(3):
        for sign in [-1, 1]:
            face_pts = np.random.uniform(-0.5, 0.5, (pts_per_face, 3)) * size
            face_pts[:, axis] = sign * size[axis] / 2
            points.append(face_pts + center)

    return np.vstack(points)

def generate_plane(corner, size, normal='z', n_points=5000):
    """Generate points on a plane (floor/wall)."""
    u = np.random.uniform(0, size[0], n_points)
    v = np.random.uniform(0, size[1], n_points)

    if normal == 'z':
        x = corner[0] + u
        y = corner[1] + v
        z = np.full(n_points, corner[2])
    elif normal == 'y':
        x = corner[0] + u
        y = np.full(n_points, corner[1])
        z = corner[2] + v
    elif normal == 'x':
        x = np.full(n_points, corner[0])
        y = corner[1] + u
        z = corner[2] + v

    return np.column_stack([x, y, z])

def add_noise(points, sigma=0.005):
    """Add Gaussian noise to points."""
    return points + np.random.normal(0, sigma, points.shape)

def save_ply(path, points, colors):
    """Save point cloud as PLY file."""
    header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int label
property int instance_id
end_header
"""

    # Create structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'i4'), ('instance_id', 'i4'),
    ]

    data = np.zeros(len(points), dtype=dtype)
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['red'] = colors[:, 0]
    data['green'] = colors[:, 1]
    data['blue'] = colors[:, 2]
    data['label'] = 0  # Unlabeled
    data['instance_id'] = 0

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

    print(f"Saved {len(points)} points to {path}")

def main():
    np.random.seed(42)

    all_points = []
    all_colors = []

    # Ground plane (will be labeled as background)
    ground = generate_plane([-5, -5, 0], [10, 10], normal='z', n_points=8000)
    ground = add_noise(ground, 0.01)
    ground_color = np.full((len(ground), 3), [100, 100, 100], dtype=np.uint8)
    all_points.append(ground)
    all_colors.append(ground_color)

    # Horizontal pipe 1
    pipe1 = generate_cylinder([0, 0, 1.5], radius=0.15, height=6, axis='x', n_points=8000)
    pipe1 = add_noise(pipe1)
    pipe1_color = np.full((len(pipe1), 3), [80, 80, 180], dtype=np.uint8)
    all_points.append(pipe1)
    all_colors.append(pipe1_color)

    # Horizontal pipe 2 (perpendicular)
    pipe2 = generate_cylinder([0, 0, 2.5], radius=0.12, height=5, axis='y', n_points=6000)
    pipe2 = add_noise(pipe2)
    pipe2_color = np.full((len(pipe2), 3), [70, 90, 170], dtype=np.uint8)
    all_points.append(pipe2)
    all_colors.append(pipe2_color)

    # Vertical pipe
    pipe3 = generate_cylinder([2, 1, 1.5], radius=0.1, height=3, axis='z', n_points=5000)
    pipe3 = add_noise(pipe3)
    pipe3_color = np.full((len(pipe3), 3), [90, 85, 175], dtype=np.uint8)
    all_points.append(pipe3)
    all_colors.append(pipe3_color)

    # Elbow 1 (connecting pipes)
    elbow1 = generate_torus([3, 0, 1.5], major_radius=0.3, minor_radius=0.15, n_points=3000)
    elbow1 = add_noise(elbow1)
    elbow1_color = np.full((len(elbow1), 3), [100, 200, 200], dtype=np.uint8)
    all_points.append(elbow1)
    all_colors.append(elbow1_color)

    # Elbow 2
    elbow2 = generate_torus([-2, 0, 1.5], major_radius=0.25, minor_radius=0.15, n_points=2500)
    elbow2 = add_noise(elbow2)
    elbow2_color = np.full((len(elbow2), 3), [90, 190, 195], dtype=np.uint8)
    all_points.append(elbow2)
    all_colors.append(elbow2_color)

    # Tank (large cylinder)
    tank = generate_cylinder([-3, 2, 1.5], radius=1.0, height=2.5, axis='z', n_points=15000)
    tank = add_noise(tank)
    tank_color = np.full((len(tank), 3), [80, 180, 80], dtype=np.uint8)
    all_points.append(tank)
    all_colors.append(tank_color)

    # Tank top cap
    tank_top_theta = np.random.uniform(0, 2*np.pi, 3000)
    tank_top_r = np.sqrt(np.random.uniform(0, 1, 3000)) * 1.0
    tank_top = np.column_stack([
        -3 + tank_top_r * np.cos(tank_top_theta),
        2 + tank_top_r * np.sin(tank_top_theta),
        np.full(3000, 2.75)
    ])
    tank_top = add_noise(tank_top)
    tank_top_color = np.full((len(tank_top), 3), [75, 175, 75], dtype=np.uint8)
    all_points.append(tank_top)
    all_colors.append(tank_top_color)

    # Valve 1 (small cylinder with box)
    valve1_body = generate_cylinder([1, 0, 1.5], radius=0.2, height=0.4, axis='y', n_points=2000)
    valve1_body = add_noise(valve1_body)
    valve1_color = np.full((len(valve1_body), 3), [200, 80, 80], dtype=np.uint8)
    all_points.append(valve1_body)
    all_colors.append(valve1_color)

    valve1_wheel = generate_cylinder([1, 0.3, 1.5], radius=0.15, height=0.05, axis='y', n_points=1000)
    valve1_wheel = add_noise(valve1_wheel)
    valve1_wheel_color = np.full((len(valve1_wheel), 3), [220, 100, 100], dtype=np.uint8)
    all_points.append(valve1_wheel)
    all_colors.append(valve1_wheel_color)

    # Valve 2
    valve2_body = generate_cylinder([0, 1.5, 2.5], radius=0.18, height=0.35, axis='x', n_points=1800)
    valve2_body = add_noise(valve2_body)
    valve2_color = np.full((len(valve2_body), 3), [190, 70, 70], dtype=np.uint8)
    all_points.append(valve2_body)
    all_colors.append(valve2_color)

    # Structural beam 1 (I-beam approximation)
    beam1 = generate_box([3, -2, 1.5], [0.2, 4, 0.3], n_points=4000)
    beam1 = add_noise(beam1)
    beam1_color = np.full((len(beam1), 3), [200, 200, 80], dtype=np.uint8)
    all_points.append(beam1)
    all_colors.append(beam1_color)

    # Structural beam 2 (vertical)
    beam2 = generate_box([3, -4, 1.5], [0.15, 0.15, 3], n_points=3000)
    beam2 = add_noise(beam2)
    beam2_color = np.full((len(beam2), 3), [210, 210, 90], dtype=np.uint8)
    all_points.append(beam2)
    all_colors.append(beam2_color)

    # Structural beam 3
    beam3 = generate_box([-4, -3, 1.5], [0.15, 0.15, 3], n_points=3000)
    beam3 = add_noise(beam3)
    beam3_color = np.full((len(beam3), 3), [205, 205, 85], dtype=np.uint8)
    all_points.append(beam3)
    all_colors.append(beam3_color)

    # Clutter - random small objects
    for i in range(5):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        clutter = generate_sphere([x, y, 0.15], radius=0.1 + np.random.uniform(0, 0.1), n_points=500)
        clutter = add_noise(clutter)
        clutter_color = np.full((len(clutter), 3), [200 + np.random.randint(0, 55),
                                                     100 + np.random.randint(0, 50),
                                                     50 + np.random.randint(0, 50)], dtype=np.uint8)
        all_points.append(clutter)
        all_colors.append(clutter_color)

    # Combine all
    points = np.vstack(all_points).astype(np.float32)
    colors = np.vstack(all_colors).astype(np.uint8)

    # Shuffle to mix up the points
    indices = np.random.permutation(len(points))
    points = points[indices]
    colors = colors[indices]

    # Save
    output_dir = Path(__file__).parent.parent / "data" / "real" / "test_scene"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_ply(output_dir / "source.ply", points, colors)

    print(f"\nGenerated test scene with:")
    print(f"  - Ground plane (background)")
    print(f"  - 3 pipes")
    print(f"  - 2 elbows")
    print(f"  - 1 tank with cap")
    print(f"  - 2 valves")
    print(f"  - 3 structural beams")
    print(f"  - 5 clutter objects")
    print(f"\nTotal: {len(points):,} points")

if __name__ == "__main__":
    main()
