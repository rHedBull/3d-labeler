# Primitive Fitting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add cylinder and box primitive fitting modes that let users define a search region and automatically fit geometric primitives to point cloud data within that region.

**Architecture:** User defines a rough search region (cylinder or box) via clicks, backend runs RANSAC fitting to find primitives within that region, frontend displays candidates for user selection.

**Tech Stack:** Python (Open3D, scipy, numpy) for RANSAC fitting, React Three Fiber for 3D visualization, Zustand for state management.

---

## Task 1: Add Selection Modes to Store

**Files:**
- Modify: `frontend/src/store/selectionStore.ts`

**Step 1: Add new mode types**

```typescript
export type SelectionMode = 'box' | 'lasso' | 'sphere' | 'geometric' | 'supervoxel' | 'rapid' | 'cylinder-fit' | 'box-fit'
```

**Step 2: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS (or existing errors unrelated to this change)

**Step 3: Commit**

```bash
git add frontend/src/store/selectionStore.ts
git commit -m "feat: add cylinder-fit and box-fit selection modes"
```

---

## Task 2: Add Mode Buttons to Toolbar

**Files:**
- Modify: `frontend/src/components/ModeToolbar.tsx`

**Step 1: Add modes to MODES array**

Find the MODES array and add two new entries after 'rapid':

```typescript
const MODES: { id: SelectionMode; key: string; label: string; icon: string }[] = [
  { id: 'box', key: 'B', label: 'Box Select', icon: '▢' },
  { id: 'lasso', key: 'L', label: 'Lasso Select', icon: '◯' },
  { id: 'sphere', key: 'S', label: 'Sphere Select', icon: '●' },
  { id: 'geometric', key: 'G', label: 'Geometric Cluster', icon: '◈' },
  { id: 'supervoxel', key: 'V', label: 'Supervoxel', icon: '⬡' },
  { id: 'rapid', key: 'R', label: 'Rapid Labeling', icon: '⚡' },
  { id: 'cylinder-fit', key: 'C', label: 'Fit Cylinder', icon: '⬭' },
  { id: 'box-fit', key: 'X', label: 'Fit Box', icon: '⬜' },
]
```

**Step 2: Verify visually**

Run: `cd frontend && npm run dev`
Expected: Two new buttons appear in toolbar with C and X hotkeys

**Step 3: Commit**

```bash
git add frontend/src/components/ModeToolbar.tsx
git commit -m "feat: add cylinder-fit and box-fit mode buttons to toolbar"
```

---

## Task 3: Create Fitting Store

**Files:**
- Create: `frontend/src/store/fittingStore.ts`

**Step 1: Create the store**

```typescript
import { create } from 'zustand'
import * as THREE from 'three'

export interface FittedCylinder {
  id: number
  center: THREE.Vector3
  axis: THREE.Vector3
  radius: number
  height: number
  pointIndices: number[]
  accepted: boolean
}

export interface FittedBox {
  id: number
  center: THREE.Vector3
  size: THREE.Vector3
  rotation: THREE.Euler
  pointIndices: number[]
  accepted: boolean
}

interface CylinderRegion {
  center: THREE.Vector3
  axis: THREE.Vector3
  radius: number
  height: number
}

interface BoxRegion {
  corners: [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  height: number
}

interface FittingState {
  // Cylinder fitting
  cylinderPhase: 'none' | 'center' | 'radius' | 'height' | 'fitting' | 'selecting'
  cylinderCenter: THREE.Vector3 | null
  cylinderAxis: THREE.Vector3 | null
  cylinderRadius: number
  cylinderHeight: number
  fittedCylinders: FittedCylinder[]

  // Box fitting
  boxPhase: 'none' | 'corner1' | 'corner2' | 'corner3' | 'height' | 'fitting' | 'selecting'
  boxCorners: THREE.Vector3[]
  boxHeight: number
  fittedBoxes: FittedBox[]

  // Settings
  tolerance: number
  minInliers: number

  // Actions
  setCylinderPhase: (phase: FittingState['cylinderPhase']) => void
  setCylinderCenter: (center: THREE.Vector3 | null) => void
  setCylinderAxis: (axis: THREE.Vector3 | null) => void
  setCylinderRadius: (radius: number) => void
  setCylinderHeight: (height: number) => void
  setFittedCylinders: (cylinders: FittedCylinder[]) => void
  toggleCylinderAccepted: (id: number) => void

  setBoxPhase: (phase: FittingState['boxPhase']) => void
  addBoxCorner: (corner: THREE.Vector3) => void
  setBoxHeight: (height: number) => void
  setFittedBoxes: (boxes: FittedBox[]) => void
  toggleBoxAccepted: (id: number) => void

  setTolerance: (tolerance: number) => void
  setMinInliers: (minInliers: number) => void

  resetCylinder: () => void
  resetBox: () => void

  getCylinderRegion: () => CylinderRegion | null
  getBoxRegion: () => BoxRegion | null
}

export const useFittingStore = create<FittingState>((set, get) => ({
  // Cylinder state
  cylinderPhase: 'none',
  cylinderCenter: null,
  cylinderAxis: null,
  cylinderRadius: 0,
  cylinderHeight: 0,
  fittedCylinders: [],

  // Box state
  boxPhase: 'none',
  boxCorners: [],
  boxHeight: 0,
  fittedBoxes: [],

  // Settings
  tolerance: 0.02,
  minInliers: 500,

  // Cylinder actions
  setCylinderPhase: (phase) => set({ cylinderPhase: phase }),
  setCylinderCenter: (center) => set({ cylinderCenter: center }),
  setCylinderAxis: (axis) => set({ cylinderAxis: axis }),
  setCylinderRadius: (radius) => set({ cylinderRadius: radius }),
  setCylinderHeight: (height) => set({ cylinderHeight: height }),
  setFittedCylinders: (cylinders) => set({ fittedCylinders: cylinders }),
  toggleCylinderAccepted: (id) => set((state) => ({
    fittedCylinders: state.fittedCylinders.map(c =>
      c.id === id ? { ...c, accepted: !c.accepted } : c
    )
  })),

  // Box actions
  setBoxPhase: (phase) => set({ boxPhase: phase }),
  addBoxCorner: (corner) => set((state) => ({
    boxCorners: [...state.boxCorners, corner]
  })),
  setBoxHeight: (height) => set({ boxHeight: height }),
  setFittedBoxes: (boxes) => set({ fittedBoxes: boxes }),
  toggleBoxAccepted: (id) => set((state) => ({
    fittedBoxes: state.fittedBoxes.map(b =>
      b.id === id ? { ...b, accepted: !b.accepted } : b
    )
  })),

  // Settings
  setTolerance: (tolerance) => set({ tolerance }),
  setMinInliers: (minInliers) => set({ minInliers }),

  // Reset
  resetCylinder: () => set({
    cylinderPhase: 'none',
    cylinderCenter: null,
    cylinderAxis: null,
    cylinderRadius: 0,
    cylinderHeight: 0,
    fittedCylinders: [],
  }),
  resetBox: () => set({
    boxPhase: 'none',
    boxCorners: [],
    boxHeight: 0,
    fittedBoxes: [],
  }),

  // Get region for API
  getCylinderRegion: () => {
    const { cylinderCenter, cylinderAxis, cylinderRadius, cylinderHeight } = get()
    if (!cylinderCenter || !cylinderAxis) return null
    return { center: cylinderCenter, axis: cylinderAxis, radius: cylinderRadius, height: cylinderHeight }
  },
  getBoxRegion: () => {
    const { boxCorners, boxHeight } = get()
    if (boxCorners.length < 3) return null
    return { corners: boxCorners.slice(0, 3) as [THREE.Vector3, THREE.Vector3, THREE.Vector3], height: boxHeight }
  },
}))
```

**Step 2: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/store/fittingStore.ts
git commit -m "feat: add fitting store for cylinder and box fitting state"
```

---

## Task 4: Create Backend Fitting Module

**Files:**
- Create: `backend/fitting.py`

**Step 1: Create cylinder fitting with RANSAC**

```python
import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
import random


def fit_cylinder_ransac(
    points: np.ndarray,
    tolerance: float = 0.02,
    min_inliers: int = 500,
    max_iterations: int = 1000,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]]:
    """
    Fit a cylinder to points using RANSAC.

    Returns: (center, axis, radius, height, inlier_indices) or None if no fit found
    """
    n_points = len(points)
    if n_points < min_inliers:
        return None

    best_inliers = []
    best_center = None
    best_axis = None
    best_radius = None

    for _ in range(max_iterations):
        # Sample 3 random points
        if n_points < 3:
            break
        sample_idx = random.sample(range(n_points), 3)
        p1, p2, p3 = points[sample_idx]

        # Estimate axis from two points
        axis = p2 - p1
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            continue
        axis = axis / axis_len

        # Project third point to estimate radius
        v = p3 - p1
        proj = np.dot(v, axis) * axis
        perp = v - proj
        radius = np.linalg.norm(perp)

        if radius < 0.01:  # Too small
            continue

        # Center is on the axis
        center = p1

        # Count inliers - points close to cylinder surface
        inliers = []
        for i, p in enumerate(points):
            # Vector from center to point
            v = p - center
            # Component along axis
            along = np.dot(v, axis)
            # Perpendicular component
            perp_vec = v - along * axis
            dist_to_surface = abs(np.linalg.norm(perp_vec) - radius)

            if dist_to_surface < tolerance:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_center = center
            best_axis = axis
            best_radius = radius

    if len(best_inliers) < min_inliers:
        return None

    # Refine center and compute height
    inlier_points = points[best_inliers]
    projections = np.dot(inlier_points - best_center, best_axis)
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    height = max_proj - min_proj

    # Adjust center to middle of cylinder
    best_center = best_center + best_axis * (min_proj + max_proj) / 2

    return best_center, best_axis, best_radius, height, np.array(best_inliers)


def fit_cylinders_in_region(
    points: np.ndarray,
    region_center: np.ndarray,
    region_axis: np.ndarray,
    region_radius: float,
    region_height: float,
    tolerance: float = 0.02,
    min_inliers: int = 500,
    max_candidates: int = 10,
) -> List[dict]:
    """
    Find all cylinders within a cylindrical search region.

    Returns list of dicts with: center, axis, radius, height, point_indices
    """
    # Extract points within the region
    region_indices = []
    for i, p in enumerate(points):
        v = p - region_center
        along = np.dot(v, region_axis)
        perp = v - along * region_axis
        dist_from_axis = np.linalg.norm(perp)

        if abs(along) <= region_height / 2 and dist_from_axis <= region_radius:
            region_indices.append(i)

    if len(region_indices) < min_inliers:
        return []

    region_points = points[region_indices]
    region_indices = np.array(region_indices)

    candidates = []
    remaining_mask = np.ones(len(region_points), dtype=bool)

    for _ in range(max_candidates):
        remaining_points = region_points[remaining_mask]
        remaining_indices = np.where(remaining_mask)[0]

        if len(remaining_points) < min_inliers:
            break

        result = fit_cylinder_ransac(remaining_points, tolerance, min_inliers)
        if result is None:
            break

        center, axis, radius, height, local_inliers = result

        # Map local inliers back to original indices
        global_inliers = region_indices[remaining_indices[local_inliers]]

        candidates.append({
            'center': center.tolist(),
            'axis': axis.tolist(),
            'radius': float(radius),
            'height': float(height),
            'point_indices': global_inliers.tolist(),
        })

        # Remove inliers from remaining
        remaining_mask[remaining_indices[local_inliers]] = False

    return candidates


def fit_boxes_in_region(
    points: np.ndarray,
    corner1: np.ndarray,
    corner2: np.ndarray,
    corner3: np.ndarray,
    height: float,
    tolerance: float = 0.02,
    min_inliers: int = 500,
    max_candidates: int = 10,
) -> List[dict]:
    """
    Find all boxes within a box-shaped search region defined by 3 corners + height.

    Returns list of dicts with: center, size, rotation, point_indices
    """
    # Compute region basis vectors
    edge1 = corner2 - corner1
    edge2 = corner3 - corner1
    normal = np.cross(edge1, edge2)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # Normalize edges
    len1 = np.linalg.norm(edge1)
    len2 = np.linalg.norm(edge2)
    if len1 < 1e-6 or len2 < 1e-6:
        return []

    u1 = edge1 / len1
    u2 = edge2 / len2

    # Region center
    corner4 = corner1 + edge1 + edge2
    base_center = (corner1 + corner2 + corner3 + corner4) / 4
    region_center = base_center + normal * height / 2

    # Extract points in region (simple AABB check in local space)
    region_indices = []
    for i, p in enumerate(points):
        v = p - corner1
        proj1 = np.dot(v, u1)
        proj2 = np.dot(v, u2)
        proj_h = np.dot(v, normal)

        if 0 <= proj1 <= len1 and 0 <= proj2 <= len2 and 0 <= proj_h <= height:
            region_indices.append(i)

    if len(region_indices) < min_inliers:
        return []

    region_points = points[region_indices]
    region_indices = np.array(region_indices)

    # For boxes, use plane fitting to find box faces
    # Simplified approach: find dominant planes and group them
    candidates = []
    remaining_mask = np.ones(len(region_points), dtype=bool)

    for _ in range(max_candidates):
        remaining_points = region_points[remaining_mask]
        remaining_local_indices = np.where(remaining_mask)[0]

        if len(remaining_points) < min_inliers:
            break

        # Use PCA to find oriented bounding box
        centered = remaining_points - remaining_points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (largest first)
        order = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, order]

        # Transform to local coordinates
        local_coords = centered @ eigenvectors

        # Compute OBB
        local_min = local_coords.min(axis=0)
        local_max = local_coords.max(axis=0)
        local_center = (local_min + local_max) / 2
        local_size = local_max - local_min

        # Transform center back to world
        world_center = (local_center @ eigenvectors.T) + remaining_points.mean(axis=0)

        # Find inliers (points on the surface of the box)
        inliers = []
        for i, lc in enumerate(local_coords):
            # Distance to nearest face
            dist_to_faces = np.minimum(
                np.abs(lc - local_min),
                np.abs(lc - local_max)
            )
            min_dist = dist_to_faces.min()

            if min_dist < tolerance:
                inliers.append(i)

        if len(inliers) < min_inliers:
            break

        global_inliers = region_indices[remaining_local_indices[inliers]]

        # Convert rotation matrix to euler angles
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(eigenvectors)
        euler = rot.as_euler('xyz')

        candidates.append({
            'center': world_center.tolist(),
            'size': local_size.tolist(),
            'rotation': euler.tolist(),
            'point_indices': global_inliers.tolist(),
        })

        remaining_mask[remaining_local_indices[inliers]] = False

    return candidates
```

**Step 2: Run Python syntax check**

Run: `cd backend && python -m py_compile fitting.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add backend/fitting.py
git commit -m "feat: add RANSAC cylinder and box fitting algorithms"
```

---

## Task 5: Add API Endpoints

**Files:**
- Modify: `backend/main.py`

**Step 1: Add imports and models**

Add after existing imports:

```python
from fitting import fit_cylinders_in_region, fit_boxes_in_region
```

Add after existing Pydantic models:

```python
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
```

**Step 2: Add endpoints**

Add after existing endpoints:

```python
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
```

**Step 3: Run backend**

Run: `cd backend && source venv/bin/activate && python -c "from main import app; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat: add /fit-cylinders and /fit-boxes API endpoints"
```

---

## Task 6: Add API Functions to Frontend

**Files:**
- Modify: `frontend/src/lib/api.ts`

**Step 1: Read current api.ts to understand patterns**

**Step 2: Add fitting API functions**

```typescript
export interface CylinderCandidate {
  id: number
  center: [number, number, number]
  axis: [number, number, number]
  radius: number
  height: number
  pointIndices: Int32Array
}

export interface BoxCandidate {
  id: number
  center: [number, number, number]
  size: [number, number, number]
  rotation: [number, number, number]
  pointIndices: Int32Array
}

export async function fitCylinders(
  center: [number, number, number],
  axis: [number, number, number],
  radius: number,
  height: number,
  tolerance: number = 0.02,
  minInliers: number = 500,
): Promise<CylinderCandidate[]> {
  const res = await fetch(`${API_BASE}/fit-cylinders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      center,
      axis,
      radius,
      height,
      tolerance,
      min_inliers: minInliers,
    }),
  })

  if (!res.ok) {
    throw new Error(`Failed to fit cylinders: ${res.statusText}`)
  }

  const data = await res.json()
  return data.candidates.map((c: any) => ({
    id: c.id,
    center: c.center,
    axis: c.axis,
    radius: c.radius,
    height: c.height,
    pointIndices: new Int32Array(
      Uint8Array.from(atob(c.point_indices), c => c.charCodeAt(0)).buffer
    ),
  }))
}

export async function fitBoxes(
  corner1: [number, number, number],
  corner2: [number, number, number],
  corner3: [number, number, number],
  height: number,
  tolerance: number = 0.02,
  minInliers: number = 500,
): Promise<BoxCandidate[]> {
  const res = await fetch(`${API_BASE}/fit-boxes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      corner1,
      corner2,
      corner3,
      height,
      tolerance,
      min_inliers: minInliers,
    }),
  })

  if (!res.ok) {
    throw new Error(`Failed to fit boxes: ${res.statusText}`)
  }

  const data = await res.json()
  return data.candidates.map((c: any) => ({
    id: c.id,
    center: c.center,
    size: c.size,
    rotation: c.rotation,
    pointIndices: new Int32Array(
      Uint8Array.from(atob(c.point_indices), c => c.charCodeAt(0)).buffer
    ),
  }))
}
```

**Step 3: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 4: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat: add fitCylinders and fitBoxes API functions"
```

---

## Task 7: Create Cylinder Fit Handler Component

**Files:**
- Create: `frontend/src/components/CylinderFitHandler.tsx`

**Step 1: Create the component**

```typescript
import { useCallback, useEffect, useRef, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { usePointCloudStore } from '../store/pointCloudStore'
import { useSelectionStore } from '../store/selectionStore'
import { useFittingStore } from '../store/fittingStore'
import { fitCylinders } from '../lib/api'

// Find closest point to ray
function findClosestPointToRay(
  ray: THREE.Ray,
  points: Float32Array,
  numPoints: number,
  threshold: number = 0.5
): { index: number; position: THREE.Vector3 } | null {
  let closestDist = Infinity
  let closestIdx = -1
  const point = new THREE.Vector3()

  for (let i = 0; i < numPoints; i++) {
    point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
    const dist = ray.distanceToPoint(point)

    if (dist < closestDist && dist < threshold) {
      closestDist = dist
      closestIdx = i
    }
  }

  if (closestIdx >= 0) {
    return {
      index: closestIdx,
      position: new THREE.Vector3(
        points[closestIdx * 3],
        points[closestIdx * 3 + 1],
        points[closestIdx * 3 + 2]
      ),
    }
  }

  return null
}

export function CylinderFitHandler({
  onCandidatesReady,
}: {
  onCandidatesReady: () => void
}) {
  const { camera, raycaster, gl } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()
  const {
    cylinderPhase,
    cylinderCenter,
    cylinderAxis,
    cylinderRadius,
    cylinderHeight,
    setCylinderPhase,
    setCylinderCenter,
    setCylinderAxis,
    setCylinderRadius,
    setCylinderHeight,
    setFittedCylinders,
    tolerance,
    minInliers,
    resetCylinder,
  } = useFittingStore()

  const dragPlane = useRef<THREE.Plane>(new THREE.Plane())

  // Initialize phase when entering mode
  useEffect(() => {
    if (mode === 'cylinder-fit' && cylinderPhase === 'none') {
      setCylinderPhase('center')
    }
    if (mode !== 'cylinder-fit') {
      resetCylinder()
    }
  }, [mode, cylinderPhase, setCylinderPhase, resetCylinder])

  const handleClick = useCallback(async (e: MouseEvent) => {
    if (mode !== 'cylinder-fit' || e.button !== 0 || !points) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)

    if (cylinderPhase === 'center') {
      // Click to set center
      const hit = findClosestPointToRay(raycaster.ray, points, numPoints)
      if (hit) {
        setCylinderCenter(hit.position)
        // Default axis is camera up
        const cameraUp = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion)
        setCylinderAxis(cameraUp.normalize())
        setCylinderPhase('radius')

        // Set up drag plane perpendicular to axis
        dragPlane.current.setFromNormalAndCoplanarPoint(cameraUp, hit.position)
      }
    } else if (cylinderPhase === 'radius') {
      // Click to lock radius
      setCylinderPhase('height')
    } else if (cylinderPhase === 'height') {
      // Click to lock height and trigger fitting
      setCylinderPhase('fitting')

      if (cylinderCenter && cylinderAxis) {
        try {
          const candidates = await fitCylinders(
            [cylinderCenter.x, cylinderCenter.y, cylinderCenter.z],
            [cylinderAxis.x, cylinderAxis.y, cylinderAxis.z],
            cylinderRadius,
            cylinderHeight,
            tolerance,
            minInliers,
          )

          setFittedCylinders(candidates.map(c => ({
            id: c.id,
            center: new THREE.Vector3(...c.center),
            axis: new THREE.Vector3(...c.axis),
            radius: c.radius,
            height: c.height,
            pointIndices: Array.from(c.pointIndices),
            accepted: true, // Default to accepted
          })))

          setCylinderPhase('selecting')
          onCandidatesReady()
        } catch (err) {
          console.error('Cylinder fitting failed:', err)
          resetCylinder()
        }
      }
    }
  }, [mode, cylinderPhase, points, numPoints, camera, raycaster, gl,
      cylinderCenter, cylinderAxis, cylinderRadius, cylinderHeight,
      setCylinderCenter, setCylinderAxis, setCylinderPhase, setCylinderRadius, setCylinderHeight,
      setFittedCylinders, tolerance, minInliers, resetCylinder, onCandidatesReady])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'cylinder-fit' || !cylinderCenter) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)

    if (cylinderPhase === 'radius') {
      // Update radius based on distance from center
      const intersection = new THREE.Vector3()
      raycaster.ray.intersectPlane(dragPlane.current, intersection)
      if (intersection) {
        const radius = cylinderCenter.distanceTo(intersection)
        setCylinderRadius(Math.max(0.1, radius))
      }
    } else if (cylinderPhase === 'height' && cylinderAxis) {
      // Update height along axis
      const closestPoint = new THREE.Vector3()
      raycaster.ray.closestPointToPoint(cylinderCenter, closestPoint)
      const toMouse = closestPoint.clone().sub(cylinderCenter)
      const heightAlong = Math.abs(toMouse.dot(cylinderAxis))
      setCylinderHeight(Math.max(0.1, heightAlong * 2))
    }
  }, [mode, cylinderPhase, cylinderCenter, cylinderAxis, camera, raycaster, gl, setCylinderRadius, setCylinderHeight])

  // Handle keyboard
  useEffect(() => {
    if (mode !== 'cylinder-fit') return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        resetCylinder()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mode, resetCylinder])

  // Attach mouse listeners
  useEffect(() => {
    if (mode !== 'cylinder-fit') return

    const canvas = gl.domElement
    canvas.addEventListener('click', handleClick)
    canvas.addEventListener('mousemove', handleMouseMove)

    return () => {
      canvas.removeEventListener('click', handleClick)
      canvas.removeEventListener('mousemove', handleMouseMove)
    }
  }, [mode, gl, handleClick, handleMouseMove])

  // Render preview cylinder
  const previewGeometry = useMemo(() => {
    if (!cylinderCenter || cylinderPhase === 'none' || cylinderPhase === 'center') return null

    return {
      center: cylinderCenter,
      axis: cylinderAxis || new THREE.Vector3(0, 1, 0),
      radius: cylinderRadius || 0.5,
      height: cylinderHeight || 1,
    }
  }, [cylinderCenter, cylinderAxis, cylinderRadius, cylinderHeight, cylinderPhase])

  if (mode !== 'cylinder-fit' || !previewGeometry) return null

  // Calculate rotation to align cylinder with axis
  const quaternion = new THREE.Quaternion()
  quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), previewGeometry.axis)

  return (
    <group position={previewGeometry.center} quaternion={quaternion}>
      <mesh>
        <cylinderGeometry args={[previewGeometry.radius, previewGeometry.radius, previewGeometry.height, 32]} />
        <meshBasicMaterial color="#00ffff" wireframe transparent opacity={0.5} />
      </mesh>
    </group>
  )
}
```

**Step 2: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/CylinderFitHandler.tsx
git commit -m "feat: add CylinderFitHandler component for cylinder region definition"
```

---

## Task 8: Create Box Fit Handler Component

**Files:**
- Create: `frontend/src/components/BoxFitHandler.tsx`

**Step 1: Create the component**

```typescript
import { useCallback, useEffect, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { usePointCloudStore } from '../store/pointCloudStore'
import { useSelectionStore } from '../store/selectionStore'
import { useFittingStore } from '../store/fittingStore'
import { fitBoxes } from '../lib/api'

// Find closest point to ray
function findClosestPointToRay(
  ray: THREE.Ray,
  points: Float32Array,
  numPoints: number,
  threshold: number = 0.5
): { index: number; position: THREE.Vector3 } | null {
  let closestDist = Infinity
  let closestIdx = -1
  const point = new THREE.Vector3()

  for (let i = 0; i < numPoints; i++) {
    point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
    const dist = ray.distanceToPoint(point)

    if (dist < closestDist && dist < threshold) {
      closestDist = dist
      closestIdx = i
    }
  }

  if (closestIdx >= 0) {
    return {
      index: closestIdx,
      position: new THREE.Vector3(
        points[closestIdx * 3],
        points[closestIdx * 3 + 1],
        points[closestIdx * 3 + 2]
      ),
    }
  }

  return null
}

export function BoxFitHandler({
  onCandidatesReady,
}: {
  onCandidatesReady: () => void
}) {
  const { camera, raycaster, gl } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()
  const {
    boxPhase,
    boxCorners,
    boxHeight,
    setBoxPhase,
    addBoxCorner,
    setBoxHeight,
    setFittedBoxes,
    tolerance,
    minInliers,
    resetBox,
  } = useFittingStore()

  // Initialize phase when entering mode
  useEffect(() => {
    if (mode === 'box-fit' && boxPhase === 'none') {
      setBoxPhase('corner1')
    }
    if (mode !== 'box-fit') {
      resetBox()
    }
  }, [mode, boxPhase, setBoxPhase, resetBox])

  const handleClick = useCallback(async (e: MouseEvent) => {
    if (mode !== 'box-fit' || e.button !== 0 || !points) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)

    const hit = findClosestPointToRay(raycaster.ray, points, numPoints)
    if (!hit && boxPhase !== 'height') return

    if (boxPhase === 'corner1') {
      if (hit) {
        addBoxCorner(hit.position)
        setBoxPhase('corner2')
      }
    } else if (boxPhase === 'corner2') {
      if (hit) {
        addBoxCorner(hit.position)
        setBoxPhase('corner3')
      }
    } else if (boxPhase === 'corner3') {
      if (hit) {
        addBoxCorner(hit.position)
        setBoxPhase('height')
      }
    } else if (boxPhase === 'height') {
      // Click to lock height and trigger fitting
      setBoxPhase('fitting')

      if (boxCorners.length >= 3) {
        try {
          const candidates = await fitBoxes(
            [boxCorners[0].x, boxCorners[0].y, boxCorners[0].z],
            [boxCorners[1].x, boxCorners[1].y, boxCorners[1].z],
            [boxCorners[2].x, boxCorners[2].y, boxCorners[2].z],
            boxHeight,
            tolerance,
            minInliers,
          )

          setFittedBoxes(candidates.map(c => ({
            id: c.id,
            center: new THREE.Vector3(...c.center),
            size: new THREE.Vector3(...c.size),
            rotation: new THREE.Euler(...c.rotation),
            pointIndices: Array.from(c.pointIndices),
            accepted: true,
          })))

          setBoxPhase('selecting')
          onCandidatesReady()
        } catch (err) {
          console.error('Box fitting failed:', err)
          resetBox()
        }
      }
    }
  }, [mode, boxPhase, boxCorners, boxHeight, points, numPoints, camera, raycaster, gl,
      addBoxCorner, setBoxPhase, setBoxHeight, setFittedBoxes, tolerance, minInliers, resetBox, onCandidatesReady])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'box-fit' || boxPhase !== 'height' || boxCorners.length < 3) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)

    // Compute base plane normal
    const edge1 = boxCorners[1].clone().sub(boxCorners[0])
    const edge2 = boxCorners[2].clone().sub(boxCorners[0])
    const normal = edge1.cross(edge2).normalize()

    // Project mouse ray onto normal direction
    const baseCenter = boxCorners[0].clone()
      .add(boxCorners[1])
      .add(boxCorners[2])
      .divideScalar(3)

    const toMouse = new THREE.Vector3()
    raycaster.ray.closestPointToPoint(baseCenter, toMouse)
    const heightAlong = Math.abs(toMouse.sub(baseCenter).dot(normal))
    setBoxHeight(Math.max(0.1, heightAlong * 2))
  }, [mode, boxPhase, boxCorners, camera, raycaster, gl, setBoxHeight])

  // Handle keyboard
  useEffect(() => {
    if (mode !== 'box-fit') return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        resetBox()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mode, resetBox])

  // Attach mouse listeners
  useEffect(() => {
    if (mode !== 'box-fit') return

    const canvas = gl.domElement
    canvas.addEventListener('click', handleClick)
    canvas.addEventListener('mousemove', handleMouseMove)

    return () => {
      canvas.removeEventListener('click', handleClick)
      canvas.removeEventListener('mousemove', handleMouseMove)
    }
  }, [mode, gl, handleClick, handleMouseMove])

  // Render preview
  const previewGeometry = useMemo(() => {
    if (boxCorners.length === 0) return null

    if (boxCorners.length === 1) {
      // Just show first corner as a sphere
      return { type: 'point' as const, position: boxCorners[0] }
    }

    if (boxCorners.length === 2) {
      // Show line between corners
      return { type: 'line' as const, points: boxCorners }
    }

    if (boxCorners.length >= 3) {
      // Show base rectangle + height
      const edge1 = boxCorners[1].clone().sub(boxCorners[0])
      const edge2 = boxCorners[2].clone().sub(boxCorners[0])
      const corner4 = boxCorners[0].clone().add(edge1).add(edge2)
      const normal = edge1.clone().cross(edge2).normalize()

      return {
        type: 'box' as const,
        corners: [boxCorners[0], boxCorners[1], corner4, boxCorners[2]],
        normal,
        height: boxHeight,
      }
    }

    return null
  }, [boxCorners, boxHeight])

  if (mode !== 'box-fit' || !previewGeometry) return null

  if (previewGeometry.type === 'point') {
    return (
      <mesh position={previewGeometry.position}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshBasicMaterial color="#ffff00" />
      </mesh>
    )
  }

  if (previewGeometry.type === 'line') {
    const linePoints = previewGeometry.points.flatMap(p => [p.x, p.y, p.z])
    return (
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array(linePoints)}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ffff00" />
      </line>
    )
  }

  if (previewGeometry.type === 'box') {
    const { corners, normal, height } = previewGeometry
    const baseCenter = corners.reduce((acc, c) => acc.add(c), new THREE.Vector3()).divideScalar(4)
    const center = baseCenter.clone().add(normal.clone().multiplyScalar(height / 2))

    // Calculate size
    const sizeX = corners[0].distanceTo(corners[1])
    const sizeZ = corners[0].distanceTo(corners[3])

    // Calculate rotation
    const edge1 = corners[1].clone().sub(corners[0]).normalize()
    const up = normal
    const edge2 = up.clone().cross(edge1)

    const matrix = new THREE.Matrix4().makeBasis(edge1, up, edge2)
    const quaternion = new THREE.Quaternion().setFromRotationMatrix(matrix)

    return (
      <group position={center} quaternion={quaternion}>
        <mesh>
          <boxGeometry args={[sizeX, height, sizeZ]} />
          <meshBasicMaterial color="#ffff00" wireframe transparent opacity={0.5} />
        </mesh>
      </group>
    )
  }

  return null
}
```

**Step 2: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/BoxFitHandler.tsx
git commit -m "feat: add BoxFitHandler component for box region definition"
```

---

## Task 9: Create Fit Candidates Panel

**Files:**
- Create: `frontend/src/components/FitCandidatesPanel.tsx`

**Step 1: Create the panel component**

```typescript
import { useSelectionStore } from '../store/selectionStore'
import { useFittingStore } from '../store/fittingStore'
import { usePointCloudStore } from '../store/pointCloudStore'

export function FitCandidatesPanel() {
  const { mode } = useSelectionStore()
  const { setSelection } = usePointCloudStore()
  const {
    cylinderPhase,
    fittedCylinders,
    toggleCylinderAccepted,
    resetCylinder,
    boxPhase,
    fittedBoxes,
    toggleBoxAccepted,
    resetBox,
    tolerance,
    setTolerance,
  } = useFittingStore()

  const isCylinderSelecting = mode === 'cylinder-fit' && cylinderPhase === 'selecting'
  const isBoxSelecting = mode === 'box-fit' && boxPhase === 'selecting'

  if (!isCylinderSelecting && !isBoxSelecting) return null

  const handleApply = () => {
    // Collect all accepted point indices
    const indices = new Set<number>()

    if (isCylinderSelecting) {
      for (const c of fittedCylinders) {
        if (c.accepted) {
          for (const i of c.pointIndices) {
            indices.add(i)
          }
        }
      }
      resetCylinder()
    }

    if (isBoxSelecting) {
      for (const b of fittedBoxes) {
        if (b.accepted) {
          for (const i of b.pointIndices) {
            indices.add(i)
          }
        }
      }
      resetBox()
    }

    setSelection(indices)
  }

  const handleCancel = () => {
    if (isCylinderSelecting) resetCylinder()
    if (isBoxSelecting) resetBox()
  }

  const candidates = isCylinderSelecting ? fittedCylinders : fittedBoxes
  const toggleAccepted = isCylinderSelecting ? toggleCylinderAccepted : toggleBoxAccepted

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        {isCylinderSelecting ? 'Fitted Cylinders' : 'Fitted Boxes'}
      </div>

      {candidates.length === 0 ? (
        <div style={styles.empty}>No primitives found. Try increasing tolerance.</div>
      ) : (
        <div style={styles.list}>
          {candidates.map((c, i) => (
            <div
              key={c.id}
              style={{
                ...styles.item,
                background: c.accepted ? 'rgba(0, 255, 0, 0.2)' : 'rgba(255, 0, 0, 0.2)',
              }}
              onClick={() => toggleAccepted(c.id)}
            >
              <span style={styles.itemIcon}>{c.accepted ? '✓' : '✗'}</span>
              <span>
                {isCylinderSelecting ? 'Cylinder' : 'Box'} {i + 1}
              </span>
              <span style={styles.pointCount}>
                {c.pointIndices.length.toLocaleString()} pts
              </span>
            </div>
          ))}
        </div>
      )}

      <div style={styles.settings}>
        <label style={styles.label}>
          Tolerance: {tolerance.toFixed(3)}m
        </label>
        <input
          type="range"
          min="0.005"
          max="0.1"
          step="0.005"
          value={tolerance}
          onChange={(e) => setTolerance(parseFloat(e.target.value))}
          style={styles.slider}
        />
      </div>

      <div style={styles.buttons}>
        <button onClick={handleApply} style={styles.applyButton}>
          Apply Selected
        </button>
        <button onClick={handleCancel} style={styles.cancelButton}>
          Cancel
        </button>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    position: 'absolute',
    top: 60,
    right: 12,
    width: 220,
    background: 'rgba(30, 30, 50, 0.95)',
    borderRadius: 8,
    padding: 12,
    color: 'white',
    zIndex: 100,
  },
  header: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 8,
    borderBottom: '1px solid rgba(255,255,255,0.2)',
    paddingBottom: 8,
  },
  empty: {
    color: '#888',
    fontSize: 12,
    padding: '12px 0',
    textAlign: 'center',
  },
  list: {
    maxHeight: 200,
    overflowY: 'auto',
    marginBottom: 12,
  },
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '6px 8px',
    borderRadius: 4,
    marginBottom: 4,
    cursor: 'pointer',
    fontSize: 12,
  },
  itemIcon: {
    fontSize: 14,
  },
  pointCount: {
    marginLeft: 'auto',
    color: '#888',
    fontSize: 11,
  },
  settings: {
    marginBottom: 12,
    paddingTop: 8,
    borderTop: '1px solid rgba(255,255,255,0.2)',
  },
  label: {
    fontSize: 11,
    color: '#888',
    display: 'block',
    marginBottom: 4,
  },
  slider: {
    width: '100%',
  },
  buttons: {
    display: 'flex',
    gap: 8,
  },
  applyButton: {
    flex: 1,
    padding: '8px 12px',
    background: '#4a9',
    border: 'none',
    borderRadius: 4,
    color: 'white',
    cursor: 'pointer',
    fontSize: 12,
  },
  cancelButton: {
    flex: 1,
    padding: '8px 12px',
    background: '#666',
    border: 'none',
    borderRadius: 4,
    color: 'white',
    cursor: 'pointer',
    fontSize: 12,
  },
}
```

**Step 2: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/FitCandidatesPanel.tsx
git commit -m "feat: add FitCandidatesPanel for selecting fitted primitives"
```

---

## Task 10: Create Fitted Primitives Visualization

**Files:**
- Create: `frontend/src/components/FittedPrimitivesView.tsx`

**Step 1: Create visualization component**

```typescript
import { useMemo } from 'react'
import * as THREE from 'three'
import { useSelectionStore } from '../store/selectionStore'
import { useFittingStore } from '../store/fittingStore'

// Colors for candidates
const COLORS = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#fd79a8', '#a29bfe', '#00b894', '#e17055']

export function FittedPrimitivesView() {
  const { mode } = useSelectionStore()
  const { cylinderPhase, fittedCylinders, boxPhase, fittedBoxes } = useFittingStore()

  const showCylinders = mode === 'cylinder-fit' && cylinderPhase === 'selecting'
  const showBoxes = mode === 'box-fit' && boxPhase === 'selecting'

  const cylinderMeshes = useMemo(() => {
    if (!showCylinders) return null

    return fittedCylinders.map((c, i) => {
      const color = COLORS[i % COLORS.length]
      const quaternion = new THREE.Quaternion()
      quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), c.axis)

      return (
        <group key={c.id} position={c.center} quaternion={quaternion}>
          <mesh>
            <cylinderGeometry args={[c.radius, c.radius, c.height, 32]} />
            <meshBasicMaterial
              color={color}
              wireframe
              transparent
              opacity={c.accepted ? 0.8 : 0.3}
            />
          </mesh>
        </group>
      )
    })
  }, [showCylinders, fittedCylinders])

  const boxMeshes = useMemo(() => {
    if (!showBoxes) return null

    return fittedBoxes.map((b, i) => {
      const color = COLORS[i % COLORS.length]

      return (
        <group key={b.id} position={b.center} rotation={b.rotation}>
          <mesh>
            <boxGeometry args={[b.size.x, b.size.y, b.size.z]} />
            <meshBasicMaterial
              color={color}
              wireframe
              transparent
              opacity={b.accepted ? 0.8 : 0.3}
            />
          </mesh>
        </group>
      )
    })
  }, [showBoxes, fittedBoxes])

  return (
    <>
      {cylinderMeshes}
      {boxMeshes}
    </>
  )
}
```

**Step 2: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 3: Commit**

```bash
git add frontend/src/components/FittedPrimitivesView.tsx
git commit -m "feat: add FittedPrimitivesView for visualizing fitted candidates"
```

---

## Task 11: Integrate Components into Viewport

**Files:**
- Modify: `frontend/src/components/Viewport.tsx`

**Step 1: Add imports**

Add after existing imports:

```typescript
import { CylinderFitHandler } from './CylinderFitHandler'
import { BoxFitHandler } from './BoxFitHandler'
import { FittedPrimitivesView } from './FittedPrimitivesView'
import { useFittingStore } from '../store/fittingStore'
```

**Step 2: Add fitting store usage in Viewport component**

Inside the Viewport component, add:

```typescript
const { cylinderPhase, boxPhase } = useFittingStore()
```

**Step 3: Add handlers and views inside Canvas/OrbitControlsContext.Provider**

Add after `<RapidLabelingController />`:

```typescript
<CylinderFitHandler onCandidatesReady={() => {}} />
<BoxFitHandler onCandidatesReady={() => {}} />
<FittedPrimitivesView />
```

**Step 4: Add instructions UI for fitting modes**

Add after the rapid labeling UI section, before the closing `</div>`:

```typescript
{/* Cylinder fit mode instructions */}
{mode === 'cylinder-fit' && (
  <div style={{
    position: 'absolute',
    bottom: 12,
    left: '50%',
    transform: 'translateX(-50%)',
    background: 'rgba(0, 0, 0, 0.7)',
    color: 'white',
    padding: '8px 16px',
    borderRadius: 4,
    fontSize: 12,
  }}>
    {cylinderPhase === 'center' && 'Click to place cylinder center'}
    {cylinderPhase === 'radius' && 'Move mouse to set radius, click to confirm'}
    {cylinderPhase === 'height' && 'Move mouse to set height, click to fit'}
    {cylinderPhase === 'fitting' && 'Fitting cylinders...'}
    {cylinderPhase === 'selecting' && 'Click candidates to toggle, then Apply or Cancel'}
  </div>
)}

{/* Box fit mode instructions */}
{mode === 'box-fit' && (
  <div style={{
    position: 'absolute',
    bottom: 12,
    left: '50%',
    transform: 'translateX(-50%)',
    background: 'rgba(0, 0, 0, 0.7)',
    color: 'white',
    padding: '8px 16px',
    borderRadius: 4,
    fontSize: 12,
  }}>
    {boxPhase === 'corner1' && 'Click first corner of base rectangle'}
    {boxPhase === 'corner2' && 'Click second corner'}
    {boxPhase === 'corner3' && 'Click third corner'}
    {boxPhase === 'height' && 'Move mouse to set height, click to fit'}
    {boxPhase === 'fitting' && 'Fitting boxes...'}
    {boxPhase === 'selecting' && 'Click candidates to toggle, then Apply or Cancel'}
  </div>
)}
```

**Step 5: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 6: Commit**

```bash
git add frontend/src/components/Viewport.tsx
git commit -m "feat: integrate cylinder and box fitting into Viewport"
```

---

## Task 12: Add FitCandidatesPanel to App

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: Add import**

```typescript
import { FitCandidatesPanel } from './components/FitCandidatesPanel'
```

**Step 2: Add panel to render**

Add `<FitCandidatesPanel />` alongside other panels in the App component.

**Step 3: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: add FitCandidatesPanel to App"
```

---

## Task 13: Add Keyboard Shortcuts

**Files:**
- Modify: `frontend/src/hooks/useKeyboard.ts`

**Step 1: Read current file to understand pattern**

**Step 2: Add C and X shortcuts for fitting modes**

Add cases for 'c' and 'x' keys to switch to cylinder-fit and box-fit modes.

**Step 3: Run TypeScript check**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

**Step 4: Commit**

```bash
git add frontend/src/hooks/useKeyboard.ts
git commit -m "feat: add C and X keyboard shortcuts for fitting modes"
```

---

## Task 14: End-to-End Testing

**Step 1: Start backend**

Run: `cd backend && source venv/bin/activate && uvicorn main:app --reload`

**Step 2: Start frontend**

Run: `cd frontend && npm run dev`

**Step 3: Manual test checklist**

- [ ] Load a point cloud
- [ ] Press C to enter cylinder fit mode
- [ ] Click to set center point
- [ ] Move mouse to adjust radius, click to confirm
- [ ] Move mouse to adjust height, click to fit
- [ ] Verify candidates appear in panel
- [ ] Click candidates to toggle acceptance
- [ ] Click Apply to select points
- [ ] Press X to enter box fit mode
- [ ] Click 3 corners to define base
- [ ] Move mouse to adjust height, click to fit
- [ ] Verify candidates appear
- [ ] Click Apply to select points
- [ ] Press Escape to cancel at any phase

**Step 4: Final commit**

```bash
git add -A
git commit -m "test: verify primitive fitting end-to-end"
```

---

## Summary

This plan implements:

1. **Backend**: RANSAC-based cylinder and box fitting algorithms
2. **Frontend**:
   - New selection modes (cylinder-fit, box-fit)
   - Interactive region definition (click-based)
   - Candidate visualization and selection UI
   - Keyboard shortcuts (C, X)
3. **Integration**: Full workflow from region definition → API call → candidate selection → point selection
