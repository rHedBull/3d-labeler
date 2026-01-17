# Point Cloud Labeling App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a web-based tool for labeling industrial point clouds to create ground truth training data.

**Architecture:** FastAPI backend handles GLB/PLY file I/O, supervoxel computation, and geometric clustering. React+Three.js frontend provides 3D viewport with 5 selection modes (box, lasso, sphere, geometric, supervoxel) and 7 class labels. Points are stored as typed arrays for performance.

**Tech Stack:** Python (FastAPI, Open3D, trimesh, plyfile), TypeScript (React, Three.js, Vite)

---

## Phase 1: Project Scaffolding

### Task 1.1: Create Backend Structure

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/main.py`

**Step 1: Create requirements.txt**

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
trimesh==4.0.8
open3d==0.18.0
plyfile==1.0.2
numpy==1.26.3
pydantic==2.5.3
```

**Step 2: Create minimal FastAPI app**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Point Cloud Labeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Step 3: Test backend starts**

```bash
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# In another terminal: curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

**Step 4: Commit**

```bash
git add backend/
git commit -m "feat: add backend scaffolding with FastAPI"
```

---

### Task 1.2: Create Frontend Structure

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/index.css`

**Step 1: Initialize Vite React project**

```bash
cd /home/hendrik/coding/3d-labeler
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install three @types/three @react-three/fiber @react-three/drei zustand
```

**Step 2: Update vite.config.ts for proxy**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
```

**Step 3: Create minimal App.tsx**

```typescript
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'

function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Canvas camera={{ position: [0, 0, 10], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <OrbitControls />
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="orange" />
        </mesh>
      </Canvas>
    </div>
  )
}

export default App
```

**Step 4: Update index.css**

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, #root {
  width: 100%;
  height: 100%;
  overflow: hidden;
}
```

**Step 5: Test frontend starts**

```bash
cd frontend && npm run dev
# Open http://localhost:5173 - should see orange cube
```

**Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: add frontend scaffolding with Vite, React, Three.js"
```

---

### Task 1.3: Create Shared Class Definitions

**Files:**
- Create: `labeling_classes.yaml`

**Step 1: Create class definitions file**

```yaml
# labeling_classes.yaml - Shared between labeling app, training repo, and visualizer
classes:
  - { id: 0, name: background, color: [128, 128, 128] }
  - { id: 1, name: pipe, color: [0, 0, 255] }
  - { id: 2, name: elbow, color: [0, 255, 255] }
  - { id: 3, name: valve, color: [255, 0, 0] }
  - { id: 4, name: tank, color: [0, 255, 0] }
  - { id: 5, name: structural, color: [255, 255, 0] }
  - { id: 6, name: clutter, color: [255, 128, 0] }
```

**Step 2: Commit**

```bash
git add labeling_classes.yaml
git commit -m "feat: add shared class definitions"
```

---

## Phase 2: Backend Core - File Loading

### Task 2.1: Point Cloud Loading Module

**Files:**
- Create: `backend/point_cloud.py`
- Modify: `backend/main.py`

**Step 1: Create point_cloud.py**

```python
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
```

**Step 2: Add load endpoint to main.py**

```python
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

    file_path = DATA_DIR / req.path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {req.path}")

    suffix = file_path.suffix.lower()
    if suffix == '.glb':
        current_pc = load_glb(file_path, req.num_samples)
    elif suffix == '.ply':
        current_pc = load_ply(file_path)
    else:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    return LoadResponse(
        num_points=len(current_pc),
        points=base64.b64encode(current_pc.points.tobytes()).decode(),
        colors=base64.b64encode(current_pc.colors.tobytes()).decode() if current_pc.colors is not None else None,
        labels=base64.b64encode(current_pc.labels.tobytes()).decode(),
    )
```

**Step 3: Create test data directory**

```bash
mkdir -p /home/hendrik/coding/3d-labeler/data/real/test_scene
```

**Step 4: Commit**

```bash
git add backend/point_cloud.py backend/main.py
git commit -m "feat: add point cloud loading (GLB/PLY support)"
```

---

### Task 2.2: Save Endpoint

**Files:**
- Modify: `backend/main.py`

**Step 1: Add save endpoint**

Add to `main.py`:

```python
class SaveRequest(BaseModel):
    labels: str  # base64 encoded Int32Array
    instance_ids: str  # base64 encoded Int32Array
    scene_name: str


class SaveResponse(BaseModel):
    success: bool
    num_points: int
    path: str


@app.post("/save", response_model=SaveResponse)
async def save_file(req: SaveRequest):
    global current_pc

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    # Decode labels
    labels = np.frombuffer(base64.b64decode(req.labels), dtype=np.int32)
    instance_ids = np.frombuffer(base64.b64decode(req.instance_ids), dtype=np.int32)

    if len(labels) != len(current_pc):
        raise HTTPException(400, f"Label count mismatch: {len(labels)} vs {len(current_pc)}")

    current_pc.labels = labels
    current_pc.instance_ids = instance_ids

    # Save to ground_truth.ply
    output_dir = DATA_DIR / req.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ground_truth.ply"

    from point_cloud import save_ply
    num_saved = save_ply(output_path, current_pc)

    return SaveResponse(
        success=True,
        num_points=num_saved,
        path=str(output_path.relative_to(DATA_DIR.parent.parent)),
    )
```

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "feat: add save endpoint for ground truth PLY export"
```

---

### Task 2.3: Files List Endpoint

**Files:**
- Modify: `backend/main.py`

**Step 1: Add files endpoint**

Add to `main.py`:

```python
from typing import List

class SceneInfo(BaseModel):
    name: str
    has_source: bool
    has_ground_truth: bool
    source_type: str | None


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
```

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "feat: add files list endpoint"
```

---

## Phase 3: Frontend Core - Point Cloud Display

### Task 3.1: API Client

**Files:**
- Create: `frontend/src/lib/api.ts`

**Step 1: Create API client**

```typescript
const API_BASE = '/api'

export interface LoadResponse {
  num_points: number
  points: string  // base64
  colors: string | null
  labels: string
}

export interface SceneInfo {
  name: string
  has_source: boolean
  has_ground_truth: boolean
  source_type: string | null
}

export async function loadPointCloud(path: string, numSamples = 500000): Promise<LoadResponse> {
  const res = await fetch(`${API_BASE}/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, num_samples: numSamples }),
  })
  if (!res.ok) throw new Error(`Load failed: ${res.statusText}`)
  return res.json()
}

export async function savePointCloud(
  labels: Int32Array,
  instanceIds: Int32Array,
  sceneName: string
): Promise<{ success: boolean; num_points: number; path: string }> {
  const res = await fetch(`${API_BASE}/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      labels: arrayToBase64(labels),
      instance_ids: arrayToBase64(instanceIds),
      scene_name: sceneName,
    }),
  })
  if (!res.ok) throw new Error(`Save failed: ${res.statusText}`)
  return res.json()
}

export async function listFiles(): Promise<SceneInfo[]> {
  const res = await fetch(`${API_BASE}/files`)
  if (!res.ok) throw new Error(`List failed: ${res.statusText}`)
  return res.json()
}

// Helpers
export function base64ToFloat32Array(b64: string): Float32Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Float32Array(bytes.buffer)
}

export function base64ToUint8Array(b64: string): Uint8Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

export function base64ToInt32Array(b64: string): Int32Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Int32Array(bytes.buffer)
}

function arrayToBase64(arr: Int32Array | Float32Array | Uint8Array): string {
  const bytes = new Uint8Array(arr.buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}
```

**Step 2: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat: add API client for backend communication"
```

---

### Task 3.2: Point Cloud Store

**Files:**
- Create: `frontend/src/store/pointCloudStore.ts`

**Step 1: Create Zustand store**

```typescript
import { create } from 'zustand'
import {
  loadPointCloud,
  savePointCloud,
  base64ToFloat32Array,
  base64ToUint8Array,
  base64ToInt32Array,
} from '../lib/api'

// Class colors from labeling_classes.yaml
export const CLASS_COLORS: Record<number, [number, number, number]> = {
  0: [128, 128, 128], // background
  1: [0, 0, 255],     // pipe
  2: [0, 255, 255],   // elbow
  3: [255, 0, 0],     // valve
  4: [0, 255, 0],     // tank
  5: [255, 255, 0],   // structural
  6: [255, 128, 0],   // clutter
}

export const CLASS_NAMES: Record<number, string> = {
  0: 'background',
  1: 'pipe',
  2: 'elbow',
  3: 'valve',
  4: 'tank',
  5: 'structural',
  6: 'clutter',
}

interface PointCloudState {
  // Data
  points: Float32Array | null
  colors: Uint8Array | null
  originalColors: Uint8Array | null
  labels: Int32Array | null
  instanceIds: Int32Array | null
  numPoints: number

  // Scene info
  sceneName: string | null
  loading: boolean
  error: string | null

  // Selection
  selectedIndices: Set<number>

  // Actions
  load: (path: string) => Promise<void>
  save: () => Promise<void>
  setLabels: (indices: number[], classId: number) => void
  setSelection: (indices: Set<number>) => void
  clearSelection: () => void
  updateColorsFromLabels: () => void
}

export const usePointCloudStore = create<PointCloudState>((set, get) => ({
  points: null,
  colors: null,
  originalColors: null,
  labels: null,
  instanceIds: null,
  numPoints: 0,
  sceneName: null,
  loading: false,
  error: null,
  selectedIndices: new Set(),

  load: async (path: string) => {
    set({ loading: true, error: null })
    try {
      const data = await loadPointCloud(path)
      const points = base64ToFloat32Array(data.points)
      const colors = data.colors ? base64ToUint8Array(data.colors) : new Uint8Array(data.num_points * 3).fill(128)
      const labels = base64ToInt32Array(data.labels)

      // Extract scene name from path
      const sceneName = path.split('/')[0]

      set({
        points,
        colors: new Uint8Array(colors),
        originalColors: new Uint8Array(colors),
        labels,
        instanceIds: new Int32Array(data.num_points),
        numPoints: data.num_points,
        sceneName,
        loading: false,
        selectedIndices: new Set(),
      })

      // Apply label colors if any labels exist
      get().updateColorsFromLabels()
    } catch (e) {
      set({ loading: false, error: String(e) })
    }
  },

  save: async () => {
    const { labels, instanceIds, sceneName } = get()
    if (!labels || !instanceIds || !sceneName) {
      set({ error: 'No point cloud loaded' })
      return
    }

    set({ loading: true, error: null })
    try {
      await savePointCloud(labels, instanceIds, sceneName)
      set({ loading: false })
    } catch (e) {
      set({ loading: false, error: String(e) })
    }
  },

  setLabels: (indices: number[], classId: number) => {
    const { labels } = get()
    if (!labels) return

    for (const idx of indices) {
      labels[idx] = classId
    }

    set({ labels: new Int32Array(labels) })
    get().updateColorsFromLabels()
  },

  setSelection: (indices: Set<number>) => {
    set({ selectedIndices: indices })
  },

  clearSelection: () => {
    set({ selectedIndices: new Set() })
  },

  updateColorsFromLabels: () => {
    const { labels, originalColors, colors, numPoints } = get()
    if (!labels || !originalColors || !colors) return

    const newColors = new Uint8Array(numPoints * 3)

    for (let i = 0; i < numPoints; i++) {
      const label = labels[i]
      if (label > 0) {
        const [r, g, b] = CLASS_COLORS[label] || [128, 128, 128]
        newColors[i * 3] = r
        newColors[i * 3 + 1] = g
        newColors[i * 3 + 2] = b
      } else {
        newColors[i * 3] = originalColors[i * 3]
        newColors[i * 3 + 1] = originalColors[i * 3 + 1]
        newColors[i * 3 + 2] = originalColors[i * 3 + 2]
      }
    }

    set({ colors: newColors })
  },
}))
```

**Step 2: Commit**

```bash
git add frontend/src/store/pointCloudStore.ts
git commit -m "feat: add Zustand store for point cloud state"
```

---

### Task 3.3: Point Cloud Viewport Component

**Files:**
- Create: `frontend/src/components/Viewport.tsx`
- Modify: `frontend/src/App.tsx`

**Step 1: Create Viewport component**

```typescript
import { useRef, useMemo, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { usePointCloudStore } from '../store/pointCloudStore'

function PointCloudMesh() {
  const meshRef = useRef<THREE.Points>(null)
  const { points, colors, numPoints, selectedIndices } = usePointCloudStore()

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()

    if (points && colors) {
      geo.setAttribute('position', new THREE.BufferAttribute(points, 3))

      // Normalize colors to 0-1 range
      const normalizedColors = new Float32Array(numPoints * 3)
      for (let i = 0; i < numPoints * 3; i++) {
        normalizedColors[i] = colors[i] / 255
      }
      geo.setAttribute('color', new THREE.BufferAttribute(normalizedColors, 3))
    }

    return geo
  }, [points, colors, numPoints])

  // Update colors when selection changes
  useEffect(() => {
    if (!meshRef.current || !colors) return

    const colorAttr = meshRef.current.geometry.getAttribute('color') as THREE.BufferAttribute
    if (!colorAttr) return

    const normalizedColors = colorAttr.array as Float32Array

    for (let i = 0; i < numPoints; i++) {
      if (selectedIndices.has(i)) {
        // Highlight selected points in white
        normalizedColors[i * 3] = 1
        normalizedColors[i * 3 + 1] = 1
        normalizedColors[i * 3 + 2] = 1
      } else {
        normalizedColors[i * 3] = colors[i * 3] / 255
        normalizedColors[i * 3 + 1] = colors[i * 3 + 1] / 255
        normalizedColors[i * 3 + 2] = colors[i * 3 + 2] / 255
      }
    }

    colorAttr.needsUpdate = true
  }, [selectedIndices, colors, numPoints])

  if (!points) return null

  return (
    <points ref={meshRef} geometry={geometry}>
      <pointsMaterial
        size={0.02}
        vertexColors
        sizeAttenuation
      />
    </points>
  )
}

function CameraController() {
  const { camera } = useThree()
  const { points } = usePointCloudStore()

  useEffect(() => {
    if (!points || points.length === 0) return

    // Calculate bounding box and center camera
    const positions = points
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

    for (let i = 0; i < positions.length; i += 3) {
      minX = Math.min(minX, positions[i])
      maxX = Math.max(maxX, positions[i])
      minY = Math.min(minY, positions[i + 1])
      maxY = Math.max(maxY, positions[i + 1])
      minZ = Math.min(minZ, positions[i + 2])
      maxZ = Math.max(maxZ, positions[i + 2])
    }

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const centerZ = (minZ + maxZ) / 2

    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ)

    camera.position.set(centerX + size, centerY + size, centerZ + size)
    camera.lookAt(centerX, centerY, centerZ)
  }, [points, camera])

  return null
}

export function Viewport() {
  return (
    <Canvas
      camera={{ position: [10, 10, 10], fov: 50, near: 0.1, far: 10000 }}
      style={{ background: '#1a1a2e' }}
    >
      <ambientLight intensity={0.5} />
      <CameraController />
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        mouseButtons={{
          LEFT: undefined, // Reserved for selection
          MIDDLE: THREE.MOUSE.PAN,
          RIGHT: THREE.MOUSE.ROTATE,
        }}
      />
      <PointCloudMesh />
    </Canvas>
  )
}
```

**Step 2: Update App.tsx**

```typescript
import { Viewport } from './components/Viewport'
import { usePointCloudStore } from './store/pointCloudStore'
import { useEffect } from 'react'

function App() {
  const { load, loading, error, numPoints, sceneName } = usePointCloudStore()

  // For testing: load a sample file on mount
  useEffect(() => {
    // Uncomment when you have test data:
    // load('test_scene/source.glb')
  }, [])

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '8px', background: '#2d2d44', color: 'white', display: 'flex', gap: '16px' }}>
        <span>Point Cloud Labeler</span>
        {loading && <span>Loading...</span>}
        {error && <span style={{ color: 'red' }}>{error}</span>}
        {sceneName && <span>Scene: {sceneName} ({numPoints.toLocaleString()} points)</span>}
      </div>
      <div style={{ flex: 1 }}>
        <Viewport />
      </div>
    </div>
  )
}

export default App
```

**Step 3: Test with cube placeholder**

```bash
cd frontend && npm run dev
# Open http://localhost:5173 - should see empty viewport with header
```

**Step 4: Commit**

```bash
git add frontend/src/components/Viewport.tsx frontend/src/App.tsx
git commit -m "feat: add Three.js point cloud viewport"
```

---

## Phase 4: UI Panels

### Task 4.1: File Panel

**Files:**
- Create: `frontend/src/components/FilePanel.tsx`

**Step 1: Create FilePanel component**

```typescript
import { useEffect, useState } from 'react'
import { listFiles, SceneInfo } from '../lib/api'
import { usePointCloudStore } from '../store/pointCloudStore'

export function FilePanel() {
  const [scenes, setScenes] = useState<SceneInfo[]>([])
  const [loadingList, setLoadingList] = useState(false)
  const { load, save, loading, sceneName } = usePointCloudStore()

  useEffect(() => {
    fetchScenes()
  }, [])

  const fetchScenes = async () => {
    setLoadingList(true)
    try {
      const data = await listFiles()
      setScenes(data)
    } catch (e) {
      console.error('Failed to fetch scenes:', e)
    }
    setLoadingList(false)
  }

  const handleLoad = (scene: SceneInfo) => {
    const path = scene.has_ground_truth
      ? `${scene.name}/ground_truth.ply`
      : `${scene.name}/source.${scene.source_type}`
    load(path)
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.title}>Files</h3>

      <div style={styles.actions}>
        <button onClick={fetchScenes} disabled={loadingList} style={styles.button}>
          Refresh
        </button>
        <button
          onClick={() => save()}
          disabled={loading || !sceneName}
          style={styles.button}
        >
          Save
        </button>
      </div>

      <div style={styles.list}>
        {loadingList ? (
          <div>Loading...</div>
        ) : scenes.length === 0 ? (
          <div style={styles.empty}>No scenes found in data/real/</div>
        ) : (
          scenes.map((scene) => (
            <div
              key={scene.name}
              onClick={() => handleLoad(scene)}
              style={{
                ...styles.item,
                background: scene.name === sceneName ? '#4a4a6a' : undefined,
              }}
            >
              <div style={styles.sceneName}>{scene.name}</div>
              <div style={styles.badges}>
                {scene.has_source && (
                  <span style={styles.badge}>{scene.source_type?.toUpperCase()}</span>
                )}
                {scene.has_ground_truth && (
                  <span style={{ ...styles.badge, background: '#2d8a2d' }}>GT</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 200,
    background: '#2d2d44',
    color: 'white',
    padding: 12,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  title: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
  },
  actions: {
    display: 'flex',
    gap: 8,
  },
  button: {
    flex: 1,
    padding: '6px 12px',
    background: '#4a4a6a',
    border: 'none',
    borderRadius: 4,
    color: 'white',
    cursor: 'pointer',
  },
  list: {
    flex: 1,
    overflow: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  item: {
    padding: 8,
    borderRadius: 4,
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sceneName: {
    fontSize: 13,
  },
  badges: {
    display: 'flex',
    gap: 4,
  },
  badge: {
    fontSize: 10,
    padding: '2px 6px',
    background: '#666',
    borderRadius: 3,
  },
  empty: {
    fontSize: 12,
    color: '#888',
    textAlign: 'center',
    padding: 16,
  },
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/FilePanel.tsx
git commit -m "feat: add file panel for scene browsing"
```

---

### Task 4.2: Class Panel

**Files:**
- Create: `frontend/src/components/ClassPanel.tsx`

**Step 1: Create ClassPanel component**

```typescript
import { usePointCloudStore, CLASS_COLORS, CLASS_NAMES } from '../store/pointCloudStore'

export function ClassPanel() {
  const { labels, selectedIndices, setLabels, clearSelection } = usePointCloudStore()

  const handleAssignClass = (classId: number) => {
    if (selectedIndices.size === 0) return
    setLabels(Array.from(selectedIndices), classId)
    clearSelection()
  }

  // Count labels
  const labelCounts: Record<number, number> = {}
  if (labels) {
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i]
      labelCounts[label] = (labelCounts[label] || 0) + 1
    }
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.title}>Classes</h3>

      {selectedIndices.size > 0 && (
        <div style={styles.selection}>
          {selectedIndices.size.toLocaleString()} selected
        </div>
      )}

      <div style={styles.list}>
        {Object.entries(CLASS_NAMES).map(([id, name]) => {
          const classId = Number(id)
          const [r, g, b] = CLASS_COLORS[classId]
          const count = labelCounts[classId] || 0

          return (
            <div
              key={id}
              onClick={() => handleAssignClass(classId)}
              style={styles.item}
            >
              <div
                style={{
                  ...styles.colorBox,
                  background: `rgb(${r}, ${g}, ${b})`,
                }}
              />
              <div style={styles.info}>
                <div style={styles.name}>
                  <span style={styles.key}>{classId}</span> {name}
                </div>
                <div style={styles.count}>{count.toLocaleString()}</div>
              </div>
            </div>
          )
        })}
      </div>

      <div style={styles.hint}>
        Press 0-6 to assign class to selection
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 180,
    background: '#2d2d44',
    color: 'white',
    padding: 12,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  title: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
  },
  selection: {
    padding: '6px 10px',
    background: '#4a6a4a',
    borderRadius: 4,
    fontSize: 12,
    textAlign: 'center',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: 6,
    borderRadius: 4,
    cursor: 'pointer',
  },
  colorBox: {
    width: 16,
    height: 16,
    borderRadius: 3,
    flexShrink: 0,
  },
  info: {
    flex: 1,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  name: {
    fontSize: 12,
  },
  key: {
    display: 'inline-block',
    width: 14,
    height: 14,
    lineHeight: '14px',
    textAlign: 'center',
    background: '#555',
    borderRadius: 2,
    marginRight: 4,
    fontSize: 10,
  },
  count: {
    fontSize: 11,
    color: '#888',
  },
  hint: {
    fontSize: 11,
    color: '#666',
    textAlign: 'center',
  },
}
```

**Step 2: Commit**

```bash
git add frontend/src/components/ClassPanel.tsx
git commit -m "feat: add class panel for label assignment"
```

---

### Task 4.3: Mode Toolbar

**Files:**
- Create: `frontend/src/components/ModeToolbar.tsx`
- Create: `frontend/src/store/selectionStore.ts`

**Step 1: Create selection store**

```typescript
import { create } from 'zustand'

export type SelectionMode = 'box' | 'lasso' | 'sphere' | 'geometric' | 'supervoxel'

interface SelectionState {
  mode: SelectionMode
  setMode: (mode: SelectionMode) => void
}

export const useSelectionStore = create<SelectionState>((set) => ({
  mode: 'box',
  setMode: (mode) => set({ mode }),
}))
```

**Step 2: Create ModeToolbar component**

```typescript
import { useSelectionStore, SelectionMode } from '../store/selectionStore'

const MODES: { id: SelectionMode; key: string; label: string; icon: string }[] = [
  { id: 'box', key: 'B', label: 'Box Select', icon: '▢' },
  { id: 'lasso', key: 'L', label: 'Lasso Select', icon: '◯' },
  { id: 'sphere', key: 'S', label: 'Sphere Select', icon: '●' },
  { id: 'geometric', key: 'G', label: 'Geometric Cluster', icon: '◈' },
  { id: 'supervoxel', key: 'V', label: 'Supervoxel', icon: '⬡' },
]

export function ModeToolbar() {
  const { mode, setMode } = useSelectionStore()

  return (
    <div style={styles.toolbar}>
      {MODES.map((m) => (
        <button
          key={m.id}
          onClick={() => setMode(m.id)}
          style={{
            ...styles.button,
            background: mode === m.id ? '#6a6a8a' : '#4a4a6a',
          }}
          title={`${m.label} (${m.key})`}
        >
          <span style={styles.icon}>{m.icon}</span>
          <span style={styles.key}>{m.key}</span>
        </button>
      ))}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  toolbar: {
    position: 'absolute',
    top: 12,
    left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex',
    gap: 4,
    background: 'rgba(45, 45, 68, 0.9)',
    padding: 4,
    borderRadius: 6,
    zIndex: 100,
  },
  button: {
    width: 40,
    height: 40,
    border: 'none',
    borderRadius: 4,
    color: 'white',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
  },
  icon: {
    fontSize: 16,
  },
  key: {
    fontSize: 10,
    opacity: 0.7,
  },
}
```

**Step 3: Commit**

```bash
git add frontend/src/store/selectionStore.ts frontend/src/components/ModeToolbar.tsx
git commit -m "feat: add selection mode toolbar"
```

---

### Task 4.4: Keyboard Shortcuts Hook

**Files:**
- Create: `frontend/src/hooks/useKeyboard.ts`

**Step 1: Create keyboard hook**

```typescript
import { useEffect } from 'react'
import { useSelectionStore, SelectionMode } from '../store/selectionStore'
import { usePointCloudStore } from '../store/pointCloudStore'

const MODE_KEYS: Record<string, SelectionMode> = {
  b: 'box',
  l: 'lasso',
  s: 'sphere',
  g: 'geometric',
  v: 'supervoxel',
}

const CLASS_KEYS = ['0', '1', '2', '3', '4', '5', '6']

export function useKeyboard() {
  const { setMode } = useSelectionStore()
  const { selectedIndices, setLabels, clearSelection } = usePointCloudStore()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      const key = e.key.toLowerCase()

      // Mode switching
      if (MODE_KEYS[key]) {
        setMode(MODE_KEYS[key])
        return
      }

      // Class assignment
      if (CLASS_KEYS.includes(key) && selectedIndices.size > 0) {
        const classId = parseInt(key)
        setLabels(Array.from(selectedIndices), classId)
        clearSelection()
        return
      }

      // Clear selection
      if (key === 'escape') {
        clearSelection()
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [setMode, selectedIndices, setLabels, clearSelection])
}
```

**Step 2: Commit**

```bash
git add frontend/src/hooks/useKeyboard.ts
git commit -m "feat: add keyboard shortcuts for modes and classes"
```

---

### Task 4.5: Integrate All Components

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: Update App.tsx with full layout**

```typescript
import { Viewport } from './components/Viewport'
import { FilePanel } from './components/FilePanel'
import { ClassPanel } from './components/ClassPanel'
import { ModeToolbar } from './components/ModeToolbar'
import { useKeyboard } from './hooks/useKeyboard'
import { usePointCloudStore } from './store/pointCloudStore'

function App() {
  const { loading, error, numPoints, sceneName } = usePointCloudStore()

  useKeyboard()

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>Point Cloud Labeler</span>
        {loading && <span style={styles.status}>Loading...</span>}
        {error && <span style={styles.error}>{error}</span>}
        {sceneName && (
          <span style={styles.status}>
            {sceneName} ({numPoints.toLocaleString()} points)
          </span>
        )}
      </div>

      {/* Main content */}
      <div style={styles.main}>
        <FilePanel />

        <div style={styles.viewport}>
          <ModeToolbar />
          <Viewport />
        </div>

        <ClassPanel />
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '100vw',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: '#1a1a2e',
  },
  header: {
    padding: '8px 16px',
    background: '#2d2d44',
    color: 'white',
    display: 'flex',
    gap: 16,
    alignItems: 'center',
  },
  title: {
    fontWeight: 600,
  },
  status: {
    fontSize: 13,
    color: '#aaa',
  },
  error: {
    fontSize: 13,
    color: '#ff6b6b',
  },
  main: {
    flex: 1,
    display: 'flex',
    overflow: 'hidden',
  },
  viewport: {
    flex: 1,
    position: 'relative',
  },
}

export default App
```

**Step 2: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: integrate all UI components"
```

---

## Phase 5: Box Selection

### Task 5.1: Box Selection Logic

**Files:**
- Create: `frontend/src/hooks/useBoxSelection.ts`
- Modify: `frontend/src/components/Viewport.tsx`

**Step 1: Create box selection hook**

```typescript
import { useState, useCallback, useRef } from 'react'
import * as THREE from 'three'
import { useThree } from '@react-three/fiber'
import { usePointCloudStore } from '../store/pointCloudStore'

interface BoxState {
  start: { x: number; y: number } | null
  end: { x: number; y: number } | null
  active: boolean
}

export function useBoxSelection() {
  const [box, setBox] = useState<BoxState>({ start: null, end: null, active: false })
  const { camera, size } = useThree()
  const { points, numPoints, selectedIndices, setSelection } = usePointCloudStore()

  const frustum = useRef(new THREE.Frustum())
  const projScreenMatrix = useRef(new THREE.Matrix4())

  const startSelection = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return // Left click only
    const rect = (e.target as HTMLElement).getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setBox({ start: { x, y }, end: { x, y }, active: true })
  }, [])

  const updateSelection = useCallback((e: React.MouseEvent) => {
    if (!box.active || !box.start) return
    const rect = (e.target as HTMLElement).getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setBox(prev => ({ ...prev, end: { x, y } }))
  }, [box.active, box.start])

  const endSelection = useCallback((e: React.MouseEvent, shiftKey: boolean, ctrlKey: boolean) => {
    if (!box.active || !box.start || !box.end || !points) {
      setBox({ start: null, end: null, active: false })
      return
    }

    // Convert screen coords to NDC
    const toNDC = (x: number, y: number) => ({
      x: (x / size.width) * 2 - 1,
      y: -(y / size.height) * 2 + 1,
    })

    const startNDC = toNDC(box.start.x, box.start.y)
    const endNDC = toNDC(box.end.x, box.end.y)

    const minX = Math.min(startNDC.x, endNDC.x)
    const maxX = Math.max(startNDC.x, endNDC.x)
    const minY = Math.min(startNDC.y, endNDC.y)
    const maxY = Math.max(startNDC.y, endNDC.y)

    // Project points and check if in box
    const newSelection = new Set<number>(shiftKey ? selectedIndices : [])
    const point = new THREE.Vector3()

    projScreenMatrix.current.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    )

    for (let i = 0; i < numPoints; i++) {
      point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
      point.applyMatrix4(projScreenMatrix.current)

      // Check if point is in front of camera and within box
      if (point.z < 1 && point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY) {
        if (ctrlKey) {
          newSelection.delete(i)
        } else {
          newSelection.add(i)
        }
      }
    }

    setSelection(newSelection)
    setBox({ start: null, end: null, active: false })
  }, [box, points, numPoints, camera, size, selectedIndices, setSelection])

  return {
    box,
    startSelection,
    updateSelection,
    endSelection,
  }
}
```

**Step 2: Update Viewport.tsx to include selection**

```typescript
import { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { usePointCloudStore } from '../store/pointCloudStore'
import { useSelectionStore } from '../store/selectionStore'

function PointCloudMesh() {
  const meshRef = useRef<THREE.Points>(null)
  const { points, colors, numPoints, selectedIndices } = usePointCloudStore()

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()

    if (points && colors) {
      geo.setAttribute('position', new THREE.BufferAttribute(points, 3))

      const normalizedColors = new Float32Array(numPoints * 3)
      for (let i = 0; i < numPoints * 3; i++) {
        normalizedColors[i] = colors[i] / 255
      }
      geo.setAttribute('color', new THREE.BufferAttribute(normalizedColors, 3))
    }

    return geo
  }, [points, colors, numPoints])

  useEffect(() => {
    if (!meshRef.current || !colors) return

    const colorAttr = meshRef.current.geometry.getAttribute('color') as THREE.BufferAttribute
    if (!colorAttr) return

    const normalizedColors = colorAttr.array as Float32Array

    for (let i = 0; i < numPoints; i++) {
      if (selectedIndices.has(i)) {
        normalizedColors[i * 3] = 1
        normalizedColors[i * 3 + 1] = 1
        normalizedColors[i * 3 + 2] = 1
      } else {
        normalizedColors[i * 3] = colors[i * 3] / 255
        normalizedColors[i * 3 + 1] = colors[i * 3 + 1] / 255
        normalizedColors[i * 3 + 2] = colors[i * 3 + 2] / 255
      }
    }

    colorAttr.needsUpdate = true
  }, [selectedIndices, colors, numPoints])

  if (!points) return null

  return (
    <points ref={meshRef} geometry={geometry}>
      <pointsMaterial size={0.02} vertexColors sizeAttenuation />
    </points>
  )
}

function CameraController() {
  const { camera } = useThree()
  const { points } = usePointCloudStore()

  useEffect(() => {
    if (!points || points.length === 0) return

    const positions = points
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

    for (let i = 0; i < positions.length; i += 3) {
      minX = Math.min(minX, positions[i])
      maxX = Math.max(maxX, positions[i])
      minY = Math.min(minY, positions[i + 1])
      maxY = Math.max(maxY, positions[i + 1])
      minZ = Math.min(minZ, positions[i + 2])
      maxZ = Math.max(maxZ, positions[i + 2])
    }

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const centerZ = (minZ + maxZ) / 2
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ)

    camera.position.set(centerX + size, centerY + size, centerZ + size)
    camera.lookAt(centerX, centerY, centerZ)
  }, [points, camera])

  return null
}

function BoxSelectionOverlay() {
  const { mode } = useSelectionStore()
  const { points, numPoints, selectedIndices, setSelection } = usePointCloudStore()
  const { camera, size } = useThree()

  const [box, setBox] = useState<{
    start: { x: number; y: number } | null
    end: { x: number; y: number } | null
  }>({ start: null, end: null })
  const [isDragging, setIsDragging] = useState(false)

  const handlePointerDown = (e: THREE.Event) => {
    if (mode !== 'box' || e.button !== 0) return
    e.stopPropagation()
    setIsDragging(true)
    setBox({ start: { x: e.clientX, y: e.clientY }, end: { x: e.clientX, y: e.clientY } })
  }

  const handlePointerMove = (e: THREE.Event) => {
    if (!isDragging || !box.start) return
    setBox(prev => ({ ...prev, end: { x: e.clientX, y: e.clientY } }))
  }

  const handlePointerUp = (e: THREE.Event) => {
    if (!isDragging || !box.start || !box.end || !points) {
      setIsDragging(false)
      setBox({ start: null, end: null })
      return
    }

    const rect = (e.target as HTMLElement).closest('canvas')?.getBoundingClientRect()
    if (!rect) return

    const toNDC = (x: number, y: number) => ({
      x: ((x - rect.left) / rect.width) * 2 - 1,
      y: -((y - rect.top) / rect.height) * 2 + 1,
    })

    const startNDC = toNDC(box.start.x, box.start.y)
    const endNDC = toNDC(box.end.x, box.end.y)

    const minX = Math.min(startNDC.x, endNDC.x)
    const maxX = Math.max(startNDC.x, endNDC.x)
    const minY = Math.min(startNDC.y, endNDC.y)
    const maxY = Math.max(startNDC.y, endNDC.y)

    const projScreenMatrix = new THREE.Matrix4()
    projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)

    const newSelection = new Set<number>(e.shiftKey ? selectedIndices : [])
    const point = new THREE.Vector3()

    for (let i = 0; i < numPoints; i++) {
      point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
      point.applyMatrix4(projScreenMatrix)

      if (point.z < 1 && point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY) {
        if (e.ctrlKey) {
          newSelection.delete(i)
        } else {
          newSelection.add(i)
        }
      }
    }

    setSelection(newSelection)
    setIsDragging(false)
    setBox({ start: null, end: null })
  }

  return (
    <mesh
      visible={false}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      <planeGeometry args={[10000, 10000]} />
      <meshBasicMaterial transparent opacity={0} />
    </mesh>
  )
}

function SelectionBox({ box }: { box: { start: { x: number; y: number } | null; end: { x: number; y: number } | null } }) {
  if (!box.start || !box.end) return null

  const left = Math.min(box.start.x, box.end.x)
  const top = Math.min(box.start.y, box.end.y)
  const width = Math.abs(box.end.x - box.start.x)
  const height = Math.abs(box.end.y - box.start.y)

  return (
    <div
      style={{
        position: 'absolute',
        left,
        top,
        width,
        height,
        border: '2px solid #fff',
        background: 'rgba(255, 255, 255, 0.1)',
        pointerEvents: 'none',
      }}
    />
  )
}

export function Viewport() {
  const { mode } = useSelectionStore()
  const { points, numPoints, selectedIndices, setSelection } = usePointCloudStore()
  const canvasRef = useRef<HTMLDivElement>(null)

  const [box, setBox] = useState<{
    start: { x: number; y: number } | null
    end: { x: number; y: number } | null
  }>({ start: null, end: null })
  const [isDragging, setIsDragging] = useState(false)

  const handleMouseDown = (e: React.MouseEvent) => {
    if (mode !== 'box' || e.button !== 0) return
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    setIsDragging(true)
    setBox({
      start: { x: e.clientX - rect.left, y: e.clientY - rect.top },
      end: { x: e.clientX - rect.left, y: e.clientY - rect.top },
    })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !box.start) return
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    setBox(prev => ({
      ...prev,
      end: { x: e.clientX - rect.left, y: e.clientY - rect.top },
    }))
  }

  const handleMouseUp = (e: React.MouseEvent) => {
    if (!isDragging || !box.start || !box.end || !points || !canvasRef.current) {
      setIsDragging(false)
      setBox({ start: null, end: null })
      return
    }

    // Selection logic will be handled inside Canvas
    setIsDragging(false)
    setBox({ start: null, end: null })
  }

  return (
    <div
      ref={canvasRef}
      style={{ width: '100%', height: '100%', position: 'relative' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Canvas
        camera={{ position: [10, 10, 10], fov: 50, near: 0.1, far: 10000 }}
        style={{ background: '#1a1a2e' }}
      >
        <ambientLight intensity={0.5} />
        <CameraController />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          enabled={mode !== 'box' || !isDragging}
          mouseButtons={{
            LEFT: mode === 'box' ? undefined : THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.ROTATE,
          }}
        />
        <PointCloudMesh />
        <BoxSelectionHandler
          isDragging={isDragging}
          box={box}
          onComplete={(indices, shiftKey, ctrlKey) => {
            const newSelection = new Set<number>(shiftKey ? selectedIndices : [])
            for (const i of indices) {
              if (ctrlKey) {
                newSelection.delete(i)
              } else {
                newSelection.add(i)
              }
            }
            setSelection(newSelection)
          }}
        />
      </Canvas>

      {/* Selection box overlay */}
      {isDragging && box.start && box.end && (
        <div
          style={{
            position: 'absolute',
            left: Math.min(box.start.x, box.end.x),
            top: Math.min(box.start.y, box.end.y),
            width: Math.abs(box.end.x - box.start.x),
            height: Math.abs(box.end.y - box.start.y),
            border: '2px solid rgba(255, 255, 255, 0.8)',
            background: 'rgba(255, 255, 255, 0.1)',
            pointerEvents: 'none',
          }}
        />
      )}
    </div>
  )
}

function BoxSelectionHandler({
  isDragging,
  box,
  onComplete,
}: {
  isDragging: boolean
  box: { start: { x: number; y: number } | null; end: { x: number; y: number } | null }
  onComplete: (indices: number[], shiftKey: boolean, ctrlKey: boolean) => void
}) {
  const { camera, size } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const prevDragging = useRef(isDragging)

  useEffect(() => {
    // Detect end of drag
    if (prevDragging.current && !isDragging && box.start && box.end && points) {
      const toNDC = (x: number, y: number) => ({
        x: (x / size.width) * 2 - 1,
        y: -(y / size.height) * 2 + 1,
      })

      const startNDC = toNDC(box.start.x, box.start.y)
      const endNDC = toNDC(box.end.x, box.end.y)

      const minX = Math.min(startNDC.x, endNDC.x)
      const maxX = Math.max(startNDC.x, endNDC.x)
      const minY = Math.min(startNDC.y, endNDC.y)
      const maxY = Math.max(startNDC.y, endNDC.y)

      const projScreenMatrix = new THREE.Matrix4()
      projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)

      const selectedIndices: number[] = []
      const point = new THREE.Vector3()

      for (let i = 0; i < numPoints; i++) {
        point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
        point.applyMatrix4(projScreenMatrix)

        if (point.z < 1 && point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY) {
          selectedIndices.push(i)
        }
      }

      // Get modifier keys from window
      const shiftKey = window.event instanceof KeyboardEvent ? window.event.shiftKey : false
      const ctrlKey = window.event instanceof KeyboardEvent ? window.event.ctrlKey : false

      onComplete(selectedIndices, shiftKey, ctrlKey)
    }
    prevDragging.current = isDragging
  }, [isDragging, box, points, numPoints, camera, size, onComplete])

  return null
}
```

**Step 3: Commit**

```bash
git add frontend/src/hooks/useBoxSelection.ts frontend/src/components/Viewport.tsx
git commit -m "feat: add box selection mode"
```

---

## Phase 6: Backend Advanced Features

### Task 6.1: Supervoxel Computation

**Files:**
- Create: `backend/supervoxels.py`
- Modify: `backend/main.py`

**Step 1: Create supervoxels module**

```python
import numpy as np
import open3d as o3d
from typing import Tuple

def compute_supervoxels(
    points: np.ndarray,
    resolution: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute supervoxels using voxel downsampling and clustering.

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
```

**Step 2: Add supervoxel endpoint to main.py**

Add imports and endpoint:

```python
from supervoxels import compute_supervoxels

class SupervoxelRequest(BaseModel):
    resolution: float = 0.1


class SupervoxelResponse(BaseModel):
    num_supervoxels: int
    supervoxel_ids: str  # base64 encoded Int32Array
    centroids: str  # base64 encoded Float32Array


# Store supervoxels
current_supervoxels: tuple[np.ndarray, np.ndarray] | None = None


@app.post("/compute-supervoxels", response_model=SupervoxelResponse)
async def compute_supervoxels_endpoint(req: SupervoxelRequest):
    global current_pc, current_supervoxels

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    sv_ids, centroids = compute_supervoxels(current_pc.points, req.resolution)
    current_supervoxels = (sv_ids, centroids)

    return SupervoxelResponse(
        num_supervoxels=len(centroids),
        supervoxel_ids=base64.b64encode(sv_ids.tobytes()).decode(),
        centroids=base64.b64encode(centroids.tobytes()).decode(),
    )
```

**Step 3: Commit**

```bash
git add backend/supervoxels.py backend/main.py
git commit -m "feat: add supervoxel computation endpoint"
```

---

### Task 6.2: Geometric Clustering

**Files:**
- Create: `backend/clustering.py`
- Modify: `backend/main.py`

**Step 1: Create clustering module**

```python
import numpy as np
from scipy.spatial import KDTree
from typing import List

def region_grow(
    points: np.ndarray,
    seed_index: int,
    normal_threshold_deg: float = 15.0,
    distance_threshold: float = 0.05,
    max_points: int = 50000,
    normals: np.ndarray | None = None,
) -> List[int]:
    """
    Region growing from a seed point based on normal similarity and distance.

    Args:
        points: Nx3 point array
        seed_index: Starting point index
        normal_threshold_deg: Max angle between normals (degrees)
        distance_threshold: Max distance between neighboring points
        max_points: Safety limit
        normals: Nx3 normal array (computed if not provided)

    Returns:
        List of point indices in the grown region
    """
    n_points = len(points)

    # Compute normals if not provided
    if normals is None:
        normals = estimate_normals(points)

    # Build KD-tree for neighbor search
    tree = KDTree(points)

    # Region growing
    normal_threshold = np.cos(np.radians(normal_threshold_deg))

    visited = np.zeros(n_points, dtype=bool)
    region = [seed_index]
    visited[seed_index] = True

    seed_normal = normals[seed_index]

    queue = [seed_index]

    while queue and len(region) < max_points:
        current = queue.pop(0)
        current_point = points[current]

        # Find neighbors within distance threshold
        neighbor_indices = tree.query_ball_point(current_point, distance_threshold)

        for neighbor in neighbor_indices:
            if visited[neighbor]:
                continue

            # Check normal similarity with seed
            neighbor_normal = normals[neighbor]
            dot_product = np.abs(np.dot(seed_normal, neighbor_normal))

            if dot_product >= normal_threshold:
                visited[neighbor] = True
                region.append(neighbor)
                queue.append(neighbor)

    return region


def estimate_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """Estimate normals using PCA on local neighborhoods."""
    tree = KDTree(points)
    normals = np.zeros_like(points)

    for i in range(len(points)):
        # Find k nearest neighbors
        _, indices = tree.query(points[i], k=min(k, len(points)))

        # PCA to find normal
        neighbors = points[indices]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Normal is eigenvector with smallest eigenvalue
        normals[i] = eigenvectors[:, 0]

    return normals
```

**Step 2: Add clustering endpoint to main.py**

```python
from clustering import region_grow

class ClusterRequest(BaseModel):
    seed_index: int
    normal_threshold: float = 15.0
    distance_threshold: float = 0.05
    max_points: int = 50000


class ClusterResponse(BaseModel):
    indices: str  # base64 encoded Int32Array
    num_points: int


@app.post("/cluster", response_model=ClusterResponse)
async def compute_cluster(req: ClusterRequest):
    global current_pc

    if current_pc is None:
        raise HTTPException(400, "No point cloud loaded")

    if req.seed_index < 0 or req.seed_index >= len(current_pc):
        raise HTTPException(400, f"Invalid seed index: {req.seed_index}")

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
```

**Step 3: Commit**

```bash
git add backend/clustering.py backend/main.py
git commit -m "feat: add geometric clustering endpoint"
```

---

## Phase 7: Frontend Advanced Selection Modes

### Task 7.1: Sphere Selection

**Files:**
- Modify: `frontend/src/components/Viewport.tsx`
- Create: `frontend/src/hooks/useSphereSelection.ts`

**Step 1: Add sphere selection to Viewport**

This task adds click-and-drag sphere selection. Create `useSphereSelection.ts`:

```typescript
import { useCallback, useRef, useState } from 'react'
import * as THREE from 'three'
import { useThree } from '@react-three/fiber'
import { usePointCloudStore } from '../store/pointCloudStore'

export function useSphereSelection() {
  const { camera, raycaster, scene } = useThree()
  const { points, numPoints, selectedIndices, setSelection } = usePointCloudStore()

  const [sphereCenter, setSphereCenter] = useState<THREE.Vector3 | null>(null)
  const [sphereRadius, setSphereRadius] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  const findClickedPoint = useCallback((event: MouseEvent, canvas: HTMLCanvasElement) => {
    if (!points) return null

    const rect = canvas.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    )

    raycaster.setFromCamera(mouse, camera)

    // Find closest point to ray
    let closestDist = Infinity
    let closestIdx = -1
    const ray = raycaster.ray
    const point = new THREE.Vector3()

    for (let i = 0; i < numPoints; i++) {
      point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
      const dist = ray.distanceToPoint(point)

      if (dist < closestDist && dist < 0.5) { // 0.5 threshold
        closestDist = dist
        closestIdx = i
      }
    }

    if (closestIdx >= 0) {
      return new THREE.Vector3(
        points[closestIdx * 3],
        points[closestIdx * 3 + 1],
        points[closestIdx * 3 + 2]
      )
    }

    return null
  }, [points, numPoints, camera, raycaster])

  const selectPointsInSphere = useCallback((center: THREE.Vector3, radius: number, shiftKey: boolean, ctrlKey: boolean) => {
    if (!points) return

    const newSelection = new Set<number>(shiftKey ? selectedIndices : [])
    const radiusSq = radius * radius

    for (let i = 0; i < numPoints; i++) {
      const dx = points[i * 3] - center.x
      const dy = points[i * 3 + 1] - center.y
      const dz = points[i * 3 + 2] - center.z
      const distSq = dx * dx + dy * dy + dz * dz

      if (distSq <= radiusSq) {
        if (ctrlKey) {
          newSelection.delete(i)
        } else {
          newSelection.add(i)
        }
      }
    }

    setSelection(newSelection)
  }, [points, numPoints, selectedIndices, setSelection])

  return {
    sphereCenter,
    sphereRadius,
    isDragging,
    setSphereCenter,
    setSphereRadius,
    setIsDragging,
    findClickedPoint,
    selectPointsInSphere,
  }
}
```

**Step 2: Commit**

```bash
git add frontend/src/hooks/useSphereSelection.ts
git commit -m "feat: add sphere selection mode"
```

---

### Task 7.2: Supervoxel Selection (Frontend)

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/store/pointCloudStore.ts`

**Step 1: Add supervoxel API call**

Add to `api.ts`:

```typescript
export interface SupervoxelResponse {
  num_supervoxels: number
  supervoxel_ids: string
  centroids: string
}

export async function computeSupervoxels(resolution = 0.1): Promise<SupervoxelResponse> {
  const res = await fetch(`${API_BASE}/compute-supervoxels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resolution }),
  })
  if (!res.ok) throw new Error(`Supervoxel computation failed: ${res.statusText}`)
  return res.json()
}
```

**Step 2: Add supervoxel state to store**

Add to `pointCloudStore.ts`:

```typescript
// Add to state interface
supervoxelIds: Int32Array | null

// Add to initial state
supervoxelIds: null,

// Add action
computeSupervoxels: async (resolution: number = 0.1) => {
  set({ loading: true, error: null })
  try {
    const data = await computeSupervoxels(resolution)
    const supervoxelIds = base64ToInt32Array(data.supervoxel_ids)
    set({ supervoxelIds, loading: false })
  } catch (e) {
    set({ loading: false, error: String(e) })
  }
},

selectSupervoxel: (pointIndex: number, shiftKey: boolean, ctrlKey: boolean) => {
  const { supervoxelIds, selectedIndices, numPoints, setSelection } = get()
  if (!supervoxelIds) return

  const targetSvId = supervoxelIds[pointIndex]
  const newSelection = new Set<number>(shiftKey ? selectedIndices : [])

  for (let i = 0; i < numPoints; i++) {
    if (supervoxelIds[i] === targetSvId) {
      if (ctrlKey) {
        newSelection.delete(i)
      } else {
        newSelection.add(i)
      }
    }
  }

  setSelection(newSelection)
},
```

**Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/store/pointCloudStore.ts
git commit -m "feat: add supervoxel selection support"
```

---

### Task 7.3: Geometric Cluster Selection (Frontend)

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/store/pointCloudStore.ts`

**Step 1: Add cluster API call**

Add to `api.ts`:

```typescript
export interface ClusterResponse {
  indices: string
  num_points: number
}

export async function computeCluster(
  seedIndex: number,
  normalThreshold = 15,
  distanceThreshold = 0.05,
  maxPoints = 50000
): Promise<ClusterResponse> {
  const res = await fetch(`${API_BASE}/cluster`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      seed_index: seedIndex,
      normal_threshold: normalThreshold,
      distance_threshold: distanceThreshold,
      max_points: maxPoints,
    }),
  })
  if (!res.ok) throw new Error(`Cluster computation failed: ${res.statusText}`)
  return res.json()
}
```

**Step 2: Add cluster action to store**

Add to `pointCloudStore.ts`:

```typescript
selectGeometricCluster: async (seedIndex: number, shiftKey: boolean, ctrlKey: boolean) => {
  const { selectedIndices, setSelection } = get()
  set({ loading: true, error: null })
  try {
    const data = await computeCluster(seedIndex)
    const indices = base64ToInt32Array(data.indices)

    const newSelection = new Set<number>(shiftKey ? selectedIndices : [])
    for (const idx of indices) {
      if (ctrlKey) {
        newSelection.delete(idx)
      } else {
        newSelection.add(idx)
      }
    }

    setSelection(newSelection)
    set({ loading: false })
  } catch (e) {
    set({ loading: false, error: String(e) })
  }
},
```

**Step 3: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/store/pointCloudStore.ts
git commit -m "feat: add geometric cluster selection"
```

---

## Phase 8: Polish & Integration

### Task 8.1: Confirmation Dialog

**Files:**
- Create: `frontend/src/components/ConfirmDialog.tsx`

**Step 1: Create dialog component**

```typescript
import { useEffect, useState } from 'react'
import { usePointCloudStore, CLASS_NAMES, CLASS_COLORS } from '../store/pointCloudStore'

export function ConfirmDialog() {
  const { selectedIndices, setLabels, clearSelection } = usePointCloudStore()
  const [pendingClass, setPendingClass] = useState<number | null>(null)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return

      const key = e.key

      // Check if it's a class key (0-6)
      if (/^[0-6]$/.test(key) && selectedIndices.size > 0) {
        setPendingClass(parseInt(key))
        return
      }

      // Confirm with Enter
      if (key === 'Enter' && pendingClass !== null) {
        setLabels(Array.from(selectedIndices), pendingClass)
        clearSelection()
        setPendingClass(null)
        return
      }

      // Cancel with Escape
      if (key === 'Escape') {
        setPendingClass(null)
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedIndices, pendingClass, setLabels, clearSelection])

  if (pendingClass === null || selectedIndices.size === 0) return null

  const [r, g, b] = CLASS_COLORS[pendingClass]
  const className = CLASS_NAMES[pendingClass]

  return (
    <div style={styles.overlay}>
      <div style={styles.dialog}>
        <div style={styles.header}>
          <div
            style={{
              ...styles.colorBox,
              background: `rgb(${r}, ${g}, ${b})`,
            }}
          />
          <span>Assign '{className}' to {selectedIndices.size.toLocaleString()} points?</span>
        </div>
        <div style={styles.actions}>
          <span style={styles.hint}>[Enter] Accept</span>
          <span style={styles.hint}>[Esc] Cancel</span>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'absolute',
    bottom: 20,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 200,
  },
  dialog: {
    background: 'rgba(45, 45, 68, 0.95)',
    borderRadius: 8,
    padding: '12px 20px',
    color: 'white',
    boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  colorBox: {
    width: 20,
    height: 20,
    borderRadius: 4,
  },
  actions: {
    display: 'flex',
    gap: 16,
    justifyContent: 'center',
  },
  hint: {
    fontSize: 12,
    color: '#888',
  },
}
```

**Step 2: Add to App.tsx**

Import and add `<ConfirmDialog />` inside the viewport div.

**Step 3: Commit**

```bash
git add frontend/src/components/ConfirmDialog.tsx frontend/src/App.tsx
git commit -m "feat: add confirmation dialog for label assignment"
```

---

### Task 8.2: Auto-Save Session

**Files:**
- Modify: `frontend/src/store/pointCloudStore.ts`

**Step 1: Add auto-save functionality**

```typescript
// Add session save/load to store
saveSession: () => {
  const { sceneName, labels, instanceIds } = get()
  if (!sceneName || !labels || !instanceIds) return

  const session = {
    source_file: sceneName,
    labels: Array.from(labels),
    instance_ids: Array.from(instanceIds),
    timestamp: new Date().toISOString(),
  }

  localStorage.setItem(`labeling-session-${sceneName}`, JSON.stringify(session))
},

loadSession: () => {
  const { sceneName, labels } = get()
  if (!sceneName || !labels) return false

  const saved = localStorage.getItem(`labeling-session-${sceneName}`)
  if (!saved) return false

  try {
    const session = JSON.parse(saved)
    if (session.labels && session.labels.length === labels.length) {
      const newLabels = new Int32Array(session.labels)
      const newInstanceIds = new Int32Array(session.instance_ids || new Array(labels.length).fill(0))
      set({ labels: newLabels, instanceIds: newInstanceIds })
      get().updateColorsFromLabels()
      return true
    }
  } catch (e) {
    console.error('Failed to load session:', e)
  }
  return false
},
```

**Step 2: Add auto-save interval in App.tsx**

```typescript
useEffect(() => {
  const interval = setInterval(() => {
    usePointCloudStore.getState().saveSession()
  }, 60000) // Every 60 seconds

  return () => clearInterval(interval)
}, [])
```

**Step 3: Commit**

```bash
git add frontend/src/store/pointCloudStore.ts frontend/src/App.tsx
git commit -m "feat: add auto-save session to localStorage"
```

---

### Task 8.3: Final Integration Test

**Files:**
- Create: `data/real/test_scene/` (test data)

**Step 1: Create test script**

```bash
# Create test data directory
mkdir -p data/real/test_scene

# If you have a test GLB file, copy it:
# cp /path/to/test.glb data/real/test_scene/source.glb
```

**Step 2: Start both servers**

```bash
# Terminal 1
cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
```

**Step 3: Manual test checklist**

- [ ] Load a scene from file panel
- [ ] Box select points (B mode, left-click drag)
- [ ] Assign class with number key + Enter
- [ ] Verify colors update
- [ ] Save ground truth
- [ ] Verify PLY file created

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete point cloud labeling app MVP"
```

---

## Summary

This plan covers:

1. **Project scaffolding** - Backend (FastAPI) + Frontend (Vite/React/Three.js)
2. **Backend core** - Load GLB/PLY, save PLY, file listing
3. **Frontend core** - Point cloud rendering, state management
4. **UI panels** - File browser, class panel, mode toolbar
5. **Box selection** - Click-drag frustum selection
6. **Advanced backend** - Supervoxels, geometric clustering
7. **Advanced selection** - Sphere, supervoxel, geometric modes
8. **Polish** - Confirmation dialog, auto-save

Total: ~35 bite-sized tasks across 8 phases.
