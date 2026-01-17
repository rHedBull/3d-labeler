# Point Cloud Labeling App Design

**Purpose:** Web-based tool for labeling real industrial point clouds to create ground truth training data.

**Related:** [Segmentation Visualizer Design](./2026-01-17-segmentation-visualizer-design.md) - uses the same data format for GT vs prediction comparison.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        React + Three.js Frontend                        │
│  ┌─────────────┐  ┌─────────────────────────┐  ┌─────────────────────┐  │
│  │ File Panel  │  │      3D Viewport        │  │    Class Panel      │  │
│  │             │  │      (Three.js)         │  │                     │  │
│  │ - File list │  │                         │  │ ■ background (0)    │  │
│  │ - Load/Save │  │  [B] [L] [S] [G] [V]    │  │ ■ pipe (1)          │  │
│  │ - Progress  │  │    Mode toolbar         │  │ ■ elbow (2)         │  │
│  │             │  │                         │  │ ■ valve (3)         │  │
│  │             │  │                         │  │ ■ tank (4)          │  │
│  │             │  │                         │  │ ■ structural (5)    │  │
│  │             │  │                         │  │ ■ clutter (6)       │  │
│  └─────────────┘  └─────────────────────────┘  └─────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ REST API
┌────────────────────────────────▼────────────────────────────────────────┐
│                           FastAPI Backend                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │ Load GLB/PLY     │  │ Supervoxel       │  │ Export PLY           │   │
│  │ → point cloud    │  │ computation      │  │ (training-ready)     │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │       File System       │
                    │  - GLB/PLY input        │
                    │  - ground_truth.ply out │
                    │  - .json session save   │
                    └─────────────────────────┘
```

**Stack:** Vite + React + Three.js + FastAPI (Python)

## Selection Modes

| Mode | Key | Interaction |
|------|-----|-------------|
| Box select | `B` | Click-drag rectangle, selects points in frustum |
| Lasso select | `L` | Freehand polygon, selects points inside |
| Sphere select | `S` | Click point → drag radius → selects points in sphere |
| Geometric cluster | `G` | Click point → region-grows by normal + distance |
| Supervoxel | `V` | Click point → selects pre-computed supervoxel |

**Selection modifiers:**
- `Shift` + select = add to selection
- `Ctrl` + select = remove from selection
- `Escape` = clear selection

## Class Assignment

**Keys:** `0-6` assign class to selection (with confirmation)

**Confirmation flow:**
```
1. Select points (any mode)
2. Press class key (e.g., `1` for pipe)
3. Popup: "Assign 'pipe' to 1,234 pts? [Enter] Accept [Esc] Cancel"
4. Points preview in class color
5. Enter confirms, Escape cancels (keeps selection)
```

## Aided Segmentation

### Geometric Clustering (mode `G`)

Click on a point → region grows outward based on:
- **Normal threshold:** 15° (configurable) - points with similar surface orientation
- **Distance threshold:** 0.05m (configurable) - max gap between points
- **Max points:** 50,000 (safety limit)

Good for: pipes, tank surfaces, flat structural steel

### Supervoxels (mode `V`)

Pre-computed on load or lazily on first use:
- **Voxel resolution:** 0.1m (configurable, recompute on change)
- Click selects entire supervoxel
- Shift+click adds adjacent supervoxels

Good for: fast rough labeling, then refine with other modes

## Camera Controls

| Action | Control |
|--------|---------|
| Orbit | Right-drag |
| Zoom | Scroll |
| Pan | Middle-drag |

## Backend API

```
POST /load
  Input: { path: "scene_name/source.glb" } or { path: "scene_name/ground_truth.ply" }
  - GLB: samples surface → points (configurable density)
  - PLY: loads directly, preserves existing labels
  Returns: { points: Float32Array, colors?: Uint8Array, labels?: Int32Array }

POST /compute-supervoxels
  Input: { resolution: 0.1 }
  Returns: { supervoxel_ids: Int32Array, centroids: Float32Array }

POST /save
  Input: { labels: Int32Array, instance_ids: Int32Array, scene_name: "pumping_station_01" }
  Writes: data/real/{scene_name}/ground_truth.ply
  Returns: { success: true, num_points: 500000 }

GET /files
  Returns: List of available scenes with GLB/PLY files
```

## File Management

**Auto-save:** Labels saved to `.json` sidecar every 60s
- JSON contains: `{ source_file, labels: [...], instance_ids: [...], timestamp }`
- Session restore: load source + apply saved JSON labels

**Export:**
- **Save PLY:** Training-ready `ground_truth.ply`
- **Export JSON:** Backup/transfer labels without points

---

## Shared Data Contract

> **IMPORTANT:** This format is shared with the [Segmentation Visualizer](./2026-01-17-segmentation-visualizer-design.md).
> Any changes must be synchronized between both tools.

### Directory Structure

```
data/
└── real/
    └── {scene_name}/                    # e.g., pumping_station_01
        ├── source.glb                   # Original NavVis mesh (input)
        ├── ground_truth.ply             # ← Labeling app outputs this
        └── ground_truth.glb             # Optional: labeled mesh export

output/
└── inference/
    └── {scene_name}/                    # Same scene name as data/real/
        ├── ground_truth.ply             # Copied from data/real/ for comparison
        ├── ground_truth.glb             # Optional
        ├── prediction.ply               # Model predictions
        ├── instance_segmentation.ply    # Instance predictions
        └── metadata.yaml                # Computed metrics
```

### PLY Schema

```
ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int label                       # REQUIRED: class ID (0-6)
property int instance_id                 # REQUIRED: instance ID (0 = unlabeled)
end_header
```

### Class Definitions

```yaml
# labeling_classes.yaml - MUST be identical in labeling app, training repo, and visualizer
classes:
  - { id: 0, name: background,  color: [128, 128, 128] }
  - { id: 1, name: pipe,        color: [0, 0, 255] }
  - { id: 2, name: elbow,       color: [0, 255, 255] }
  - { id: 3, name: valve,       color: [255, 0, 0] }
  - { id: 4, name: tank,        color: [0, 255, 0] }
  - { id: 5, name: structural,  color: [255, 255, 0] }
  - { id: 6, name: clutter,     color: [255, 128, 0] }
```

---

## End-to-End Workflow

```
1. LABEL (this app)
   Load: data/real/{scene}/source.glb
   Output: data/real/{scene}/ground_truth.ply

2. TRAIN (training repo)
   Input: data/real/*/ground_truth.ply
   Output: weights/{run_name}/best_model.pth

3. INFER (training repo)
   Input: data/real/{scene}/source.glb + trained model
   Output: output/inference/{scene}/prediction.ply
   Copy: ground_truth.ply → output/inference/{scene}/

4. VISUALIZE (visualizer)
   Input: output/inference/{scene}/
   Compare: ground_truth.ply vs prediction.ply
```

---

## Project Structure (Separate Repo)

```
pointcloud-labeling-app/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── point_cloud.py           # Load GLB/PLY, sampling
│   ├── supervoxels.py           # Supervoxel computation (Open3D)
│   ├── clustering.py            # Geometric region growing
│   └── requirements.txt         # fastapi, uvicorn, trimesh, open3d, plyfile
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Viewport.tsx          # Three.js canvas
│   │   │   ├── ModeToolbar.tsx       # B/L/S/G/V mode buttons
│   │   │   ├── ClassPanel.tsx        # Class buttons + point stats
│   │   │   ├── FilePanel.tsx         # File browser, load/save
│   │   │   └── ConfirmDialog.tsx     # Label assignment confirmation
│   │   ├── hooks/
│   │   │   ├── useSelection.ts       # Selection state + logic
│   │   │   ├── usePointCloud.ts      # Point cloud data state
│   │   │   └── useKeyboard.ts        # Keyboard shortcuts
│   │   └── lib/
│   │       ├── api.ts                # Backend API calls
│   │       └── three-utils.ts        # Three.js helpers
│   ├── package.json
│   └── vite.config.ts
│
├── labeling_classes.yaml            # Shared class definitions
└── README.md
```

## Running

```bash
# Terminal 1: Backend
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open `http://localhost:5173`
