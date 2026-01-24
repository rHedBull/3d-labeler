# Primitive Fitting for AI-Assisted Labeling

## Overview

Add cylinder and box primitive fitting to accelerate labeling of industrial point clouds. User defines a search region, algorithm fits geometric primitives within that region, user selects which candidates to accept.

## Workflow

### Cylinder Fit Mode (C)

1. Click to place center point on surface
2. Drag/click to set radius (preview circle)
3. Drag/click along axis to set length (preview cylinder wireframe)
4. Algorithm fits cylinders within that region using RANSAC
5. Show candidates with different colors, user clicks to accept/reject each
6. Accepted points get selected

### Box Fit Mode (X)

1. Click 3 corners of base rectangle
2. 4th corner auto-computed (perpendicular)
3. Drag to set height (preview box wireframe)
4. Algorithm fits planar surfaces / boxes within region
5. Show candidates, user picks which to accept
6. Accepted points get selected

## Algorithm Approach

### Cylinder Fitting (RANSAC-based)

1. Extract points within the user-defined cylindrical region
2. Run RANSAC cylinder fitting:
   - Sample 2 points to define axis direction
   - Sample 1 more point to define radius
   - Count inliers (points within tolerance of cylinder surface)
   - Iterate to find best fit
3. Cluster remaining points, repeat to find additional cylinders
4. Return list of fitted cylinders with their inlier point indices

### Box Fitting (plane-based)

1. Extract points within the user-defined box region
2. Run RANSAC plane fitting multiple times:
   - Find dominant plane, extract inliers
   - Find perpendicular planes in remaining points
   - Group planes that form box faces
3. Alternative: PCA-based oriented bounding box fitting
4. Return candidate boxes with their inlier points

### Libraries

- **Open3D** - `segment_plane()`, custom cylinder RANSAC
- **scipy** - least squares optimization for refinement

## Interaction Details

### Cylinder Definition

1. **Click 1** - Place center point (raycast to nearest point)
2. **Move mouse** - Preview circle perpendicular to view
3. **Click 2** - Lock radius
4. **Move mouse** - Preview cylinder extending along axis
5. **Click 3** - Lock length, trigger fitting
6. **Scroll wheel** (optional) - Rotate axis orientation

### Box Definition

1. **Click 1** - First corner (raycast to cloud)
2. **Click 2** - Second corner (defines one edge)
3. **Click 3** - Third corner (defines base plane)
4. **Move mouse** - Preview box with adjustable height
5. **Click 4** - Lock height, trigger fitting

### Candidate Selection

- Each fitted primitive shown as colored wireframe
- Inlier points highlighted in matching color
- Panel lists candidates with point counts
- Click wireframe or list item to toggle accept/reject
- "Apply" button to select all accepted points
- "Cancel" to discard

## Implementation Structure

### Backend (Python)

- `fitting.py` (new) - RANSAC cylinder and box fitting
  - `fit_cylinders(points, center, radius, length, axis, tolerance)` → candidates
  - `fit_boxes(points, corners, height, tolerance)` → candidates
- `main.py` - New endpoints:
  - `POST /fit-cylinders`
  - `POST /fit-boxes`

### Frontend (React/Three.js)

- `CylinderFitHandler.tsx` (new) - 3-click cylinder region definition
- `BoxFitHandler.tsx` (new) - 4-click box region definition
- `FitCandidatesPanel.tsx` (new) - candidate list UI
- `CylinderPreview.tsx` (new) - wireframe cylinder
- `BoxPreview.tsx` (new) - wireframe box
- `fittingStore.ts` (new) - candidates state, tolerance
- `selectionStore.ts` - add `cylinder-fit` and `box-fit` modes
- `ModeToolbar.tsx` - add mode buttons
- `Viewport.tsx` - integrate handlers

## Settings

| Parameter | Default | Range |
|-----------|---------|-------|
| Tolerance | 0.02m | 0.005m - 0.1m |
| Min inliers | 500 | 100 - 5000 |
| RANSAC iterations | 1000 | - |
| Max candidates | 10 | - |

## Keyboard Shortcuts

- `C` - Enter cylinder fit mode
- `X` - Enter box fit mode
- `Enter` - Apply accepted candidates
- `Escape` - Cancel fitting

## Edge Cases

- **No candidates found** - Show message, let user retry with larger tolerance
- **Too many candidates** - Limit to top 10 by point count
- **Overlapping candidates** - On accept, remove points from other candidates
- **Small regions** - Require minimum 100 points, show warning if fewer
