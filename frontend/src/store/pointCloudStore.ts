import { create } from 'zustand'
import {
  loadPointCloud,
  savePointCloud,
  computeCluster,
  base64ToFloat32Array,
  base64ToUint8Array,
  base64ToInt32Array,
  computeSupervoxels as computeSupervoxelsAPI,
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
  supervoxelIds: Int32Array | null
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
  computeSupervoxels: (resolution?: number) => Promise<void>
  selectSupervoxel: (pointIndex: number, shiftKey: boolean, ctrlKey: boolean) => void
  selectGeometricCluster: (seedIndex: number, shiftKey: boolean, ctrlKey: boolean) => Promise<void>
}

export const usePointCloudStore = create<PointCloudState>((set, get) => ({
  points: null,
  colors: null,
  originalColors: null,
  labels: null,
  instanceIds: null,
  supervoxelIds: null,
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

  computeSupervoxels: async (resolution: number = 0.1) => {
    set({ loading: true, error: null })
    try {
      const data = await computeSupervoxelsAPI(resolution)
      const supervoxelIds = base64ToInt32Array(data.supervoxel_ids)
      set({ supervoxelIds, loading: false })
    } catch (e) {
      set({ loading: false, error: String(e) })
    }
  },

  selectSupervoxel: (pointIndex: number, shiftKey: boolean, ctrlKey: boolean) => {
    const { supervoxelIds, selectedIndices, numPoints } = get()
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

    get().setSelection(newSelection)
  },

  selectGeometricCluster: async (seedIndex: number, shiftKey: boolean, ctrlKey: boolean) => {
    const { selectedIndices } = get()
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

      get().setSelection(newSelection)
      set({ loading: false })
    } catch (e) {
      set({ loading: false, error: String(e) })
    }
  },
}))
