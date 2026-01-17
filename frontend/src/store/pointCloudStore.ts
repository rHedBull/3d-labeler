import { create } from 'zustand'
import {
  loadPointCloud,
  savePointCloud,
  computeCluster,
  base64ToFloat32Array,
  base64ToUint8Array,
  base64ToInt32Array,
  computeSupervoxels as computeSupervoxelsAPI,
  type SupervoxelHull,
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
  supervoxelCentroids: Float32Array | null
  supervoxelHulls: SupervoxelHull[] | null
  numPoints: number

  // Scene info
  sceneName: string | null
  loading: boolean
  error: string | null

  // Selection
  selectedIndices: Set<number>

  // View options
  hideLabeledPoints: boolean

  // Instance tracking
  nextInstanceId: number

  // Actions
  load: (path: string) => Promise<void>
  save: () => Promise<void>
  setLabels: (indices: number[], classId: number) => void
  setSelection: (indices: Set<number>) => void
  clearSelection: () => void
  updateColorsFromLabels: () => void
  computeSupervoxels: (resolution?: number) => Promise<void>
  selectSupervoxel: (pointIndex: number, shiftKey: boolean, ctrlKey: boolean) => void
  selectSupervoxelById: (supervoxelId: number, ctrlKey: boolean) => void
  selectGeometricCluster: (seedIndex: number, shiftKey: boolean, ctrlKey: boolean) => Promise<void>
  saveSession: () => void
  loadSession: () => boolean
  setHideLabeledPoints: (hide: boolean) => void
}

export const usePointCloudStore = create<PointCloudState>((set, get) => ({
  points: null,
  colors: null,
  originalColors: null,
  labels: null,
  instanceIds: null,
  supervoxelIds: null,
  supervoxelCentroids: null,
  supervoxelHulls: null,
  numPoints: 0,
  sceneName: null,
  loading: false,
  error: null,
  selectedIndices: new Set(),
  hideLabeledPoints: true,
  nextInstanceId: 1,

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
    const { labels, instanceIds, nextInstanceId } = get()
    if (!labels || !instanceIds) return

    // Each selection becomes a new instance (or clears instance if classId is 0)
    const newInstanceId = classId > 0 ? nextInstanceId : 0

    for (const idx of indices) {
      labels[idx] = classId
      instanceIds[idx] = newInstanceId
    }

    set({
      labels: new Int32Array(labels),
      instanceIds: new Int32Array(instanceIds),
      nextInstanceId: classId > 0 ? nextInstanceId + 1 : nextInstanceId,
    })
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
    const { labels } = get()
    set({ loading: true, error: null })
    try {
      // Pass labels as exclude mask - points with label > 0 will be excluded
      const data = await computeSupervoxelsAPI(resolution, labels || undefined)
      const supervoxelIds = base64ToInt32Array(data.supervoxel_ids)
      const supervoxelCentroids = base64ToFloat32Array(data.centroids)
      set({
        supervoxelIds,
        supervoxelCentroids,
        supervoxelHulls: data.hulls,
        loading: false,
      })
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

  selectSupervoxelById: (supervoxelId: number, ctrlKey: boolean) => {
    const { supervoxelIds, selectedIndices, numPoints } = get()
    if (!supervoxelIds) return

    // Always accumulate, use ctrl to remove
    const newSelection = new Set<number>(selectedIndices)

    for (let i = 0; i < numPoints; i++) {
      if (supervoxelIds[i] === supervoxelId) {
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

  saveSession: () => {
    const { sceneName, labels, instanceIds, nextInstanceId } = get()
    if (!sceneName || !labels || !instanceIds) return

    const session = {
      source_file: sceneName,
      labels: Array.from(labels),
      instance_ids: Array.from(instanceIds),
      next_instance_id: nextInstanceId,
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
        // Restore nextInstanceId, or compute from max existing instance ID
        const maxInstanceId = Math.max(0, ...newInstanceIds)
        const nextId = session.next_instance_id || maxInstanceId + 1
        set({ labels: newLabels, instanceIds: newInstanceIds, nextInstanceId: nextId })
        get().updateColorsFromLabels()
        return true
      }
    } catch (e) {
      console.error('Failed to load session:', e)
    }
    return false
  },

  setHideLabeledPoints: (hide: boolean) => {
    set({ hideLabeledPoints: hide })
  },
}))
