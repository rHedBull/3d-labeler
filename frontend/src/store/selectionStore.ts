import { create } from 'zustand'

export type SelectionMode = 'box' | 'lasso' | 'sphere' | 'geometric' | 'supervoxel' | 'rapid' | 'cylinder-fit' | 'box-fit'
export type NavigationMode = 'orbit' | 'walk'

interface SelectionState {
  mode: SelectionMode
  navigationMode: NavigationMode
  supervoxelResolution: number
  // Rapid labeling state
  rapidCurrentIndex: number
  rapidLabeling: boolean
  setMode: (mode: SelectionMode) => void
  setNavigationMode: (mode: NavigationMode) => void
  setSupervoxelResolution: (resolution: number) => void
  setRapidCurrentIndex: (index: number) => void
  startRapidLabeling: () => void
  stopRapidLabeling: () => void
}

export const useSelectionStore = create<SelectionState>((set) => ({
  mode: 'box',
  navigationMode: 'orbit',
  supervoxelResolution: 0.1,
  rapidCurrentIndex: 0,
  rapidLabeling: false,
  setMode: (mode) => set({ mode }),
  setNavigationMode: (mode) => set({ navigationMode: mode }),
  setSupervoxelResolution: (resolution) => set({ supervoxelResolution: resolution }),
  setRapidCurrentIndex: (index) => set({ rapidCurrentIndex: index }),
  startRapidLabeling: () => set({ rapidLabeling: true, rapidCurrentIndex: 0 }),
  stopRapidLabeling: () => set({ rapidLabeling: false }),
}))
