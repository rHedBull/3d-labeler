import { create } from 'zustand'

export type SelectionMode = 'box' | 'lasso' | 'sphere' | 'geometric' | 'supervoxel'

interface SelectionState {
  mode: SelectionMode
  supervoxelResolution: number
  setMode: (mode: SelectionMode) => void
  setSupervoxelResolution: (resolution: number) => void
}

export const useSelectionStore = create<SelectionState>((set) => ({
  mode: 'box',
  supervoxelResolution: 0.1,
  setMode: (mode) => set({ mode }),
  setSupervoxelResolution: (resolution) => set({ supervoxelResolution: resolution }),
}))
