import { create } from 'zustand'

export type SelectionMode = 'box' | 'lasso' | 'sphere' | 'geometric' | 'supervoxel'
export type NavigationMode = 'orbit' | 'walk'

interface SelectionState {
  mode: SelectionMode
  navigationMode: NavigationMode
  supervoxelResolution: number
  setMode: (mode: SelectionMode) => void
  setNavigationMode: (mode: NavigationMode) => void
  setSupervoxelResolution: (resolution: number) => void
}

export const useSelectionStore = create<SelectionState>((set) => ({
  mode: 'box',
  navigationMode: 'orbit',
  supervoxelResolution: 0.1,
  setMode: (mode) => set({ mode }),
  setNavigationMode: (mode) => set({ navigationMode: mode }),
  setSupervoxelResolution: (resolution) => set({ supervoxelResolution: resolution }),
}))
