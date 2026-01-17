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
