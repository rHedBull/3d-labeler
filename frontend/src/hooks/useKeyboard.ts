import { useEffect } from 'react'
import { useSelectionStore, type SelectionMode } from '../store/selectionStore'
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
