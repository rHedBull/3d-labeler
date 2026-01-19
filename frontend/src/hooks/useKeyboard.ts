import { useEffect } from 'react'
import { useSelectionStore, type SelectionMode } from '../store/selectionStore'
import { usePointCloudStore } from '../store/pointCloudStore'

const MODE_KEYS: Record<string, SelectionMode> = {
  b: 'box',
  l: 'lasso',
  s: 'sphere',
  g: 'geometric',
  v: 'supervoxel',
  r: 'rapid',
}

const CLASS_KEYS = ['0', '1', '2', '3', '4', '5', '6']

export function useKeyboard() {
  const { setMode, navigationMode, setNavigationMode } = useSelectionStore()
  const { selectedIndices, setLabels, clearSelection } = usePointCloudStore()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      const key = e.key.toLowerCase()

      // Toggle walk mode with F
      if (key === 'f') {
        setNavigationMode(navigationMode === 'orbit' ? 'walk' : 'orbit')
        return
      }

      // Mode switching (only when not in walk mode to avoid conflicts with WASD)
      if (MODE_KEYS[key] && navigationMode !== 'walk') {
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
  }, [setMode, setNavigationMode, navigationMode, selectedIndices, setLabels, clearSelection])
}
