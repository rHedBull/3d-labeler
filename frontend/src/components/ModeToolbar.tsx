import { useState, useEffect, useRef } from 'react'
import { useSelectionStore, type SelectionMode } from '../store/selectionStore'
import { usePointCloudStore } from '../store/pointCloudStore'

const MODES: { id: SelectionMode; key: string; label: string; icon: string }[] = [
  { id: 'box', key: 'B', label: 'Box Select', icon: '▢' },
  { id: 'lasso', key: 'L', label: 'Lasso Select', icon: '◯' },
  { id: 'sphere', key: 'S', label: 'Sphere Select', icon: '●' },
  { id: 'geometric', key: 'G', label: 'Geometric Cluster', icon: '◈' },
  { id: 'supervoxel', key: 'V', label: 'Supervoxel', icon: '⬡' },
]

export function ModeToolbar() {
  const { mode, setMode, supervoxelResolution, setSupervoxelResolution } = useSelectionStore()
  const { computeSupervoxels, supervoxelIds, loading } = usePointCloudStore()

  // Local slider value for immediate visual feedback
  const [localResolution, setLocalResolution] = useState(supervoxelResolution)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Sync local state when store value changes (e.g., on initial load)
  useEffect(() => {
    setLocalResolution(supervoxelResolution)
  }, [supervoxelResolution])

  // Debounced computation - only triggers 500ms after user stops dragging
  const handleResolutionChange = (newResolution: number) => {
    setLocalResolution(newResolution)

    // Clear any pending computation
    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }

    // Schedule new computation after debounce delay
    debounceRef.current = setTimeout(async () => {
      setSupervoxelResolution(newResolution)
      await computeSupervoxels(newResolution)
    }, 500)
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [])

  return (
    <div style={styles.container}>
      <div style={styles.toolbar}>
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            style={{
              ...styles.button,
              background: mode === m.id ? '#6a6a8a' : '#4a4a6a',
            }}
            title={`${m.label} (${m.key})`}
          >
            <span style={styles.icon}>{m.icon}</span>
            <span style={styles.key}>{m.key}</span>
          </button>
        ))}
      </div>

      {/* Supervoxel resolution slider - shown when in supervoxel mode */}
      {mode === 'supervoxel' && (
        <div style={styles.sliderContainer}>
          <label style={styles.sliderLabel}>
            Voxel Size: {localResolution.toFixed(2)}m
            {localResolution !== supervoxelResolution && !loading && ' (drag to apply)'}
          </label>
          <input
            type="range"
            min="0.05"
            max="0.5"
            step="0.01"
            value={localResolution}
            onChange={(e) => handleResolutionChange(parseFloat(e.target.value))}
            style={styles.slider}
          />
          <div style={styles.sliderHints}>
            <span>Fine</span>
            <span>Coarse</span>
          </div>
          {loading && <span style={styles.computing}>Computing...</span>}
          {supervoxelIds && !loading && (
            <span style={styles.voxelCount}>
              {new Set(supervoxelIds).size} supervoxels
            </span>
          )}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'absolute',
    top: 12,
    left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 8,
    zIndex: 100,
  },
  toolbar: {
    display: 'flex',
    gap: 4,
    background: 'rgba(45, 45, 68, 0.9)',
    padding: 4,
    borderRadius: 6,
  },
  button: {
    width: 40,
    height: 40,
    border: 'none',
    borderRadius: 4,
    color: 'white',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
  },
  icon: {
    fontSize: 16,
  },
  key: {
    fontSize: 10,
    opacity: 0.7,
  },
  sliderContainer: {
    background: 'rgba(45, 45, 68, 0.9)',
    padding: '8px 12px',
    borderRadius: 6,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 4,
    minWidth: 180,
  },
  sliderLabel: {
    color: 'white',
    fontSize: 12,
    fontWeight: 500,
  },
  slider: {
    width: '100%',
    cursor: 'pointer',
  },
  sliderHints: {
    width: '100%',
    display: 'flex',
    justifyContent: 'space-between',
    color: '#888',
    fontSize: 10,
  },
  computing: {
    color: '#88f',
    fontSize: 11,
  },
  voxelCount: {
    color: '#8f8',
    fontSize: 11,
  },
}
