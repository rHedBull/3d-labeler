import { useEffect, useState } from 'react'
import { usePointCloudStore, CLASS_NAMES, CLASS_COLORS } from '../store/pointCloudStore'

export function ConfirmDialog() {
  const { selectedIndices, setLabels, clearSelection } = usePointCloudStore()
  const [pendingClass, setPendingClass] = useState<number | null>(null)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return

      const key = e.key

      // Check if it's a class key (0-6)
      if (/^[0-6]$/.test(key) && selectedIndices.size > 0) {
        setPendingClass(parseInt(key))
        return
      }

      // Confirm with Enter
      if (key === 'Enter' && pendingClass !== null) {
        setLabels(Array.from(selectedIndices), pendingClass)
        clearSelection()
        setPendingClass(null)
        return
      }

      // Cancel with Escape
      if (key === 'Escape') {
        setPendingClass(null)
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedIndices, pendingClass, setLabels, clearSelection])

  if (pendingClass === null || selectedIndices.size === 0) return null

  const [r, g, b] = CLASS_COLORS[pendingClass]
  const className = CLASS_NAMES[pendingClass]

  return (
    <div style={styles.overlay}>
      <div style={styles.dialog}>
        <div style={styles.header}>
          <div
            style={{
              ...styles.colorBox,
              background: `rgb(${r}, ${g}, ${b})`,
            }}
          />
          <span>Assign '{className}' to {selectedIndices.size.toLocaleString()} points?</span>
        </div>
        <div style={styles.actions}>
          <span style={styles.hint}>[Enter] Accept</span>
          <span style={styles.hint}>[Esc] Cancel</span>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'absolute',
    bottom: 20,
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 200,
  },
  dialog: {
    background: 'rgba(45, 45, 68, 0.95)',
    borderRadius: 8,
    padding: '12px 20px',
    color: 'white',
    boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  colorBox: {
    width: 20,
    height: 20,
    borderRadius: 4,
  },
  actions: {
    display: 'flex',
    gap: 16,
    justifyContent: 'center',
  },
  hint: {
    fontSize: 12,
    color: '#888',
  },
}
