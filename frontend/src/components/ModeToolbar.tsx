import { useSelectionStore, type SelectionMode } from '../store/selectionStore'

const MODES: { id: SelectionMode; key: string; label: string; icon: string }[] = [
  { id: 'box', key: 'B', label: 'Box Select', icon: '▢' },
  { id: 'lasso', key: 'L', label: 'Lasso Select', icon: '◯' },
  { id: 'sphere', key: 'S', label: 'Sphere Select', icon: '●' },
  { id: 'geometric', key: 'G', label: 'Geometric Cluster', icon: '◈' },
  { id: 'supervoxel', key: 'V', label: 'Supervoxel', icon: '⬡' },
]

export function ModeToolbar() {
  const { mode, setMode } = useSelectionStore()

  return (
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
  )
}

const styles: Record<string, React.CSSProperties> = {
  toolbar: {
    position: 'absolute',
    top: 12,
    left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex',
    gap: 4,
    background: 'rgba(45, 45, 68, 0.9)',
    padding: 4,
    borderRadius: 6,
    zIndex: 100,
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
}
