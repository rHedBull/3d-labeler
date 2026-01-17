import { usePointCloudStore, CLASS_COLORS, CLASS_NAMES } from '../store/pointCloudStore'

export function ClassPanel() {
  const { labels, selectedIndices, setLabels, clearSelection } = usePointCloudStore()

  const handleAssignClass = (classId: number) => {
    if (selectedIndices.size === 0) return
    setLabels(Array.from(selectedIndices), classId)
    clearSelection()
  }

  // Count labels
  const labelCounts: Record<number, number> = {}
  if (labels) {
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i]
      labelCounts[label] = (labelCounts[label] || 0) + 1
    }
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.title}>Classes</h3>

      {selectedIndices.size > 0 && (
        <div style={styles.selection}>
          {selectedIndices.size.toLocaleString()} selected
        </div>
      )}

      <div style={styles.list}>
        {Object.entries(CLASS_NAMES).map(([id, name]) => {
          const classId = Number(id)
          const [r, g, b] = CLASS_COLORS[classId]
          const count = labelCounts[classId] || 0

          return (
            <div
              key={id}
              onClick={() => handleAssignClass(classId)}
              style={styles.item}
            >
              <div
                style={{
                  ...styles.colorBox,
                  background: `rgb(${r}, ${g}, ${b})`,
                }}
              />
              <div style={styles.info}>
                <div style={styles.name}>
                  <span style={styles.key}>{classId}</span> {name}
                </div>
                <div style={styles.count}>{count.toLocaleString()}</div>
              </div>
            </div>
          )
        })}
      </div>

      <div style={styles.hint}>
        Press 0-6 to assign class to selection
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 180,
    background: '#2d2d44',
    color: 'white',
    padding: 12,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  title: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
  },
  selection: {
    padding: '6px 10px',
    background: '#4a6a4a',
    borderRadius: 4,
    fontSize: 12,
    textAlign: 'center',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: 6,
    borderRadius: 4,
    cursor: 'pointer',
  },
  colorBox: {
    width: 16,
    height: 16,
    borderRadius: 3,
    flexShrink: 0,
  },
  info: {
    flex: 1,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  name: {
    fontSize: 12,
  },
  key: {
    display: 'inline-block',
    width: 14,
    height: 14,
    lineHeight: '14px',
    textAlign: 'center',
    background: '#555',
    borderRadius: 2,
    marginRight: 4,
    fontSize: 10,
  },
  count: {
    fontSize: 11,
    color: '#888',
  },
  hint: {
    fontSize: 11,
    color: '#666',
    textAlign: 'center',
  },
}
