import { useMemo } from 'react'
import { usePointCloudStore, CLASS_COLORS, CLASS_NAMES } from '../store/pointCloudStore'

interface LabeledInstance {
  classId: number
  instanceId: number
  instanceNumber: number  // Per-class instance number (1, 2, 3...)
  pointCount: number
  indices: number[]
}

export function ClassPanel() {
  const {
    labels,
    instanceIds,
    selectedIndices,
    setLabels,
    clearSelection,
    setSelection,
    hideLabeledPoints,
    setHideLabeledPoints,
  } = usePointCloudStore()

  const handleAssignClass = (classId: number) => {
    if (selectedIndices.size === 0) return
    setLabels(Array.from(selectedIndices), classId)
    clearSelection()
  }

  // Count labels and collect instances
  const { labelCounts, instances } = useMemo(() => {
    const counts: Record<number, number> = {}
    // Map of "classId-instanceId" -> point indices
    const instanceMap: Record<string, number[]> = {}

    if (labels && instanceIds) {
      for (let i = 0; i < labels.length; i++) {
        const label = labels[i]
        counts[label] = (counts[label] || 0) + 1

        // Track explicitly labeled instances (instanceId > 0 means it was assigned)
        if (instanceIds[i] > 0) {
          const key = `${label}-${instanceIds[i]}`
          if (!instanceMap[key]) instanceMap[key] = []
          instanceMap[key].push(i)
        }
      }
    }

    // Convert to instance list and assign per-class instance numbers
    const classInstanceCounters: Record<number, number> = {}
    const instList: LabeledInstance[] = []

    // Sort by instanceId to maintain consistent ordering
    const sortedKeys = Object.keys(instanceMap).sort((a, b) => {
      const [, instA] = a.split('-').map(Number)
      const [, instB] = b.split('-').map(Number)
      return instA - instB
    })

    for (const key of sortedKeys) {
      const [classIdStr, instanceIdStr] = key.split('-')
      const classId = Number(classIdStr)
      const instanceId = Number(instanceIdStr)

      // Assign per-class instance number
      classInstanceCounters[classId] = (classInstanceCounters[classId] || 0) + 1

      instList.push({
        classId,
        instanceId,
        instanceNumber: classInstanceCounters[classId],
        pointCount: instanceMap[key].length,
        indices: instanceMap[key],
      })
    }

    return { labelCounts: counts, instances: instList }
  }, [labels, instanceIds])

  const handleInstanceClick = (instance: LabeledInstance) => {
    // Select all points in this instance
    setSelection(new Set(instance.indices))
  }

  const totalLabeled = instances.reduce((sum, i) => sum + i.pointCount, 0)

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

      {/* Labeled Instances Section */}
      {instances.length > 0 && (
        <>
          <div style={styles.divider} />
          <h3 style={styles.title}>
            Labeled ({totalLabeled.toLocaleString()})
          </h3>

          <label style={styles.toggleLabel}>
            <input
              type="checkbox"
              checked={hideLabeledPoints}
              onChange={(e) => setHideLabeledPoints(e.target.checked)}
            />
            Hide labeled points
          </label>

          <div style={styles.instanceList}>
            {instances.map((instance) => {
              const [r, g, b] = CLASS_COLORS[instance.classId]
              const name = CLASS_NAMES[instance.classId]
              const instanceName = `${name}_${instance.instanceNumber}`

              return (
                <div
                  key={`${instance.classId}-${instance.instanceId}`}
                  onClick={() => handleInstanceClick(instance)}
                  style={styles.instanceItem}
                >
                  <div
                    style={{
                      ...styles.colorBox,
                      background: `rgb(${r}, ${g}, ${b})`,
                    }}
                  />
                  <div style={styles.info}>
                    <div style={styles.instanceName}>{instanceName}</div>
                    <div style={styles.count}>
                      {instance.pointCount.toLocaleString()}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
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
  divider: {
    height: 1,
    background: '#444',
    margin: '8px 0',
  },
  toggleLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 12,
    cursor: 'pointer',
    marginBottom: 8,
  },
  instanceList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    maxHeight: 200,
    overflow: 'auto',
  },
  instanceItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '4px 6px',
    borderRadius: 4,
    cursor: 'pointer',
    background: '#3a3a5a',
  },
  instanceName: {
    fontSize: 11,
  },
}
