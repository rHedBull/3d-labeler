import { useSelectionStore } from '../store/selectionStore'
import { usePointCloudStore } from '../store/pointCloudStore'
import { useFittingStore, type FittedCylinder, type FittedBox } from '../store/fittingStore'

export function FitCandidatesPanel() {
  const { mode } = useSelectionStore()
  const { setSelection } = usePointCloudStore()
  const {
    cylinderPhase,
    boxPhase,
    fittedCylinders,
    fittedBoxes,
    tolerance,
    setTolerance,
    toggleCylinderAccepted,
    toggleBoxAccepted,
    resetCylinder,
    resetBox,
  } = useFittingStore()

  // Only show when in selecting phase
  const isCylinderSelecting = mode === 'cylinder-fit' && cylinderPhase === 'selecting'
  const isBoxSelecting = mode === 'box-fit' && boxPhase === 'selecting'

  if (!isCylinderSelecting && !isBoxSelecting) {
    return null
  }

  const candidates = isCylinderSelecting ? fittedCylinders : fittedBoxes
  const title = isCylinderSelecting ? 'Fitted Cylinders' : 'Fitted Boxes'
  const toggleAccepted = isCylinderSelecting ? toggleCylinderAccepted : toggleBoxAccepted
  const reset = isCylinderSelecting ? resetCylinder : resetBox

  const handleApply = () => {
    // Collect all point indices from accepted candidates
    const acceptedCandidates = candidates.filter((c: FittedCylinder | FittedBox) => c.accepted)
    const allIndices: number[] = []

    for (const candidate of acceptedCandidates) {
      allIndices.push(...candidate.pointIndices)
    }

    // Set selection and reset fitting state
    setSelection(new Set<number>(allIndices))
    reset()
  }

  const handleCancel = () => {
    reset()
  }

  const acceptedCount = candidates.filter((c: FittedCylinder | FittedBox) => c.accepted).length

  return (
    <div style={styles.panel}>
      <h3 style={styles.title}>{title}</h3>

      <div style={styles.list}>
        {candidates.length === 0 ? (
          <div style={styles.empty}>No candidates found</div>
        ) : (
          candidates.map((candidate: FittedCylinder | FittedBox, index: number) => (
            <div
              key={candidate.id}
              onClick={() => toggleAccepted(candidate.id)}
              style={{
                ...styles.item,
                background: candidate.accepted ? '#4a6a4a' : '#3a3a5a',
              }}
            >
              <span style={styles.icon}>
                {candidate.accepted ? '\u2713' : '\u2717'}
              </span>
              <span style={styles.label}>
                {isCylinderSelecting ? 'Cylinder' : 'Box'} {index + 1}
              </span>
              <span style={styles.count}>
                {candidate.pointIndices.length.toLocaleString()}
              </span>
            </div>
          ))
        )}
      </div>

      <div style={styles.sliderContainer}>
        <label style={styles.sliderLabel}>
          Tolerance: {(tolerance * 1000).toFixed(0)}mm
        </label>
        <input
          type="range"
          min="0.005"
          max="0.1"
          step="0.005"
          value={tolerance}
          onChange={(e) => setTolerance(parseFloat(e.target.value))}
          style={styles.slider}
        />
        <div style={styles.sliderHints}>
          <span>5mm</span>
          <span>100mm</span>
        </div>
      </div>

      <div style={styles.buttons}>
        <button
          onClick={handleApply}
          disabled={acceptedCount === 0}
          style={{
            ...styles.button,
            ...styles.applyButton,
            opacity: acceptedCount === 0 ? 0.5 : 1,
            cursor: acceptedCount === 0 ? 'not-allowed' : 'pointer',
          }}
        >
          Apply ({acceptedCount})
        </button>
        <button onClick={handleCancel} style={{ ...styles.button, ...styles.cancelButton }}>
          Cancel
        </button>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    position: 'absolute',
    top: 12,
    right: 12,
    width: 220,
    background: 'rgba(45, 45, 68, 0.95)',
    color: 'white',
    padding: 12,
    borderRadius: 8,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
    zIndex: 100,
  },
  title: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    maxHeight: 200,
    overflow: 'auto',
  },
  empty: {
    color: '#888',
    fontSize: 12,
    textAlign: 'center',
    padding: 8,
  },
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '6px 8px',
    borderRadius: 4,
    cursor: 'pointer',
  },
  icon: {
    fontSize: 14,
    width: 16,
    textAlign: 'center',
  },
  label: {
    flex: 1,
    fontSize: 12,
  },
  count: {
    fontSize: 11,
    color: '#aaa',
  },
  sliderContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  sliderLabel: {
    fontSize: 12,
    fontWeight: 500,
  },
  slider: {
    width: '100%',
    cursor: 'pointer',
  },
  sliderHints: {
    display: 'flex',
    justifyContent: 'space-between',
    color: '#888',
    fontSize: 10,
  },
  buttons: {
    display: 'flex',
    gap: 8,
  },
  button: {
    flex: 1,
    padding: '8px 12px',
    border: 'none',
    borderRadius: 4,
    fontSize: 12,
    fontWeight: 500,
    cursor: 'pointer',
  },
  applyButton: {
    background: '#4a8a4a',
    color: 'white',
  },
  cancelButton: {
    background: '#6a4a4a',
    color: 'white',
  },
}
