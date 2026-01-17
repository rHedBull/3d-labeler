import { Viewport } from './components/Viewport'
import { FilePanel } from './components/FilePanel'
import { ClassPanel } from './components/ClassPanel'
import { ModeToolbar } from './components/ModeToolbar'
import { useKeyboard } from './hooks/useKeyboard'
import { usePointCloudStore } from './store/pointCloudStore'

function App() {
  const { loading, error, numPoints, sceneName } = usePointCloudStore()

  useKeyboard()

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.title}>Point Cloud Labeler</span>
        {loading && <span style={styles.status}>Loading...</span>}
        {error && <span style={styles.error}>{error}</span>}
        {sceneName && (
          <span style={styles.status}>
            {sceneName} ({numPoints.toLocaleString()} points)
          </span>
        )}
      </div>

      {/* Main content */}
      <div style={styles.main}>
        <FilePanel />

        <div style={styles.viewport}>
          <ModeToolbar />
          <Viewport />
        </div>

        <ClassPanel />
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '100vw',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: '#1a1a2e',
  },
  header: {
    padding: '8px 16px',
    background: '#2d2d44',
    color: 'white',
    display: 'flex',
    gap: 16,
    alignItems: 'center',
  },
  title: {
    fontWeight: 600,
  },
  status: {
    fontSize: 13,
    color: '#aaa',
  },
  error: {
    fontSize: 13,
    color: '#ff6b6b',
  },
  main: {
    flex: 1,
    display: 'flex',
    overflow: 'hidden',
  },
  viewport: {
    flex: 1,
    position: 'relative',
  },
}

export default App
