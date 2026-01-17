import { Viewport } from './components/Viewport'
import { usePointCloudStore } from './store/pointCloudStore'

function App() {
  const { loading, error, numPoints, sceneName } = usePointCloudStore()

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '8px', background: '#2d2d44', color: 'white', display: 'flex', gap: '16px' }}>
        <span>Point Cloud Labeler</span>
        {loading && <span>Loading...</span>}
        {error && <span style={{ color: 'red' }}>{error}</span>}
        {sceneName && <span>Scene: {sceneName} ({numPoints.toLocaleString()} points)</span>}
      </div>
      <div style={{ flex: 1 }}>
        <Viewport />
      </div>
    </div>
  )
}

export default App
