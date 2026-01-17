import { useEffect, useState } from 'react'
import { listFiles, type SceneInfo } from '../lib/api'
import { usePointCloudStore } from '../store/pointCloudStore'

export function FilePanel() {
  const [scenes, setScenes] = useState<SceneInfo[]>([])
  const [loadingList, setLoadingList] = useState(false)
  const { load, save, loading, sceneName } = usePointCloudStore()

  useEffect(() => {
    fetchScenes()
  }, [])

  const fetchScenes = async () => {
    setLoadingList(true)
    try {
      const data = await listFiles()
      setScenes(data)
    } catch (e) {
      console.error('Failed to fetch scenes:', e)
    }
    setLoadingList(false)
  }

  const handleLoad = (scene: SceneInfo) => {
    const path = scene.has_ground_truth
      ? `${scene.name}/ground_truth.ply`
      : `${scene.name}/source.${scene.source_type}`
    load(path)
  }

  return (
    <div style={styles.panel}>
      <h3 style={styles.title}>Files</h3>

      <div style={styles.actions}>
        <button onClick={fetchScenes} disabled={loadingList} style={styles.button}>
          Refresh
        </button>
        <button
          onClick={() => save()}
          disabled={loading || !sceneName}
          style={styles.button}
        >
          Save
        </button>
      </div>

      <div style={styles.list}>
        {loadingList ? (
          <div>Loading...</div>
        ) : scenes.length === 0 ? (
          <div style={styles.empty}>No scenes found in data/real/</div>
        ) : (
          scenes.map((scene) => (
            <div
              key={scene.name}
              onClick={() => handleLoad(scene)}
              style={{
                ...styles.item,
                background: scene.name === sceneName ? '#4a4a6a' : undefined,
              }}
            >
              <div style={styles.sceneName}>{scene.name}</div>
              <div style={styles.badges}>
                {scene.has_source && (
                  <span style={styles.badge}>{scene.source_type?.toUpperCase()}</span>
                )}
                {scene.has_ground_truth && (
                  <span style={{ ...styles.badge, background: '#2d8a2d' }}>GT</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 200,
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
  actions: {
    display: 'flex',
    gap: 8,
  },
  button: {
    flex: 1,
    padding: '6px 12px',
    background: '#4a4a6a',
    border: 'none',
    borderRadius: 4,
    color: 'white',
    cursor: 'pointer',
  },
  list: {
    flex: 1,
    overflow: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  item: {
    padding: 8,
    borderRadius: 4,
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sceneName: {
    fontSize: 13,
  },
  badges: {
    display: 'flex',
    gap: 4,
  },
  badge: {
    fontSize: 10,
    padding: '2px 6px',
    background: '#666',
    borderRadius: 3,
  },
  empty: {
    fontSize: 12,
    color: '#888',
    textAlign: 'center',
    padding: 16,
  },
}
