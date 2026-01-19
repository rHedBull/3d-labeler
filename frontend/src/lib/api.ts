const API_BASE = '/api'

export interface LoadResponse {
  num_points: number
  points: string  // base64
  colors: string | null
  labels: string
  instance_ids: string | null
}

export interface SceneInfo {
  name: string
  has_source: boolean
  has_ground_truth: boolean
  source_type: string | null
}

export async function loadPointCloud(path: string, numSamples = 500000): Promise<LoadResponse> {
  const res = await fetch(`${API_BASE}/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, num_samples: numSamples }),
  })
  if (!res.ok) throw new Error(`Load failed: ${res.statusText}`)
  return res.json()
}

export async function savePointCloud(
  labels: Int32Array,
  instanceIds: Int32Array,
  sceneName: string
): Promise<{ success: boolean; num_points: number; path: string }> {
  const res = await fetch(`${API_BASE}/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      labels: arrayToBase64(labels),
      instance_ids: arrayToBase64(instanceIds),
      scene_name: sceneName,
    }),
  })
  if (!res.ok) throw new Error(`Save failed: ${res.statusText}`)
  return res.json()
}

export async function listFiles(): Promise<SceneInfo[]> {
  const res = await fetch(`${API_BASE}/files`)
  if (!res.ok) throw new Error(`List failed: ${res.statusText}`)
  return res.json()
}

export interface SupervoxelHull {
  vertices: number[][]  // Nx3 hull vertices
  faces: number[][]     // Triangle faces as vertex indices
}

export interface SupervoxelResponse {
  num_supervoxels: number
  supervoxel_ids: string
  centroids: string
  hulls: SupervoxelHull[]
}

export async function computeSupervoxels(resolution = 0.1, excludeMask?: Int32Array): Promise<SupervoxelResponse> {
  const body: Record<string, unknown> = { resolution }
  if (excludeMask) {
    body.exclude_mask = arrayToBase64(excludeMask)
  }
  const res = await fetch(`${API_BASE}/compute-supervoxels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Supervoxel computation failed: ${res.statusText}`)
  return res.json()
}

export interface ClusterResponse {
  indices: string
  num_points: number
}

export async function computeCluster(
  seedIndex: number,
  normalThreshold = 15,
  distanceThreshold = 0.05,
  maxPoints = 50000
): Promise<ClusterResponse> {
  const res = await fetch(`${API_BASE}/cluster`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      seed_index: seedIndex,
      normal_threshold: normalThreshold,
      distance_threshold: distanceThreshold,
      max_points: maxPoints,
    }),
  })
  if (!res.ok) throw new Error(`Cluster computation failed: ${res.statusText}`)
  return res.json()
}

// Helpers
export function base64ToFloat32Array(b64: string): Float32Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Float32Array(bytes.buffer)
}

export function base64ToUint8Array(b64: string): Uint8Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

export function base64ToInt32Array(b64: string): Int32Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Int32Array(bytes.buffer)
}

function arrayToBase64(arr: Int32Array | Float32Array | Uint8Array): string {
  const bytes = new Uint8Array(arr.buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}
