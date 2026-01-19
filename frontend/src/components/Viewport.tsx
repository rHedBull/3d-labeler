import { useRef, useMemo, useEffect, useState, useCallback, createContext, useContext } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

// Context to share OrbitControls ref with child components
const OrbitControlsContext = createContext<React.RefObject<OrbitControlsImpl | null> | null>(null)
import * as THREE from 'three'
import { usePointCloudStore } from '../store/pointCloudStore'
import { useSelectionStore } from '../store/selectionStore'

// Generate a distinct color for a supervoxel ID using golden ratio for good distribution
function getSupervoxelColor(id: number): [number, number, number] {
  const goldenRatio = 0.618033988749895
  const hue = (id * goldenRatio) % 1
  // Convert HSL to RGB (saturation=0.7, lightness=0.5)
  const s = 0.7
  const l = 0.5
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs((hue * 6) % 2 - 1))
  const m = l - c / 2
  let r = 0, g = 0, b = 0
  if (hue < 1/6) { r = c; g = x; b = 0 }
  else if (hue < 2/6) { r = x; g = c; b = 0 }
  else if (hue < 3/6) { r = 0; g = c; b = x }
  else if (hue < 4/6) { r = 0; g = x; b = c }
  else if (hue < 5/6) { r = x; g = 0; b = c }
  else { r = c; g = 0; b = x }
  return [r + m, g + m, b + m]
}

// Render all supervoxel hulls as a single merged geometry for performance
function SupervoxelHulls({
  onHullClick,
}: {
  onHullClick: (supervoxelId: number, ctrlKey: boolean) => void
}) {
  const { supervoxelHulls, supervoxelIds, selectedIndices } = usePointCloudStore()
  const { mode } = useSelectionStore()
  const meshRef = useRef<THREE.Mesh>(null)

  // Build a set of selected supervoxel IDs for highlighting
  const selectedSupervoxelIds = useMemo(() => {
    if (!supervoxelIds) return new Set<number>()
    const ids = new Set<number>()
    for (const idx of selectedIndices) {
      ids.add(supervoxelIds[idx])
    }
    return ids
  }, [supervoxelIds, selectedIndices])

  // Merge all hulls into single geometry with vertex colors
  // Also build face-to-supervoxel mapping for click detection
  const { geometry, faceToSupervoxel } = useMemo(() => {
    if (!supervoxelHulls) return { geometry: null, faceToSupervoxel: [] }

    const allPositions: number[] = []
    const allColors: number[] = []
    const allIndices: number[] = []
    const faceMap: number[] = [] // Maps face index to supervoxel ID

    let vertexOffset = 0
    const goldenRatio = 0.618033988749895

    for (let svId = 0; svId < supervoxelHulls.length; svId++) {
      const hull = supervoxelHulls[svId]
      if (hull.faces.length === 0) continue

      const isSelected = selectedSupervoxelIds.has(svId)
      const hue = (svId * goldenRatio) % 1
      const color = new THREE.Color().setHSL(hue, 0.7, isSelected ? 0.8 : 0.5)

      // Add vertices
      for (const vertex of hull.vertices) {
        allPositions.push(vertex[0], vertex[1], vertex[2])
        allColors.push(color.r, color.g, color.b)
      }

      // Add faces with offset indices
      for (const face of hull.faces) {
        allIndices.push(face[0] + vertexOffset, face[1] + vertexOffset, face[2] + vertexOffset)
        faceMap.push(svId)
      }

      vertexOffset += hull.vertices.length
    }

    if (allPositions.length === 0) return { geometry: null, faceToSupervoxel: [] }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(allPositions, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(allColors, 3))
    geo.setIndex(allIndices)
    geo.computeVertexNormals()

    return { geometry: geo, faceToSupervoxel: faceMap }
  }, [supervoxelHulls, selectedSupervoxelIds])

  // Handle click - find which supervoxel was clicked
  const handleClick = useCallback((e: { stopPropagation: () => void; faceIndex?: number; nativeEvent?: MouseEvent }) => {
    if (!faceToSupervoxel.length) return
    e.stopPropagation()

    // Get face index from intersection
    const faceIndex = e.faceIndex
    if (faceIndex !== undefined && faceIndex !== null) {
      const svId = faceToSupervoxel[faceIndex]
      if (svId !== undefined) {
        onHullClick(svId, e.nativeEvent?.ctrlKey || false)
      }
    }
  }, [faceToSupervoxel, onHullClick])

  if (mode !== 'supervoxel' || !geometry) return null

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      onClick={handleClick}
      onPointerOver={() => { document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { document.body.style.cursor = 'default' }}
    >
      <meshBasicMaterial
        vertexColors
        transparent
        opacity={0.25}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}

// Morton code helper for spatial sorting (also used in RapidLabelingController)
function computeMortonCode(x: number, y: number, z: number): number {
  const ix = Math.min(1023, Math.max(0, Math.floor(x * 1023)))
  const iy = Math.min(1023, Math.max(0, Math.floor(y * 1023)))
  const iz = Math.min(1023, Math.max(0, Math.floor(z * 1023)))
  let code = 0
  for (let i = 0; i < 10; i++) {
    code |= ((ix >> i) & 1) << (3 * i)
    code |= ((iy >> i) & 1) << (3 * i + 1)
    code |= ((iz >> i) & 1) << (3 * i + 2)
  }
  return code
}

function PointCloudMesh() {
  const meshRef = useRef<THREE.Points>(null)
  const { points, colors, numPoints, selectedIndices, supervoxelIds, supervoxelPointMap, supervoxelCentroids, labels, instanceIds, hideLabeledPoints } = usePointCloudStore()
  const { mode, rapidCurrentIndex, rapidLabeling } = useSelectionStore()

  // Calculate unlabeled supervoxels list sorted spatially
  // Use instanceIds > 0 to track labeled (not labels === 0, since 0 is background class)
  const unlabeledSupervoxelsList = useMemo(() => {
    if (!supervoxelPointMap || !instanceIds || !supervoxelCentroids) return []

    // Find bounding box for normalization
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
    for (let i = 0; i < supervoxelCentroids.length / 3; i++) {
      const x = supervoxelCentroids[i * 3], y = supervoxelCentroids[i * 3 + 1], z = supervoxelCentroids[i * 3 + 2]
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
    }
    const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1, rangeZ = maxZ - minZ || 1

    const unlabeled: { svId: number; morton: number }[] = []
    for (const [svId, indices] of supervoxelPointMap) {
      // A supervoxel is unlabeled if ANY point has instanceId === 0 (not yet explicitly labeled)
      const hasUnlabeledPoint = indices.some(i => instanceIds[i] === 0)
      if (hasUnlabeledPoint) {
        const cx = supervoxelCentroids[svId * 3], cy = supervoxelCentroids[svId * 3 + 1], cz = supervoxelCentroids[svId * 3 + 2]
        const nx = (cx - minX) / rangeX, ny = (cy - minY) / rangeY, nz = (cz - minZ) / rangeZ
        unlabeled.push({ svId, morton: computeMortonCode(nx, ny, nz) })
      }
    }

    unlabeled.sort((a, b) => a.morton - b.morton)
    return unlabeled.map(u => u.svId)
  }, [supervoxelPointMap, instanceIds, supervoxelCentroids])

  // Get current supervoxel ID for rapid mode - O(1) lookup
  const currentRapidSupervoxelId = useMemo(() => {
    if (mode !== 'rapid' || !rapidLabeling) return -1
    return unlabeledSupervoxelsList[rapidCurrentIndex] ?? -1
  }, [mode, rapidLabeling, unlabeledSupervoxelsList, rapidCurrentIndex])

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()

    if (points && colors) {
      // Create a copy of positions that we can modify for hiding
      const positions = new Float32Array(points)
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))

      // Normalize colors to 0-1 range
      const normalizedColors = new Float32Array(numPoints * 3)
      for (let i = 0; i < numPoints * 3; i++) {
        normalizedColors[i] = colors[i] / 255
      }
      geo.setAttribute('color', new THREE.BufferAttribute(normalizedColors, 3))
    }

    return geo
  }, [points, colors, numPoints])

  // Update positions, colors when selection, mode, or hide setting changes
  useEffect(() => {
    if (!meshRef.current || !colors || !points) return

    const posAttr = meshRef.current.geometry.getAttribute('position') as THREE.BufferAttribute
    const colorAttr = meshRef.current.geometry.getAttribute('color') as THREE.BufferAttribute
    if (!colorAttr || !posAttr) return

    const positions = posAttr.array as Float32Array
    const normalizedColors = colorAttr.array as Float32Array
    const showSupervoxels = mode === 'supervoxel' && supervoxelIds
    const isRapidMode = mode === 'rapid' && rapidLabeling && supervoxelIds
    const isRapidModeWaiting = mode === 'rapid' && !supervoxelIds // Waiting for supervoxel computation

    for (let i = 0; i < numPoints; i++) {
      // Hide explicitly labeled points (instanceId > 0) by moving them far away
      const isLabeled = instanceIds && instanceIds[i] > 0
      const shouldHide = hideLabeledPoints && isLabeled

      if (shouldHide) {
        positions[i * 3] = 1e10
        positions[i * 3 + 1] = 1e10
        positions[i * 3 + 2] = 1e10
      } else {
        positions[i * 3] = points[i * 3]
        positions[i * 3 + 1] = points[i * 3 + 1]
        positions[i * 3 + 2] = points[i * 3 + 2]
      }

      if (selectedIndices.has(i)) {
        // Highlight selected points in bright cyan for high contrast
        normalizedColors[i * 3] = 0
        normalizedColors[i * 3 + 1] = 1
        normalizedColors[i * 3 + 2] = 1
      } else if (isRapidMode) {
        // In rapid mode: highlight current supervoxel strongly, dim everything else
        const svId = supervoxelIds[i]
        if (svId === currentRapidSupervoxelId) {
          // Current supervoxel - VERY bright white/yellow for maximum visibility
          normalizedColors[i * 3] = 1
          normalizedColors[i * 3 + 1] = 1
          normalizedColors[i * 3 + 2] = 0.3
        } else if (labels && labels[i] > 0) {
          // Already labeled - show class color but very dimmed
          const classColors: Record<number, [number, number, number]> = {
            1: [0, 0, 1], 2: [0, 1, 1], 3: [1, 0, 0],
            4: [0, 1, 0], 5: [1, 1, 0], 6: [1, 0.5, 0],
          }
          const cc = classColors[labels[i]] || [0.5, 0.5, 0.5]
          normalizedColors[i * 3] = cc[0] * 0.15
          normalizedColors[i * 3 + 1] = cc[1] * 0.15
          normalizedColors[i * 3 + 2] = cc[2] * 0.15
        } else {
          // Unlabeled, not current - very dim gray to maximize contrast with current
          normalizedColors[i * 3] = 0.12
          normalizedColors[i * 3 + 1] = 0.12
          normalizedColors[i * 3 + 2] = 0.12
        }
      } else if (isRapidModeWaiting) {
        // Waiting for supervoxels - show original colors with a slight tint to indicate loading
        normalizedColors[i * 3] = colors[i * 3] / 255 * 0.7 + 0.1
        normalizedColors[i * 3 + 1] = colors[i * 3 + 1] / 255 * 0.7 + 0.1
        normalizedColors[i * 3 + 2] = colors[i * 3 + 2] / 255 * 0.7
      } else if (showSupervoxels) {
        // Show supervoxel colors when in supervoxel mode
        const svColor = getSupervoxelColor(supervoxelIds[i])
        normalizedColors[i * 3] = svColor[0]
        normalizedColors[i * 3 + 1] = svColor[1]
        normalizedColors[i * 3 + 2] = svColor[2]
      } else {
        normalizedColors[i * 3] = colors[i * 3] / 255
        normalizedColors[i * 3 + 1] = colors[i * 3 + 1] / 255
        normalizedColors[i * 3 + 2] = colors[i * 3 + 2] / 255
      }
    }

    posAttr.needsUpdate = true
    colorAttr.needsUpdate = true
  }, [selectedIndices, colors, numPoints, mode, supervoxelIds, labels, instanceIds, hideLabeledPoints, points, rapidLabeling, currentRapidSupervoxelId])

  if (!points) return null

  return (
    <points ref={meshRef} geometry={geometry}>
      <pointsMaterial
        size={0.02}
        vertexColors
        sizeAttenuation
      />
    </points>
  )
}

function CameraController() {
  const { camera } = useThree()
  const { points } = usePointCloudStore()

  useEffect(() => {
    if (!points || points.length === 0) return

    // Calculate bounding box and center camera
    const positions = points
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

    for (let i = 0; i < positions.length; i += 3) {
      minX = Math.min(minX, positions[i])
      maxX = Math.max(maxX, positions[i])
      minY = Math.min(minY, positions[i + 1])
      maxY = Math.max(maxY, positions[i + 1])
      minZ = Math.min(minZ, positions[i + 2])
      maxZ = Math.max(maxZ, positions[i + 2])
    }

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const centerZ = (minZ + maxZ) / 2

    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ)

    camera.position.set(centerX + size, centerY + size, centerZ + size)
    camera.lookAt(centerX, centerY, centerZ)
  }, [points, camera])

  return null
}

// Find the point closest to a ray (for click-based selection)
function findClosestPointToRay(
  ray: THREE.Ray,
  points: Float32Array,
  numPoints: number,
  threshold: number = 0.5
): { index: number; position: THREE.Vector3 } | null {
  let closestDist = Infinity
  let closestIdx = -1
  const point = new THREE.Vector3()

  for (let i = 0; i < numPoints; i++) {
    point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
    const dist = ray.distanceToPoint(point)

    if (dist < closestDist && dist < threshold) {
      closestDist = dist
      closestIdx = i
    }
  }

  if (closestIdx >= 0) {
    return {
      index: closestIdx,
      position: new THREE.Vector3(
        points[closestIdx * 3],
        points[closestIdx * 3 + 1],
        points[closestIdx * 3 + 2]
      ),
    }
  }

  return null
}

// Handler for sphere selection
function SphereSelectionHandler({
  sphereState,
  onSphereUpdate,
  onSphereComplete,
}: {
  sphereState: { center: THREE.Vector3 | null; radius: number; isDragging: boolean }
  onSphereUpdate: (center: THREE.Vector3 | null, radius: number) => void
  onSphereComplete: (indices: number[], shiftKey: boolean, ctrlKey: boolean) => void
}) {
  const { camera, raycaster, gl } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()

  const handleMouseDown = useCallback((e: MouseEvent) => {
    if (mode !== 'sphere' || e.button !== 0 || !points) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )

    raycaster.setFromCamera(mouse, camera)
    const hit = findClosestPointToRay(raycaster.ray, points, numPoints)

    if (hit) {
      onSphereUpdate(hit.position, 0)
    }
  }, [mode, points, numPoints, camera, raycaster, gl, onSphereUpdate])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'sphere' || !sphereState.center || !sphereState.isDragging) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )

    raycaster.setFromCamera(mouse, camera)

    // Calculate radius based on distance from center to ray
    const closestPoint = new THREE.Vector3()
    raycaster.ray.closestPointToPoint(sphereState.center, closestPoint)
    const radius = sphereState.center.distanceTo(closestPoint)

    onSphereUpdate(sphereState.center, radius)
  }, [mode, sphereState.center, sphereState.isDragging, camera, raycaster, gl, onSphereUpdate])

  const handleMouseUp = useCallback((e: MouseEvent) => {
    if (mode !== 'sphere' || !sphereState.center || !sphereState.isDragging || !points) {
      return
    }

    // Select points within sphere
    const radiusSq = sphereState.radius * sphereState.radius
    const newIndices: number[] = []

    for (let i = 0; i < numPoints; i++) {
      const dx = points[i * 3] - sphereState.center.x
      const dy = points[i * 3 + 1] - sphereState.center.y
      const dz = points[i * 3 + 2] - sphereState.center.z
      const distSq = dx * dx + dy * dy + dz * dz

      if (distSq <= radiusSq) {
        newIndices.push(i)
      }
    }

    // Always accumulate (shiftKey=true), use ctrlKey to remove
    onSphereComplete(newIndices, true, e.ctrlKey)
    onSphereUpdate(null, 0)
  }, [mode, sphereState, points, numPoints, onSphereComplete, onSphereUpdate])

  useEffect(() => {
    const canvas = gl.domElement
    canvas.addEventListener('mousedown', handleMouseDown)
    canvas.addEventListener('mousemove', handleMouseMove)
    canvas.addEventListener('mouseup', handleMouseUp)

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown)
      canvas.removeEventListener('mousemove', handleMouseMove)
      canvas.removeEventListener('mouseup', handleMouseUp)
    }
  }, [gl, handleMouseDown, handleMouseMove, handleMouseUp])

  // Render sphere preview
  if (!sphereState.center || sphereState.radius === 0) return null

  return (
    <mesh position={sphereState.center}>
      <sphereGeometry args={[sphereState.radius, 32, 32]} />
      <meshBasicMaterial color="#ffffff" transparent opacity={0.2} wireframe />
    </mesh>
  )
}

// Handler for lasso selection
function LassoSelectionHandler({
  lassoPoints,
  onLassoUpdate,
  onLassoComplete,
}: {
  lassoPoints: { x: number; y: number }[]
  onLassoUpdate: (points: { x: number; y: number }[]) => void
  onLassoComplete: (indices: number[], shiftKey: boolean, ctrlKey: boolean) => void
}) {
  const { camera, size, gl } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()
  const isDrawing = useRef(false)

  const handleMouseDown = useCallback((e: MouseEvent) => {
    if (mode !== 'lasso' || e.button !== 0) return

    const rect = gl.domElement.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    isDrawing.current = true
    onLassoUpdate([{ x, y }])
  }, [mode, gl, onLassoUpdate])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'lasso' || !isDrawing.current) return

    const rect = gl.domElement.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    onLassoUpdate([...lassoPoints, { x, y }])
  }, [mode, lassoPoints, gl, onLassoUpdate])

  const handleMouseUp = useCallback((e: MouseEvent) => {
    if (mode !== 'lasso' || !isDrawing.current || lassoPoints.length < 3 || !points) {
      isDrawing.current = false
      onLassoUpdate([])
      return
    }

    isDrawing.current = false

    // Convert lasso points to NDC
    const lassoNDC = lassoPoints.map(p => ({
      x: (p.x / size.width) * 2 - 1,
      y: -(p.y / size.height) * 2 + 1,
    }))

    // Project points and check if inside lasso polygon
    const projScreenMatrix = new THREE.Matrix4()
    projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)

    const newIndices: number[] = []
    const point = new THREE.Vector3()

    for (let i = 0; i < numPoints; i++) {
      point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
      point.applyMatrix4(projScreenMatrix)

      // Check if point is in front of camera and inside polygon
      if (point.z < 1 && isPointInPolygon(point.x, point.y, lassoNDC)) {
        newIndices.push(i)
      }
    }

    // Always accumulate (shiftKey=true), use ctrlKey to remove
    onLassoComplete(newIndices, true, e.ctrlKey)
    onLassoUpdate([])
  }, [mode, lassoPoints, points, numPoints, camera, size, onLassoComplete, onLassoUpdate])

  useEffect(() => {
    const canvas = gl.domElement
    canvas.addEventListener('mousedown', handleMouseDown)
    canvas.addEventListener('mousemove', handleMouseMove)
    canvas.addEventListener('mouseup', handleMouseUp)

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown)
      canvas.removeEventListener('mousemove', handleMouseMove)
      canvas.removeEventListener('mouseup', handleMouseUp)
    }
  }, [gl, handleMouseDown, handleMouseMove, handleMouseUp])

  return null
}

// Point-in-polygon test using ray casting algorithm
function isPointInPolygon(x: number, y: number, polygon: { x: number; y: number }[]): boolean {
  let inside = false
  const n = polygon.length

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y
    const xj = polygon[j].x, yj = polygon[j].y

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside
    }
  }

  return inside
}

// 3D Box selection with draggable corners and rotation
interface Box3DState {
  min: THREE.Vector3
  max: THREE.Vector3
  rotation: THREE.Euler
  isActive: boolean
  phase: 'none' | 'placing' | 'adjusting'
}

function Box3DSelector({
  boxState,
  onBoxChange,
  onSelectionUpdate,
}: {
  boxState: Box3DState
  onBoxChange: (state: Box3DState) => void
  onSelectionUpdate: (indices: number[]) => void
}) {
  const { camera, raycaster, gl } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()
  const dragHandle = useRef<string | null>(null)
  const dragPlane = useRef<THREE.Plane>(new THREE.Plane())
  const dragStart = useRef<THREE.Vector3>(new THREE.Vector3())
  const boxStartMin = useRef<THREE.Vector3>(new THREE.Vector3())
  const boxStartMax = useRef<THREE.Vector3>(new THREE.Vector3())
  const boxStartRotation = useRef<THREE.Euler>(new THREE.Euler())
  const rotationStartAngle = useRef<number>(0)
  const isMouseDown = useRef(false)
  const initialClickPos = useRef<THREE.Vector3 | null>(null)

  // Calculate points inside rotated box whenever it changes
  useEffect(() => {
    if (!boxState.isActive || !points || boxState.phase === 'none') return

    const indices: number[] = []
    const min = boxState.min
    const max = boxState.max
    const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5)
    const halfSize = new THREE.Vector3().subVectors(max, min).multiplyScalar(0.5)

    // Create inverse rotation matrix to transform points into box local space
    const rotationMatrix = new THREE.Matrix4().makeRotationFromEuler(boxState.rotation)
    const inverseRotation = rotationMatrix.clone().invert()

    const point = new THREE.Vector3()
    const localPoint = new THREE.Vector3()

    for (let i = 0; i < numPoints; i++) {
      point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])

      // Transform point to box local space
      localPoint.copy(point).sub(center)
      localPoint.applyMatrix4(inverseRotation)

      // Check if in box bounds (in local space, box is axis-aligned)
      if (Math.abs(localPoint.x) <= halfSize.x &&
          Math.abs(localPoint.y) <= halfSize.y &&
          Math.abs(localPoint.z) <= halfSize.z) {
        indices.push(i)
      }
    }

    onSelectionUpdate(indices)
  }, [boxState.min, boxState.max, boxState.rotation, boxState.isActive, boxState.phase, points, numPoints, onSelectionUpdate])

  // Get handle positions (8 corners + center for moving)
  // Handles are in local box space, then transformed to world space
  const { cornerHandles, center, size } = useMemo(() => {
    if (!boxState.isActive) return { cornerHandles: [], center: new THREE.Vector3(), size: new THREE.Vector3() }
    const { min, max, rotation } = boxState
    const c = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5)
    const s = new THREE.Vector3().subVectors(max, min)
    const halfSize = s.clone().multiplyScalar(0.5)

    // Local corner positions (before rotation)
    const localCorners = [
      { id: 'min-min-min', local: new THREE.Vector3(-halfSize.x, -halfSize.y, -halfSize.z) },
      { id: 'max-min-min', local: new THREE.Vector3(halfSize.x, -halfSize.y, -halfSize.z) },
      { id: 'min-max-min', local: new THREE.Vector3(-halfSize.x, halfSize.y, -halfSize.z) },
      { id: 'max-max-min', local: new THREE.Vector3(halfSize.x, halfSize.y, -halfSize.z) },
      { id: 'min-min-max', local: new THREE.Vector3(-halfSize.x, -halfSize.y, halfSize.z) },
      { id: 'max-min-max', local: new THREE.Vector3(halfSize.x, -halfSize.y, halfSize.z) },
      { id: 'min-max-max', local: new THREE.Vector3(-halfSize.x, halfSize.y, halfSize.z) },
      { id: 'max-max-max', local: new THREE.Vector3(halfSize.x, halfSize.y, halfSize.z) },
    ]

    // Transform corners to world space
    const rotationMatrix = new THREE.Matrix4().makeRotationFromEuler(rotation)
    const handles = localCorners.map(({ id, local }) => {
      const world = local.clone().applyMatrix4(rotationMatrix).add(c)
      return { id, pos: world, color: '#ffff00' }
    })

    // Add center handle
    handles.push({ id: 'center', pos: c.clone(), color: '#00ff00' })

    return { cornerHandles: handles, center: c, size: s }
  }, [boxState])

  // Get mouse position in 3D at a certain depth
  const getMousePosition3D = useCallback((e: MouseEvent, depth: number) => {
    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)
    const pos = new THREE.Vector3()
    raycaster.ray.at(depth, pos)
    return pos
  }, [gl, camera, raycaster])

  // Get intersection with a plane
  const getPlaneIntersection = useCallback((e: MouseEvent, plane: THREE.Plane) => {
    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)
    const intersection = new THREE.Vector3()
    raycaster.ray.intersectPlane(plane, intersection)
    return intersection
  }, [gl, camera, raycaster])

  // Handle mouse down - start placing or dragging
  const handleMouseDown = useCallback((e: MouseEvent) => {
    if (mode !== 'box' || e.button !== 0) return

    isMouseDown.current = true

    if (boxState.phase === 'none' || !boxState.isActive) {
      // Start placing a new box - place at distance from camera
      const cameraDir = new THREE.Vector3()
      camera.getWorldDirection(cameraDir)
      const distance = 10 // Default distance
      const startPos = getMousePosition3D(e, distance)

      initialClickPos.current = startPos.clone()

      // Create a plane perpendicular to camera for drawing
      dragPlane.current.setFromNormalAndCoplanarPoint(cameraDir, startPos)

      onBoxChange({
        min: startPos.clone(),
        max: startPos.clone().add(new THREE.Vector3(0.1, 0.1, 0.1)),
        rotation: new THREE.Euler(0, 0, 0),
        isActive: true,
        phase: 'placing',
      })
    }
  }, [mode, boxState, camera, getMousePosition3D, onBoxChange])

  // Handle mouse move
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'box' || !isMouseDown.current) return

    if (boxState.phase === 'placing' && initialClickPos.current) {
      // Extend box from initial click position
      const currentPos = getPlaneIntersection(e, dragPlane.current)

      if (currentPos) {
        const min = new THREE.Vector3(
          Math.min(initialClickPos.current.x, currentPos.x),
          Math.min(initialClickPos.current.y, currentPos.y),
          Math.min(initialClickPos.current.z, currentPos.z)
        )
        const max = new THREE.Vector3(
          Math.max(initialClickPos.current.x, currentPos.x),
          Math.max(initialClickPos.current.y, currentPos.y),
          Math.max(initialClickPos.current.z, currentPos.z)
        )

        // Ensure minimum size
        if (max.x - min.x < 0.1) max.x = min.x + 0.1
        if (max.y - min.y < 0.1) max.y = min.y + 0.1
        if (max.z - min.z < 0.1) max.z = min.z + 0.1

        onBoxChange({
          ...boxState,
          min,
          max,
        })
      }
    } else if (boxState.phase === 'adjusting' && dragHandle.current) {
      const intersection = getPlaneIntersection(e, dragPlane.current)

      if (intersection) {
        if (dragHandle.current.startsWith('rotate-')) {
          // Handle rotation
          const boxCenter = new THREE.Vector3().addVectors(boxStartMin.current, boxStartMax.current).multiplyScalar(0.5)
          const toMouse = intersection.clone().sub(boxCenter)
          let currentAngle = 0
          const newRotation = boxStartRotation.current.clone()

          if (dragHandle.current === 'rotate-y') {
            currentAngle = Math.atan2(toMouse.x, toMouse.z)
            newRotation.y = boxStartRotation.current.y + (currentAngle - rotationStartAngle.current)
          } else if (dragHandle.current === 'rotate-x') {
            currentAngle = Math.atan2(toMouse.y, toMouse.z)
            newRotation.x = boxStartRotation.current.x + (currentAngle - rotationStartAngle.current)
          } else if (dragHandle.current === 'rotate-z') {
            currentAngle = Math.atan2(toMouse.y, toMouse.x)
            newRotation.z = boxStartRotation.current.z + (currentAngle - rotationStartAngle.current)
          }

          onBoxChange({
            ...boxState,
            rotation: newRotation,
          })
        } else {
          const delta = intersection.clone().sub(dragStart.current)

          if (dragHandle.current === 'center') {
            // Move entire box
            onBoxChange({
              ...boxState,
              min: boxStartMin.current.clone().add(delta),
              max: boxStartMax.current.clone().add(delta),
            })
          } else {
            // Resize from corner - transform delta to local space for proper scaling
            const inverseRotation = new THREE.Matrix4().makeRotationFromEuler(boxState.rotation).invert()
            const localDelta = delta.clone().applyMatrix4(inverseRotation)

            const handleParts = dragHandle.current.split('-')
            const newMin = boxStartMin.current.clone()
            const newMax = boxStartMax.current.clone()

            if (handleParts[0] === 'min') newMin.x = Math.min(boxStartMin.current.x + localDelta.x, newMax.x - 0.1)
            else newMax.x = Math.max(boxStartMax.current.x + localDelta.x, newMin.x + 0.1)

            if (handleParts[1] === 'min') newMin.y = Math.min(boxStartMin.current.y + localDelta.y, newMax.y - 0.1)
            else newMax.y = Math.max(boxStartMax.current.y + localDelta.y, newMin.y + 0.1)

            if (handleParts[2] === 'min') newMin.z = Math.min(boxStartMin.current.z + localDelta.z, newMax.z - 0.1)
            else newMax.z = Math.max(boxStartMax.current.z + localDelta.z, newMin.z + 0.1)

            onBoxChange({
              ...boxState,
              min: newMin,
              max: newMax,
            })
          }
        }
      }
    }
  }, [mode, boxState, getPlaneIntersection, onBoxChange])

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    if (mode !== 'box') return

    isMouseDown.current = false
    initialClickPos.current = null

    if (boxState.phase === 'placing') {
      onBoxChange({
        ...boxState,
        phase: 'adjusting',
      })
    }

    dragHandle.current = null
  }, [mode, boxState, onBoxChange])

  // Handle keyboard for apply/cancel
  useEffect(() => {
    if (mode !== 'box') return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && boxState.isActive) {
        // Apply selection (already updated via effect)
        onBoxChange({ min: new THREE.Vector3(), max: new THREE.Vector3(), rotation: new THREE.Euler(), isActive: false, phase: 'none' })
      } else if (e.key === 'Escape') {
        // Cancel
        onSelectionUpdate([])
        onBoxChange({ min: new THREE.Vector3(), max: new THREE.Vector3(), rotation: new THREE.Euler(), isActive: false, phase: 'none' })
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mode, boxState.isActive, onBoxChange, onSelectionUpdate])

  // Attach mouse listeners to canvas
  useEffect(() => {
    if (mode !== 'box') return

    const canvas = gl.domElement
    canvas.addEventListener('mousedown', handleMouseDown)
    canvas.addEventListener('mousemove', handleMouseMove)
    canvas.addEventListener('mouseup', handleMouseUp)

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown)
      canvas.removeEventListener('mousemove', handleMouseMove)
      canvas.removeEventListener('mouseup', handleMouseUp)
    }
  }, [mode, gl, handleMouseDown, handleMouseMove, handleMouseUp])

  // Handle starting to drag a corner, center, or rotation handle
  const startDragHandle = useCallback((handleId: string, handlePos: THREE.Vector3, e: { stopPropagation: () => void }) => {
    if (boxState.phase !== 'adjusting') return

    e.stopPropagation()
    isMouseDown.current = true
    dragHandle.current = handleId

    // Create a plane perpendicular to camera for dragging
    const cameraDir = new THREE.Vector3()
    camera.getWorldDirection(cameraDir)
    dragPlane.current.setFromNormalAndCoplanarPoint(cameraDir, handlePos)

    // Store starting positions
    dragStart.current.copy(handlePos)
    boxStartMin.current.copy(boxState.min)
    boxStartMax.current.copy(boxState.max)
    boxStartRotation.current.copy(boxState.rotation)

    // For rotation, calculate starting angle
    if (handleId.startsWith('rotate-')) {
      const boxCenter = new THREE.Vector3().addVectors(boxState.min, boxState.max).multiplyScalar(0.5)
      const toHandle = handlePos.clone().sub(boxCenter)
      if (handleId === 'rotate-y') {
        rotationStartAngle.current = Math.atan2(toHandle.x, toHandle.z)
      } else if (handleId === 'rotate-x') {
        rotationStartAngle.current = Math.atan2(toHandle.y, toHandle.z)
      } else if (handleId === 'rotate-z') {
        rotationStartAngle.current = Math.atan2(toHandle.y, toHandle.x)
      }
    }
  }, [boxState, camera])

  if (mode !== 'box' || !boxState.isActive) return null

  // Calculate ring radius based on box size
  const ringRadius = Math.max(size.x, size.y, size.z) * 0.6

  return (
    <>
      {/* Box group - rotated around center */}
      <group position={center} rotation={boxState.rotation}>
        {/* Wireframe box */}
        <mesh>
          <boxGeometry args={[size.x, size.y, size.z]} />
          <meshBasicMaterial color="#00ffff" wireframe transparent opacity={0.8} />
        </mesh>

        {/* Semi-transparent faces */}
        <mesh>
          <boxGeometry args={[size.x, size.y, size.z]} />
          <meshBasicMaterial color="#00ffff" transparent opacity={0.1} side={THREE.DoubleSide} />
        </mesh>

        {/* Rotation rings - only in adjusting phase */}
        {boxState.phase === 'adjusting' && (
          <>
            {/* Y-axis rotation ring (green) - horizontal ring */}
            <mesh
              rotation={[Math.PI / 2, 0, 0]}
              onPointerDown={(e) => {
                const worldPos = new THREE.Vector3(ringRadius, 0, 0).applyEuler(boxState.rotation).add(center)
                startDragHandle('rotate-y', worldPos, e)
              }}
              onPointerOver={() => { document.body.style.cursor = 'ew-resize' }}
              onPointerOut={() => { document.body.style.cursor = 'default' }}
            >
              <torusGeometry args={[ringRadius, 0.05, 8, 32]} />
              <meshBasicMaterial color="#00ff00" transparent opacity={0.6} />
            </mesh>

            {/* X-axis rotation ring (red) - vertical ring around X */}
            <mesh
              rotation={[0, Math.PI / 2, 0]}
              onPointerDown={(e) => {
                const worldPos = new THREE.Vector3(0, ringRadius, 0).applyEuler(boxState.rotation).add(center)
                startDragHandle('rotate-x', worldPos, e)
              }}
              onPointerOver={() => { document.body.style.cursor = 'ns-resize' }}
              onPointerOut={() => { document.body.style.cursor = 'default' }}
            >
              <torusGeometry args={[ringRadius, 0.05, 8, 32]} />
              <meshBasicMaterial color="#ff0000" transparent opacity={0.6} />
            </mesh>

            {/* Z-axis rotation ring (blue) */}
            <mesh
              onPointerDown={(e) => {
                const worldPos = new THREE.Vector3(0, ringRadius, 0).applyEuler(boxState.rotation).add(center)
                startDragHandle('rotate-z', worldPos, e)
              }}
              onPointerOver={() => { document.body.style.cursor = 'nesw-resize' }}
              onPointerOut={() => { document.body.style.cursor = 'default' }}
            >
              <torusGeometry args={[ringRadius, 0.05, 8, 32]} />
              <meshBasicMaterial color="#0088ff" transparent opacity={0.6} />
            </mesh>
          </>
        )}
      </group>

      {/* Corner and center handles - in world space (outside rotated group) */}
      {boxState.phase === 'adjusting' && cornerHandles.map((handle) => (
        <mesh
          key={handle.id}
          position={handle.pos}
          onPointerDown={(e) => startDragHandle(handle.id, handle.pos, e)}
          onPointerOver={() => { document.body.style.cursor = handle.id === 'center' ? 'move' : 'grab' }}
          onPointerOut={() => { document.body.style.cursor = 'default' }}
        >
          <sphereGeometry args={[handle.id === 'center' ? 0.2 : 0.15, 16, 16]} />
          <meshBasicMaterial color={handle.color} />
        </mesh>
      ))}
    </>
  )
}

// Rapid labeling mode - auto-focus on each supervoxel, press number to label and advance
function RapidLabelingController() {
  const { camera } = useThree()
  const controlsRef = useContext(OrbitControlsContext)
  const { points, numPoints, supervoxelIds, supervoxelPointMap, supervoxelCentroids, computeSupervoxels, labels, instanceIds, setLabels } = usePointCloudStore()
  const { mode, rapidCurrentIndex, setRapidCurrentIndex, supervoxelResolution, rapidLabeling, startRapidLabeling, stopRapidLabeling } = useSelectionStore()

  // Get unlabeled supervoxels sorted spatially using Morton code
  // Use instanceIds === 0 to track unlabeled (not labels === 0, since 0 is background class)
  const unlabeledSupervoxels = useMemo(() => {
    if (!supervoxelPointMap || !instanceIds || !supervoxelCentroids) return []

    // Find bounding box of all centroids for normalization
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

    for (let i = 0; i < supervoxelCentroids.length / 3; i++) {
      const x = supervoxelCentroids[i * 3]
      const y = supervoxelCentroids[i * 3 + 1]
      const z = supervoxelCentroids[i * 3 + 2]
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z)
    }

    const rangeX = maxX - minX || 1
    const rangeY = maxY - minY || 1
    const rangeZ = maxZ - minZ || 1

    // Collect unlabeled supervoxels with their Morton codes
    const unlabeled: { svId: number; morton: number }[] = []
    for (const [svId, indices] of supervoxelPointMap) {
      // A supervoxel is unlabeled if ANY point has instanceId === 0 (not yet explicitly labeled)
      const hasUnlabeledPoint = indices.some(i => instanceIds[i] === 0)
      if (hasUnlabeledPoint) {
        // Get centroid and compute Morton code
        const cx = supervoxelCentroids[svId * 3]
        const cy = supervoxelCentroids[svId * 3 + 1]
        const cz = supervoxelCentroids[svId * 3 + 2]

        // Normalize to 0-1 range
        const nx = (cx - minX) / rangeX
        const ny = (cy - minY) / rangeY
        const nz = (cz - minZ) / rangeZ

        unlabeled.push({ svId, morton: computeMortonCode(nx, ny, nz) })
      }
    }

    // Sort by Morton code for spatial locality
    unlabeled.sort((a, b) => a.morton - b.morton)
    return unlabeled.map(u => u.svId)
  }, [supervoxelPointMap, instanceIds, supervoxelCentroids])

  // Keep index in bounds when list shrinks (after labeling)
  useEffect(() => {
    if (unlabeledSupervoxels.length > 0 && rapidCurrentIndex >= unlabeledSupervoxels.length) {
      setRapidCurrentIndex(0)
    }
  }, [unlabeledSupervoxels.length, rapidCurrentIndex, setRapidCurrentIndex])

  // Auto-compute supervoxels when entering rapid mode
  useEffect(() => {
    if (mode === 'rapid' && points && !supervoxelIds) {
      computeSupervoxels(supervoxelResolution)
    }
    if (mode === 'rapid' && supervoxelIds && !rapidLabeling) {
      startRapidLabeling()
    }
    if (mode !== 'rapid' && rapidLabeling) {
      stopRapidLabeling()
    }
  }, [mode, points, supervoxelIds, computeSupervoxels, supervoxelResolution, rapidLabeling, startRapidLabeling, stopRapidLabeling])

  // Get current supervoxel to label
  const currentSupervoxelId = unlabeledSupervoxels[rapidCurrentIndex] ?? -1

  // Auto-focus camera on current supervoxel whenever it changes
  useEffect(() => {
    if (mode !== 'rapid' || !rapidLabeling) return
    if (currentSupervoxelId < 0 || !supervoxelPointMap || !points) return

    // Use the point map for O(k) lookup where k is points in this supervoxel
    const indices = supervoxelPointMap.get(currentSupervoxelId)
    if (!indices || indices.length === 0) return

    // Calculate centroid and size from the supervoxel's points
    let sumX = 0, sumY = 0, sumZ = 0
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

    for (const i of indices) {
      const x = points[i * 3]
      const y = points[i * 3 + 1]
      const z = points[i * 3 + 2]
      sumX += x
      sumY += y
      sumZ += z
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      minZ = Math.min(minZ, z)
      maxX = Math.max(maxX, x)
      maxY = Math.max(maxY, y)
      maxZ = Math.max(maxZ, z)
    }

    const count = indices.length
    const centroid = new THREE.Vector3(sumX / count, sumY / count, sumZ / count)
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.5)

    // Position camera based on patch size - further for larger patches
    const distance = Math.max(size * 2.5, 2)
    const offset = new THREE.Vector3(distance, distance * 0.5, distance)
    const targetPosition = centroid.clone().add(offset)

    camera.position.copy(targetPosition)
    camera.lookAt(centroid)

    // Update OrbitControls target so it doesn't fight with our camera position
    if (controlsRef?.current) {
      controlsRef.current.target.copy(centroid)
      controlsRef.current.update()
    }
  }, [mode, currentSupervoxelId, rapidLabeling, supervoxelPointMap, points, camera, controlsRef])

  // Handle keyboard for rapid labeling
  useEffect(() => {
    if (mode !== 'rapid' || !rapidLabeling) return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      const key = e.key

      // Number keys 0-6 to label current supervoxel
      if (['0', '1', '2', '3', '4', '5', '6'].includes(key) && currentSupervoxelId >= 0 && supervoxelPointMap) {
        const classId = parseInt(key)

        // Get all point indices for current supervoxel using the map - O(1) lookup
        const indices = supervoxelPointMap.get(currentSupervoxelId) || []

        // Label them
        setLabels(indices, classId)

        // For class 0 (background), manually advance since the list won't auto-shrink
        // (labels[i] === 0 is still considered "unlabeled" in our tracking)
        if (classId === 0) {
          const nextIndex = (rapidCurrentIndex + 1) % unlabeledSupervoxels.length
          setRapidCurrentIndex(nextIndex)
        }
        // For classes 1-6, the list shrinks automatically and the bounds effect handles wrapping
        return
      }

      // Arrow keys to navigate (wrap around)
      if (key === 'ArrowRight' || key === ' ') {
        const nextIndex = (rapidCurrentIndex + 1) % unlabeledSupervoxels.length
        setRapidCurrentIndex(nextIndex)
        return
      }
      if (key === 'ArrowLeft') {
        const prevIndex = rapidCurrentIndex === 0 ? unlabeledSupervoxels.length - 1 : rapidCurrentIndex - 1
        setRapidCurrentIndex(prevIndex)
        return
      }

      // Escape to exit rapid mode
      if (key === 'Escape') {
        stopRapidLabeling()
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mode, rapidLabeling, currentSupervoxelId, supervoxelPointMap, setLabels, rapidCurrentIndex, setRapidCurrentIndex, unlabeledSupervoxels.length, stopRapidLabeling])

  // Highlight current supervoxel
  useEffect(() => {
    if (mode !== 'rapid' || currentSupervoxelId < 0) return

    // This is handled via the point colors in PointCloudMesh
  }, [mode, currentSupervoxelId])

  return null
}

// Walk controls for FPS-style navigation
// Works alongside selection modes - right-click+drag to look, WASD to move
function WalkControls() {
  const { camera, gl } = useThree()
  const { navigationMode } = useSelectionStore()
  const keysPressed = useRef<Set<string>>(new Set())
  const isDragging = useRef(false)
  const lastMouse = useRef({ x: 0, y: 0 })
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'))
  const moveSpeed = 5 // units per second
  const verticalSpeed = 3 // units per second
  const lookSpeed = 0.002

  // Initialize euler from camera on mode switch
  useEffect(() => {
    if (navigationMode === 'walk') {
      euler.current.setFromQuaternion(camera.quaternion, 'YXZ')
    }
  }, [navigationMode, camera])

  // Track key presses
  useEffect(() => {
    if (navigationMode !== 'walk') return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't capture if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      keysPressed.current.add(e.key.toLowerCase())
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      keysPressed.current.delete(e.key.toLowerCase())
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      keysPressed.current.clear()
    }
  }, [navigationMode])

  // Right-click+drag for looking around
  useEffect(() => {
    if (navigationMode !== 'walk') return

    const handleMouseDown = (e: MouseEvent) => {
      if (e.button === 2) { // Right click
        isDragging.current = true
        lastMouse.current = { x: e.clientX, y: e.clientY }
        e.preventDefault()
      }
    }

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return

      const deltaX = e.clientX - lastMouse.current.x
      const deltaY = e.clientY - lastMouse.current.y
      lastMouse.current = { x: e.clientX, y: e.clientY }

      // Update euler angles
      euler.current.y -= deltaX * lookSpeed
      euler.current.x -= deltaY * lookSpeed

      // Clamp vertical look to avoid flipping
      euler.current.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, euler.current.x))

      // Apply to camera
      camera.quaternion.setFromEuler(euler.current)
    }

    const handleMouseUp = (e: MouseEvent) => {
      if (e.button === 2) {
        isDragging.current = false
      }
    }

    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault() // Prevent context menu on right-click
    }

    const canvas = gl.domElement
    canvas.addEventListener('mousedown', handleMouseDown)
    canvas.addEventListener('mousemove', handleMouseMove)
    canvas.addEventListener('mouseup', handleMouseUp)
    canvas.addEventListener('contextmenu', handleContextMenu)

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown)
      canvas.removeEventListener('mousemove', handleMouseMove)
      canvas.removeEventListener('mouseup', handleMouseUp)
      canvas.removeEventListener('contextmenu', handleContextMenu)
    }
  }, [navigationMode, gl, camera])

  // Movement in useFrame
  useFrame((_, delta) => {
    if (navigationMode !== 'walk') return

    const keys = keysPressed.current

    // Get camera's horizontal forward direction (ignore pitch for W always horizontal)
    const forward = new THREE.Vector3()
    camera.getWorldDirection(forward)
    forward.y = 0 // Make horizontal
    forward.normalize()

    // Get right vector (strafe direction)
    const right = new THREE.Vector3()
    right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize()

    // Calculate movement
    const movement = new THREE.Vector3()

    if (keys.has('w')) movement.add(forward)
    if (keys.has('s')) movement.sub(forward)
    if (keys.has('a')) movement.sub(right)
    if (keys.has('d')) movement.add(right)

    // Vertical movement
    if (keys.has('q')) movement.y -= 1
    if (keys.has('e')) movement.y += 1

    // Apply movement
    if (movement.length() > 0) {
      // Separate horizontal and vertical for different speeds
      const horizontal = new THREE.Vector3(movement.x, 0, movement.z)
      if (horizontal.length() > 0) {
        horizontal.normalize().multiplyScalar(moveSpeed * delta)
        camera.position.add(horizontal)
      }

      if (movement.y !== 0) {
        camera.position.y += Math.sign(movement.y) * verticalSpeed * delta
      }
    }
  })

  return null
}

// Handler for click-based selection (geometric, supervoxel)
// These modes accumulate selections by default (always add, unless Ctrl to remove)
function ClickSelectionHandler() {
  const { camera, raycaster, gl, size } = useThree()
  const { points, numPoints, selectSupervoxel, selectGeometricCluster, supervoxelIds, computeSupervoxels } = usePointCloudStore()
  const { mode, supervoxelResolution } = useSelectionStore()

  // Auto-compute supervoxels when entering supervoxel mode
  useEffect(() => {
    if (mode === 'supervoxel' && points && !supervoxelIds) {
      computeSupervoxels(supervoxelResolution)
    }
  }, [mode, points, supervoxelIds, computeSupervoxels, supervoxelResolution])

  const handleClick = useCallback(async (e: MouseEvent) => {
    if (!['geometric', 'supervoxel'].includes(mode) || e.button !== 0 || !points) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )

    raycaster.setFromCamera(mouse, camera)

    // Use larger threshold for supervoxel mode to make clicking easier
    const threshold = mode === 'supervoxel' ? 2.0 : 0.5
    const hit = findClosestPointToRay(raycaster.ray, points, numPoints, threshold)

    // For supervoxel mode, also try finding by screen projection if ray miss
    if (!hit && mode === 'supervoxel' && supervoxelIds) {
      // Find closest point by 2D screen projection
      const projScreenMatrix = new THREE.Matrix4()
      projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)

      let closestDist = Infinity
      let closestIdx = -1
      const point = new THREE.Vector3()
      const mouseNDC = { x: mouse.x, y: mouse.y }

      for (let i = 0; i < numPoints; i++) {
        point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
        point.applyMatrix4(projScreenMatrix)

        if (point.z < 1) { // In front of camera
          const dx = point.x - mouseNDC.x
          const dy = point.y - mouseNDC.y
          const dist = dx * dx + dy * dy

          if (dist < closestDist) {
            closestDist = dist
            closestIdx = i
          }
        }
      }

      // Accept if within reasonable screen distance (0.1 in NDC = ~5% of screen)
      if (closestIdx >= 0 && closestDist < 0.01) {
        const addToSelection = true
        const removeFromSelection = e.ctrlKey
        selectSupervoxel(closestIdx, addToSelection, removeFromSelection)
        return
      }
    }

    if (!hit) return

    // Always accumulate selections (act as if Shift is held), unless Ctrl to remove
    const addToSelection = true  // Always add to existing selection
    const removeFromSelection = e.ctrlKey

    if (mode === 'geometric') {
      await selectGeometricCluster(hit.index, addToSelection, removeFromSelection)
    } else if (mode === 'supervoxel') {
      // Compute supervoxels if not already done
      if (!supervoxelIds) {
        await computeSupervoxels(supervoxelResolution)
      }
      selectSupervoxel(hit.index, addToSelection, removeFromSelection)
    }
  }, [mode, points, numPoints, camera, raycaster, gl, size, selectGeometricCluster, selectSupervoxel, supervoxelIds, computeSupervoxels, supervoxelResolution])

  useEffect(() => {
    const canvas = gl.domElement
    canvas.addEventListener('click', handleClick)

    return () => {
      canvas.removeEventListener('click', handleClick)
    }
  }, [gl, handleClick])

  return null
}

export function Viewport() {
  const { mode, navigationMode, rapidLabeling, rapidCurrentIndex } = useSelectionStore()
  const { selectedIndices, setSelection, selectSupervoxelById, supervoxelIds, supervoxelPointMap, labels, instanceIds } = usePointCloudStore()
  const canvasRef = useRef<HTMLDivElement>(null)
  const orbitControlsRef = useRef<OrbitControlsImpl | null>(null)

  // Handler for clicking on supervoxel hulls
  const handleHullClick = useCallback((supervoxelId: number, ctrlKey: boolean) => {
    selectSupervoxelById(supervoxelId, ctrlKey)
  }, [selectSupervoxelById])

  // Compute rapid labeling progress using the point map - O(m) where m is number of supervoxels
  const rapidProgress = useMemo(() => {
    if (!supervoxelPointMap || !instanceIds) return { unlabeled: 0, total: 0, labeled: 0 }

    let unlabeledCount = 0
    for (const [, indices] of supervoxelPointMap) {
      // Use instanceIds to track labeling - instanceId > 0 means explicitly labeled
      const hasUnlabeledPoint = indices.some(i => instanceIds[i] === 0)
      if (hasUnlabeledPoint) unlabeledCount++
    }

    const total = supervoxelPointMap.size
    return {
      unlabeled: unlabeledCount,
      total,
      labeled: total - unlabeledCount,
    }
  }, [supervoxelPointMap, instanceIds])

  // 3D Box selection state
  const [box3DState, setBox3DState] = useState<Box3DState>({
    min: new THREE.Vector3(),
    max: new THREE.Vector3(),
    rotation: new THREE.Euler(),
    isActive: false,
    phase: 'none',
  })

  // Sphere selection state
  const [sphereState, setSphereState] = useState<{
    center: THREE.Vector3 | null
    radius: number
    isDragging: boolean
  }>({ center: null, radius: 0, isDragging: false })

  // Lasso selection state
  const [lassoPoints, setLassoPoints] = useState<{ x: number; y: number }[]>([])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (mode === 'sphere' && e.button === 0) {
      // Sphere dragging is started when center is set
      setSphereState(prev => ({ ...prev, isDragging: true }))
    }
  }

  const handleMouseMove = (_e: React.MouseEvent) => {
    // Box mode now handled by Box3DSelector
  }

  const handleMouseUp = () => {
    // Box mode now handled by Box3DSelector
  }

  // Handle 3D box selection updates
  const handleBox3DSelectionUpdate = useCallback((indices: number[]) => {
    // Update selection in real-time as box changes
    setSelection(new Set(indices))
  }, [setSelection])

  // All selection modes accumulate by default (always add to existing selection)
  // Use Ctrl+select to remove from selection
  const handleSelectionComplete = (indices: number[], _shiftKey: boolean, ctrlKey: boolean) => {
    const newSelection = new Set<number>(selectedIndices) // Always keep existing selection
    for (const i of indices) {
      if (ctrlKey) {
        newSelection.delete(i)
      } else {
        newSelection.add(i)
      }
    }
    setSelection(newSelection)
  }

  const handleSphereUpdate = (center: THREE.Vector3 | null, radius: number) => {
    setSphereState(prev => ({
      ...prev,
      center,
      radius,
      isDragging: center !== null,
    }))
  }

  // Determine if OrbitControls should be enabled
  // Keep orbit enabled for right-click/middle-click camera controls in box mode
  const orbitEnabled = navigationMode === 'orbit' && !(
    (mode === 'sphere' && sphereState.isDragging) ||
    (mode === 'lasso' && lassoPoints.length > 0)
  )

  // Determine mouse button behavior
  const getLeftMouseAction = () => {
    // Disable left-click for orbit when in selection modes
    if (['box', 'sphere', 'lasso'].includes(mode)) {
      return undefined // Reserved for selection
    }
    return THREE.MOUSE.ROTATE
  }

  return (
    <div
      ref={canvasRef}
      style={{ width: '100%', height: '100%', position: 'relative' }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Canvas
        camera={{ position: [10, 10, 10], fov: 50, near: 0.1, far: 10000 }}
        style={{ background: '#1a1a2e' }}
      >
        <ambientLight intensity={0.5} />
        <CameraController />
        <OrbitControls
          ref={orbitControlsRef}
          enableDamping
          dampingFactor={0.05}
          enabled={orbitEnabled}
          mouseButtons={{
            LEFT: getLeftMouseAction(),
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.ROTATE,
          }}
        />
        <OrbitControlsContext.Provider value={orbitControlsRef}>
          <PointCloudMesh />
          <SupervoxelHulls onHullClick={handleHullClick} />
          <Box3DSelector
            boxState={box3DState}
            onBoxChange={setBox3DState}
            onSelectionUpdate={handleBox3DSelectionUpdate}
          />
          <SphereSelectionHandler
            sphereState={sphereState}
            onSphereUpdate={handleSphereUpdate}
            onSphereComplete={handleSelectionComplete}
          />
          <LassoSelectionHandler
            lassoPoints={lassoPoints}
            onLassoUpdate={setLassoPoints}
            onLassoComplete={handleSelectionComplete}
          />
          <ClickSelectionHandler />
          <WalkControls />
          <RapidLabelingController />
        </OrbitControlsContext.Provider>
      </Canvas>

      {/* Box mode instructions */}
      {mode === 'box' && (
        <div style={{
          position: 'absolute',
          bottom: 12,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '8px 16px',
          borderRadius: 4,
          fontSize: 12,
        }}>
          {box3DState.phase === 'none' && 'Left-click+drag to draw box | Right-drag: rotate camera | Middle-drag: pan'}
          {box3DState.phase === 'placing' && 'Drag to set box size'}
          {box3DState.phase === 'adjusting' && 'Corners: resize | Center: move | Rings: rotate | Enter: apply | Esc: cancel'}
        </div>
      )}

      {/* Lasso selection overlay */}
      {lassoPoints.length > 1 && (
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        >
          <polygon
            points={lassoPoints.map(p => `${p.x},${p.y}`).join(' ')}
            fill="rgba(255, 255, 255, 0.1)"
            stroke="rgba(255, 255, 255, 0.8)"
            strokeWidth="2"
          />
        </svg>
      )}

      {/* Rapid labeling UI */}
      {mode === 'rapid' && (
        <>
          {/* Progress bar - positioned below toolbar and voxel slider */}
          <div style={{
            position: 'absolute',
            top: 140,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0, 0, 0, 0.85)',
            padding: '12px 20px',
            borderRadius: 8,
            color: 'white',
            textAlign: 'center',
            minWidth: 200,
          }}>
            <div style={{ fontSize: 14, fontWeight: 'bold', marginBottom: 8 }}>
              Rapid Labeling
            </div>
            {!supervoxelIds ? (
              <div style={{ fontSize: 14, color: '#88f' }}>
                Computing supervoxels...
              </div>
            ) : (
              <div style={{ fontSize: 24, fontWeight: 'bold', color: '#ffff00' }}>
                {rapidCurrentIndex + 1} / {rapidProgress.unlabeled}
              </div>
            )}
            <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>
              {rapidProgress.labeled} labeled of {rapidProgress.total} total
            </div>
            {/* Progress bar */}
            <div style={{
              marginTop: 8,
              height: 6,
              background: 'rgba(255,255,255,0.2)',
              borderRadius: 3,
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${rapidProgress.total > 0 ? (rapidProgress.labeled / rapidProgress.total) * 100 : 0}%`,
                height: '100%',
                background: '#4ade80',
                transition: 'width 0.2s',
              }} />
            </div>
          </div>

          {/* Class legend on left side - only when supervoxels ready */}
          {supervoxelIds && rapidLabeling && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: 12,
              transform: 'translateY(-50%)',
              background: 'rgba(0, 0, 0, 0.85)',
              padding: '12px 16px',
              borderRadius: 8,
              color: 'white',
            }}>
              <div style={{ fontSize: 12, fontWeight: 'bold', marginBottom: 8, color: '#888' }}>
                Press key to label:
              </div>
              {[
                { key: '0', label: 'Background', color: '#444444' },
                { key: '1', label: 'Class 1', color: '#0000ff' },
                { key: '2', label: 'Class 2', color: '#00ffff' },
                { key: '3', label: 'Class 3', color: '#ff0000' },
                { key: '4', label: 'Class 4', color: '#00ff00' },
                { key: '5', label: 'Class 5', color: '#ffff00' },
                { key: '6', label: 'Class 6', color: '#ff8800' },
              ].map(({ key, label, color }) => (
                <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <span style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: 24,
                    height: 24,
                    background: '#333',
                    borderRadius: 4,
                    fontWeight: 'bold',
                    fontSize: 14,
                  }}>{key}</span>
                  <span style={{
                    display: 'inline-block',
                    width: 12,
                    height: 12,
                    background: color,
                    borderRadius: 2,
                  }} />
                  <span style={{ fontSize: 12 }}>{label}</span>
                </div>
              ))}
            </div>
          )}

          {/* Instructions at bottom */}
          <div style={{
            position: 'absolute',
            bottom: 12,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '8px 16px',
            borderRadius: 4,
            fontSize: 12,
          }}>
            {supervoxelIds && rapidLabeling
              ? '0-6: label patch | Arrow keys / Space: navigate | Esc: exit'
              : 'Waiting for supervoxels... Use slider above to adjust voxel size'}
          </div>
        </>
      )}
    </div>
  )
}
