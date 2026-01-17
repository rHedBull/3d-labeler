import { useRef, useMemo, useEffect, useState, useCallback } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
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

function PointCloudMesh() {
  const meshRef = useRef<THREE.Points>(null)
  const { points, colors, numPoints, selectedIndices, supervoxelIds, labels, hideLabeledPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()

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

    for (let i = 0; i < numPoints; i++) {
      // Hide labeled points by moving them far away
      const isLabeled = labels && labels[i] > 0
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
        // Highlight selected points in white
        normalizedColors[i * 3] = 1
        normalizedColors[i * 3 + 1] = 1
        normalizedColors[i * 3 + 2] = 1
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
  }, [selectedIndices, colors, numPoints, mode, supervoxelIds, labels, hideLabeledPoints, points])

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

// Component inside Canvas to handle box selection projection
function BoxSelectionHandler({
  isDragging,
  box,
  onComplete,
}: {
  isDragging: boolean
  box: { start: { x: number; y: number } | null; end: { x: number; y: number } | null }
  onComplete: (indices: number[], shiftKey: boolean, ctrlKey: boolean) => void
}) {
  const { camera, size } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const prevDragging = useRef(isDragging)
  const lastEvent = useRef<{ shiftKey: boolean; ctrlKey: boolean }>({ shiftKey: false, ctrlKey: false })

  useEffect(() => {
    // Detect end of drag
    if (prevDragging.current && !isDragging && box.start && box.end && points) {
      const toNDC = (x: number, y: number) => ({
        x: (x / size.width) * 2 - 1,
        y: -(y / size.height) * 2 + 1,
      })

      const startNDC = toNDC(box.start.x, box.start.y)
      const endNDC = toNDC(box.end.x, box.end.y)

      const minX = Math.min(startNDC.x, endNDC.x)
      const maxX = Math.max(startNDC.x, endNDC.x)
      const minY = Math.min(startNDC.y, endNDC.y)
      const maxY = Math.max(startNDC.y, endNDC.y)

      const projScreenMatrix = new THREE.Matrix4()
      projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)

      const selectedIndices: number[] = []
      const point = new THREE.Vector3()

      for (let i = 0; i < numPoints; i++) {
        point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
        point.applyMatrix4(projScreenMatrix)

        if (point.z < 1 && point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY) {
          selectedIndices.push(i)
        }
      }

      onComplete(selectedIndices, lastEvent.current.shiftKey, lastEvent.current.ctrlKey)
    }
    prevDragging.current = isDragging
  }, [isDragging, box, points, numPoints, camera, size, onComplete])

  // Store modifier keys on mouseup
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      lastEvent.current = { shiftKey: e.shiftKey, ctrlKey: e.ctrlKey }
    }
    window.addEventListener('mouseup', handler)
    return () => window.removeEventListener('mouseup', handler)
  }, [])

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
  const { mode } = useSelectionStore()
  const { selectedIndices, setSelection, selectSupervoxelById } = usePointCloudStore()
  const canvasRef = useRef<HTMLDivElement>(null)

  // Handler for clicking on supervoxel hulls
  const handleHullClick = useCallback((supervoxelId: number, ctrlKey: boolean) => {
    selectSupervoxelById(supervoxelId, ctrlKey)
  }, [selectSupervoxelById])

  // Box selection state
  const [box, setBox] = useState<{
    start: { x: number; y: number } | null
    end: { x: number; y: number } | null
  }>({ start: null, end: null })
  const [isDragging, setIsDragging] = useState(false)

  // Sphere selection state
  const [sphereState, setSphereState] = useState<{
    center: THREE.Vector3 | null
    radius: number
    isDragging: boolean
  }>({ center: null, radius: 0, isDragging: false })

  // Lasso selection state
  const [lassoPoints, setLassoPoints] = useState<{ x: number; y: number }[]>([])

  const handleMouseDown = (e: React.MouseEvent) => {
    if (mode === 'box' && e.button === 0) {
      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return
      setIsDragging(true)
      setBox({
        start: { x: e.clientX - rect.left, y: e.clientY - rect.top },
        end: { x: e.clientX - rect.left, y: e.clientY - rect.top },
      })
    } else if (mode === 'sphere' && e.button === 0) {
      // Sphere dragging is started when center is set
      setSphereState(prev => ({ ...prev, isDragging: true }))
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (mode === 'box' && isDragging && box.start) {
      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return
      setBox(prev => ({
        ...prev,
        end: { x: e.clientX - rect.left, y: e.clientY - rect.top },
      }))
    }
  }

  const handleMouseUp = () => {
    if (mode === 'box') {
      setIsDragging(false)
      setBox({ start: null, end: null })
    } else if (mode === 'sphere') {
      // Sphere completion is handled in SphereSelectionHandler
    }
  }

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
  const orbitEnabled = !(
    (mode === 'box' && isDragging) ||
    (mode === 'sphere' && sphereState.isDragging) ||
    (mode === 'lasso' && lassoPoints.length > 0)
  )

  // Determine mouse button behavior
  const getLeftMouseAction = () => {
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
          enableDamping
          dampingFactor={0.05}
          enabled={orbitEnabled}
          mouseButtons={{
            LEFT: getLeftMouseAction(),
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.ROTATE,
          }}
        />
        <PointCloudMesh />
        <SupervoxelHulls onHullClick={handleHullClick} />
        <BoxSelectionHandler
          isDragging={isDragging}
          box={box}
          onComplete={handleSelectionComplete}
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
      </Canvas>

      {/* Selection box overlay */}
      {isDragging && box.start && box.end && (
        <div
          style={{
            position: 'absolute',
            left: Math.min(box.start.x, box.end.x),
            top: Math.min(box.start.y, box.end.y),
            width: Math.abs(box.end.x - box.start.x),
            height: Math.abs(box.end.y - box.start.y),
            border: '2px solid rgba(255, 255, 255, 0.8)',
            background: 'rgba(255, 255, 255, 0.1)',
            pointerEvents: 'none',
          }}
        />
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
    </div>
  )
}
