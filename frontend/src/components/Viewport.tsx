import { useRef, useMemo, useEffect, useState, useCallback } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
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
  const { points, colors, numPoints, selectedIndices, supervoxelIds, labels, instanceIds, hideLabeledPoints } = usePointCloudStore()
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
  }, [selectedIndices, colors, numPoints, mode, supervoxelIds, labels, instanceIds, hideLabeledPoints, points])

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

// 3D Box selection with draggable corners
interface Box3DState {
  min: THREE.Vector3
  max: THREE.Vector3
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
  const dragOffset = useRef<THREE.Vector3>(new THREE.Vector3())

  // Calculate points inside box whenever it changes
  useEffect(() => {
    if (!boxState.isActive || !points || boxState.phase === 'none') return

    const indices: number[] = []
    const min = boxState.min
    const max = boxState.max

    for (let i = 0; i < numPoints; i++) {
      const x = points[i * 3]
      const y = points[i * 3 + 1]
      const z = points[i * 3 + 2]

      if (x >= min.x && x <= max.x &&
          y >= min.y && y <= max.y &&
          z >= min.z && z <= max.z) {
        indices.push(i)
      }
    }

    onSelectionUpdate(indices)
  }, [boxState.min, boxState.max, boxState.isActive, boxState.phase, points, numPoints, onSelectionUpdate])

  // Get handle positions (8 corners)
  const handles = useMemo(() => {
    if (!boxState.isActive) return []
    const { min, max } = boxState
    return [
      { id: 'min-min-min', pos: new THREE.Vector3(min.x, min.y, min.z) },
      { id: 'max-min-min', pos: new THREE.Vector3(max.x, min.y, min.z) },
      { id: 'min-max-min', pos: new THREE.Vector3(min.x, max.y, min.z) },
      { id: 'max-max-min', pos: new THREE.Vector3(max.x, max.y, min.z) },
      { id: 'min-min-max', pos: new THREE.Vector3(min.x, min.y, max.z) },
      { id: 'max-min-max', pos: new THREE.Vector3(max.x, min.y, max.z) },
      { id: 'min-max-max', pos: new THREE.Vector3(min.x, max.y, max.z) },
      { id: 'max-max-max', pos: new THREE.Vector3(max.x, max.y, max.z) },
    ]
  }, [boxState])

  // Handle initial box placement
  const handleCanvasClick = useCallback((e: MouseEvent) => {
    if (mode !== 'box' || e.button !== 0) return
    if (boxState.phase === 'adjusting') return // Don't start new box while adjusting

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )

    raycaster.setFromCamera(mouse, camera)

    // Find intersection point - either with a point or at a fixed distance
    let hitPoint: THREE.Vector3 | null = null

    if (points) {
      const hit = findClosestPointToRay(raycaster.ray, points, numPoints, 1.0)
      if (hit) {
        hitPoint = hit.position
      }
    }

    if (!hitPoint) {
      // Place at a fixed distance from camera
      hitPoint = new THREE.Vector3()
      raycaster.ray.at(10, hitPoint)
    }

    // Start placing mode
    onBoxChange({
      min: hitPoint.clone(),
      max: hitPoint.clone(),
      isActive: true,
      phase: 'placing',
    })
  }, [mode, boxState.phase, gl, camera, raycaster, points, numPoints, onBoxChange])

  // Handle mouse move for box placement and handle dragging
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'box') return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )

    raycaster.setFromCamera(mouse, camera)

    if (boxState.phase === 'placing') {
      // Extend box on the XZ plane at the initial Y level
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -boxState.min.y)
      const intersection = new THREE.Vector3()
      raycaster.ray.intersectPlane(plane, intersection)

      if (intersection) {
        const newMax = new THREE.Vector3(
          intersection.x,
          boxState.min.y + 2, // Give it some default height
          intersection.z
        )

        // Ensure min/max are correct
        const actualMin = new THREE.Vector3(
          Math.min(boxState.min.x, newMax.x),
          boxState.min.y,
          Math.min(boxState.min.z, newMax.z)
        )
        const actualMax = new THREE.Vector3(
          Math.max(boxState.min.x, newMax.x),
          newMax.y,
          Math.max(boxState.min.z, newMax.z)
        )

        onBoxChange({
          ...boxState,
          min: actualMin,
          max: actualMax,
        })
      }
    } else if (boxState.phase === 'adjusting' && dragHandle.current) {
      // Drag handle
      const intersection = new THREE.Vector3()
      raycaster.ray.intersectPlane(dragPlane.current, intersection)

      if (intersection) {
        const newPos = intersection.sub(dragOffset.current)
        const handleParts = dragHandle.current.split('-')

        const newMin = boxState.min.clone()
        const newMax = boxState.max.clone()

        // Update the appropriate coordinates based on which corner is being dragged
        if (handleParts[0] === 'min') newMin.x = Math.min(newPos.x, newMax.x - 0.1)
        else newMax.x = Math.max(newPos.x, newMin.x + 0.1)

        if (handleParts[1] === 'min') newMin.y = Math.min(newPos.y, newMax.y - 0.1)
        else newMax.y = Math.max(newPos.y, newMin.y + 0.1)

        if (handleParts[2] === 'min') newMin.z = Math.min(newPos.z, newMax.z - 0.1)
        else newMax.z = Math.max(newPos.z, newMin.z + 0.1)

        onBoxChange({
          ...boxState,
          min: newMin,
          max: newMax,
        })
      }
    }
  }, [mode, boxState, gl, camera, raycaster, onBoxChange])

  // Handle mouse up to finish placing
  const handleMouseUp = useCallback((e: MouseEvent) => {
    if (mode !== 'box') return

    if (boxState.phase === 'placing') {
      // Switch to adjusting mode
      onBoxChange({
        ...boxState,
        phase: 'adjusting',
      })
    }

    dragHandle.current = null
  }, [mode, boxState, onBoxChange])

  // Handle keyboard for apply/cancel
  useEffect(() => {
    if (mode !== 'box' || !boxState.isActive) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        // Apply selection (already updated via effect)
        onBoxChange({ min: new THREE.Vector3(), max: new THREE.Vector3(), isActive: false, phase: 'none' })
      } else if (e.key === 'Escape') {
        // Cancel
        onSelectionUpdate([])
        onBoxChange({ min: new THREE.Vector3(), max: new THREE.Vector3(), isActive: false, phase: 'none' })
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mode, boxState.isActive, onBoxChange, onSelectionUpdate])

  // Attach mouse listeners
  useEffect(() => {
    if (mode !== 'box') return

    const canvas = gl.domElement
    canvas.addEventListener('click', handleCanvasClick)
    canvas.addEventListener('mousemove', handleMouseMove)
    canvas.addEventListener('mouseup', handleMouseUp)

    return () => {
      canvas.removeEventListener('click', handleCanvasClick)
      canvas.removeEventListener('mousemove', handleMouseMove)
      canvas.removeEventListener('mouseup', handleMouseUp)
    }
  }, [mode, gl, handleCanvasClick, handleMouseMove, handleMouseUp])

  // Handle starting to drag a corner
  const startDragHandle = useCallback((handleId: string, handlePos: THREE.Vector3) => {
    if (boxState.phase !== 'adjusting') return

    dragHandle.current = handleId

    // Create a plane perpendicular to camera for dragging
    const cameraDir = new THREE.Vector3()
    camera.getWorldDirection(cameraDir)
    dragPlane.current.setFromNormalAndCoplanarPoint(cameraDir, handlePos)

    // Calculate offset from ray intersection to handle position
    const intersection = new THREE.Vector3()
    const mouse = new THREE.Vector2()
    // We'll update this on the next mouse move
    dragOffset.current.set(0, 0, 0)
  }, [boxState.phase, camera])

  if (mode !== 'box' || !boxState.isActive) return null

  const center = new THREE.Vector3().addVectors(boxState.min, boxState.max).multiplyScalar(0.5)
  const size = new THREE.Vector3().subVectors(boxState.max, boxState.min)

  return (
    <group>
      {/* Wireframe box */}
      <mesh position={center}>
        <boxGeometry args={[size.x, size.y, size.z]} />
        <meshBasicMaterial color="#00ffff" wireframe transparent opacity={0.5} />
      </mesh>

      {/* Semi-transparent faces */}
      <mesh position={center}>
        <boxGeometry args={[size.x, size.y, size.z]} />
        <meshBasicMaterial color="#00ffff" transparent opacity={0.1} side={THREE.DoubleSide} />
      </mesh>

      {/* Corner handles */}
      {boxState.phase === 'adjusting' && handles.map((handle) => (
        <mesh
          key={handle.id}
          position={handle.pos}
          onPointerDown={(e) => {
            e.stopPropagation()
            startDragHandle(handle.id, handle.pos)
          }}
          onPointerOver={() => { document.body.style.cursor = 'grab' }}
          onPointerOut={() => { document.body.style.cursor = 'default' }}
        >
          <sphereGeometry args={[0.15, 16, 16]} />
          <meshBasicMaterial color="#ffff00" />
        </mesh>
      ))}
    </group>
  )
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
  const { mode, navigationMode } = useSelectionStore()
  const { selectedIndices, setSelection, selectSupervoxelById } = usePointCloudStore()
  const canvasRef = useRef<HTMLDivElement>(null)

  // Handler for clicking on supervoxel hulls
  const handleHullClick = useCallback((supervoxelId: number, ctrlKey: boolean) => {
    selectSupervoxelById(supervoxelId, ctrlKey)
  }, [selectSupervoxelById])

  // 3D Box selection state
  const [box3DState, setBox3DState] = useState<Box3DState>({
    min: new THREE.Vector3(),
    max: new THREE.Vector3(),
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
  const orbitEnabled = navigationMode === 'orbit' && !(
    (mode === 'box' && box3DState.isActive) ||
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
          {box3DState.phase === 'none' && 'Click to place box corner'}
          {box3DState.phase === 'placing' && 'Drag to set box size, release to adjust'}
          {box3DState.phase === 'adjusting' && 'Drag corners to adjust | Enter to apply | Esc to cancel'}
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
    </div>
  )
}
