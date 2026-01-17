import { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { usePointCloudStore } from '../store/pointCloudStore'
import { useSelectionStore } from '../store/selectionStore'

function PointCloudMesh() {
  const meshRef = useRef<THREE.Points>(null)
  const { points, colors, numPoints, selectedIndices } = usePointCloudStore()

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()

    if (points && colors) {
      geo.setAttribute('position', new THREE.BufferAttribute(points, 3))

      // Normalize colors to 0-1 range
      const normalizedColors = new Float32Array(numPoints * 3)
      for (let i = 0; i < numPoints * 3; i++) {
        normalizedColors[i] = colors[i] / 255
      }
      geo.setAttribute('color', new THREE.BufferAttribute(normalizedColors, 3))
    }

    return geo
  }, [points, colors, numPoints])

  // Update colors when selection changes
  useEffect(() => {
    if (!meshRef.current || !colors) return

    const colorAttr = meshRef.current.geometry.getAttribute('color') as THREE.BufferAttribute
    if (!colorAttr) return

    const normalizedColors = colorAttr.array as Float32Array

    for (let i = 0; i < numPoints; i++) {
      if (selectedIndices.has(i)) {
        // Highlight selected points in white
        normalizedColors[i * 3] = 1
        normalizedColors[i * 3 + 1] = 1
        normalizedColors[i * 3 + 2] = 1
      } else {
        normalizedColors[i * 3] = colors[i * 3] / 255
        normalizedColors[i * 3 + 1] = colors[i * 3 + 1] / 255
        normalizedColors[i * 3 + 2] = colors[i * 3 + 2] / 255
      }
    }

    colorAttr.needsUpdate = true
  }, [selectedIndices, colors, numPoints])

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

export function Viewport() {
  const { mode } = useSelectionStore()
  const { selectedIndices, setSelection } = usePointCloudStore()
  const canvasRef = useRef<HTMLDivElement>(null)

  const [box, setBox] = useState<{
    start: { x: number; y: number } | null
    end: { x: number; y: number } | null
  }>({ start: null, end: null })
  const [isDragging, setIsDragging] = useState(false)

  const handleMouseDown = (e: React.MouseEvent) => {
    if (mode !== 'box' || e.button !== 0) return
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    setIsDragging(true)
    setBox({
      start: { x: e.clientX - rect.left, y: e.clientY - rect.top },
      end: { x: e.clientX - rect.left, y: e.clientY - rect.top },
    })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !box.start) return
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    setBox(prev => ({
      ...prev,
      end: { x: e.clientX - rect.left, y: e.clientY - rect.top },
    }))
  }

  const handleMouseUp = () => {
    // Selection completion is handled in BoxSelectionHandler
    setIsDragging(false)
    setBox({ start: null, end: null })
  }

  const handleSelectionComplete = (indices: number[], shiftKey: boolean, ctrlKey: boolean) => {
    const newSelection = new Set<number>(shiftKey ? selectedIndices : [])
    for (const i of indices) {
      if (ctrlKey) {
        newSelection.delete(i)
      } else {
        newSelection.add(i)
      }
    }
    setSelection(newSelection)
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
          enabled={mode !== 'box' || !isDragging}
          mouseButtons={{
            LEFT: mode === 'box' ? undefined : THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.ROTATE,
          }}
        />
        <PointCloudMesh />
        <BoxSelectionHandler
          isDragging={isDragging}
          box={box}
          onComplete={handleSelectionComplete}
        />
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
    </div>
  )
}
