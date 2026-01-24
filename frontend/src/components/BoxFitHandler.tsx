import { useCallback, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useThree, useFrame } from '@react-three/fiber'
import { useSelectionStore } from '../store/selectionStore'
import { useFittingStore, type FittedBox } from '../store/fittingStore'
import { usePointCloudStore } from '../store/pointCloudStore'
import { fitBoxes } from '../lib/api'

interface BoxFitHandlerProps {
  onCandidatesReady: () => void
}

/**
 * Find the point closest to a ray (for click-based selection on point cloud)
 */
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

/**
 * BoxFitHandler component handles the 4-click workflow for defining
 * a box region and fitting boxes to the point cloud.
 *
 * Workflow:
 * 1. Click 1: Place first corner (raycast to nearest point in cloud)
 * 2. Click 2: Place second corner (defines one edge)
 * 3. Click 3: Place third corner (defines base rectangle, 4th corner inferred)
 * 4. Move mouse: Preview height along normal
 * 5. Click 4: Lock height and trigger fitting API call
 */
export function BoxFitHandler({ onCandidatesReady }: BoxFitHandlerProps) {
  const { camera, raycaster, gl } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()
  const {
    boxPhase,
    boxCorners,
    boxHeight,
    tolerance,
    minInliers,
    setBoxPhase,
    addBoxCorner,
    setBoxHeight,
    setFittedBoxes,
    resetBox,
  } = useFittingStore()

  // Refs for preview during mouse movement
  const previewHeightRef = useRef<number>(0)
  const heightPlaneRef = useRef<THREE.Plane>(new THREE.Plane())
  const baseNormalRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 1, 0))

  // Initialize box fitting when mode changes to box-fit
  useEffect(() => {
    if (mode === 'box-fit') {
      // Start with corner1 phase if not already in progress
      if (boxPhase === 'none') {
        setBoxPhase('corner1')
      }
    } else {
      // Reset when leaving box-fit mode
      if (boxPhase !== 'none') {
        resetBox()
      }
    }
  }, [mode, boxPhase, setBoxPhase, resetBox])

  // Handle Escape key to cancel/reset
  useEffect(() => {
    if (mode !== 'box-fit') return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        resetBox()
        setBoxPhase('corner1')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mode, resetBox, setBoxPhase])

  // Get mouse position on a plane
  const getPlaneIntersection = useCallback((e: MouseEvent, plane: THREE.Plane): THREE.Vector3 | null => {
    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)
    const intersection = new THREE.Vector3()
    const result = raycaster.ray.intersectPlane(plane, intersection)
    return result ? intersection : null
  }, [gl, camera, raycaster])

  // Compute base normal from cross product of edges (when we have 3 corners)
  const computeBaseNormal = useCallback((corners: THREE.Vector3[]): THREE.Vector3 => {
    if (corners.length < 3) {
      return new THREE.Vector3(0, 1, 0)
    }
    const edge1 = corners[1].clone().sub(corners[0])
    const edge2 = corners[2].clone().sub(corners[0])
    const normal = edge1.cross(edge2).normalize()
    // Ensure normal points in a consistent direction (towards camera)
    const cameraDir = new THREE.Vector3()
    camera.getWorldDirection(cameraDir)
    if (normal.dot(cameraDir) > 0) {
      normal.negate()
    }
    return normal
  }, [camera])

  // Handle click events for the 4-click workflow
  const handleClick = useCallback(async (e: MouseEvent) => {
    if (mode !== 'box-fit' || e.button !== 0 || !points) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)

    if (boxPhase === 'corner1') {
      // Click 1: Place first corner
      const hit = findClosestPointToRay(raycaster.ray, points, numPoints, 1.0)
      if (hit) {
        addBoxCorner(hit.position)
        setBoxPhase('corner2')
      }
    } else if (boxPhase === 'corner2') {
      // Click 2: Place second corner (defines one edge)
      const hit = findClosestPointToRay(raycaster.ray, points, numPoints, 1.0)
      if (hit) {
        addBoxCorner(hit.position)
        setBoxPhase('corner3')
      }
    } else if (boxPhase === 'corner3') {
      // Click 3: Place third corner (defines base rectangle)
      const hit = findClosestPointToRay(raycaster.ray, points, numPoints, 1.0)
      if (hit) {
        addBoxCorner(hit.position)

        // Compute base normal for height adjustment
        const cornersWithNew = [...boxCorners, hit.position]
        baseNormalRef.current = computeBaseNormal(cornersWithNew)

        // Create plane for height adjustment (perpendicular to view direction, through center)
        const viewDir = new THREE.Vector3()
        camera.getWorldDirection(viewDir)
        const center = cornersWithNew[0].clone()
          .add(cornersWithNew[1])
          .add(cornersWithNew[2])
          .divideScalar(3)
        heightPlaneRef.current.setFromNormalAndCoplanarPoint(viewDir, center)

        setBoxPhase('height')
      }
    } else if (boxPhase === 'height') {
      // Click 4: Lock height and trigger fitting
      if (previewHeightRef.current > 0.01) {
        setBoxHeight(previewHeightRef.current)
        setBoxPhase('fitting')

        // Trigger API call
        if (boxCorners.length >= 3) {
          try {
            const corner1: [number, number, number] = [
              boxCorners[0].x,
              boxCorners[0].y,
              boxCorners[0].z,
            ]
            const corner2: [number, number, number] = [
              boxCorners[1].x,
              boxCorners[1].y,
              boxCorners[1].z,
            ]
            const corner3: [number, number, number] = [
              boxCorners[2].x,
              boxCorners[2].y,
              boxCorners[2].z,
            ]

            const candidates = await fitBoxes(
              corner1,
              corner2,
              corner3,
              previewHeightRef.current,
              tolerance,
              minInliers
            )

            // Convert API response to FittedBox format
            const fittedBoxes: FittedBox[] = candidates.map(c => ({
              id: c.id,
              center: new THREE.Vector3(c.center[0], c.center[1], c.center[2]),
              size: new THREE.Vector3(c.size[0], c.size[1], c.size[2]),
              rotation: new THREE.Euler(c.rotation[0], c.rotation[1], c.rotation[2]),
              pointIndices: Array.from(c.pointIndices),
              accepted: true, // Default to accepted
            }))

            setFittedBoxes(fittedBoxes)
            setBoxPhase('selecting')
            onCandidatesReady()
          } catch (error) {
            console.error('Failed to fit boxes:', error)
            setBoxPhase('corner1')
          }
        }
      }
    }
  }, [
    mode, points, numPoints, camera, raycaster, gl,
    boxPhase, boxCorners, tolerance, minInliers,
    setBoxPhase, addBoxCorner, setBoxHeight, setFittedBoxes,
    computeBaseNormal, onCandidatesReady
  ])

  // Handle mouse move for preview
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'box-fit') return

    if (boxPhase === 'height' && boxCorners.length >= 3) {
      // Calculate height by projecting mouse movement along base normal
      const intersection = getPlaneIntersection(e, heightPlaneRef.current)
      if (intersection) {
        // Calculate center of base triangle
        const baseCenter = boxCorners[0].clone()
          .add(boxCorners[1])
          .add(boxCorners[2])
          .divideScalar(3)

        // Project the vector from center to intersection onto the base normal
        const toIntersection = intersection.clone().sub(baseCenter)
        const projectedLength = Math.abs(toIntersection.dot(baseNormalRef.current))
        previewHeightRef.current = projectedLength
      }
    }
  }, [mode, boxPhase, boxCorners, getPlaneIntersection])

  // Attach event listeners
  useEffect(() => {
    if (mode !== 'box-fit') return

    const canvas = gl.domElement
    canvas.addEventListener('click', handleClick)
    canvas.addEventListener('mousemove', handleMouseMove)

    return () => {
      canvas.removeEventListener('click', handleClick)
      canvas.removeEventListener('mousemove', handleMouseMove)
    }
  }, [mode, gl, handleClick, handleMouseMove])

  // Update preview values each frame for smooth rendering
  useFrame(() => {
    // This ensures the preview geometry updates smoothly
  })

  // Compute 4th corner from 3 corners: corner4 = corner1 + (corner2 - corner1) + (corner3 - corner1)
  // Simplified: corner4 = corner2 + corner3 - corner1
  const corner4 = useMemo(() => {
    if (boxCorners.length < 3) return null
    return boxCorners[1].clone().add(boxCorners[2]).sub(boxCorners[0])
  }, [boxCorners])

  // Calculate box geometry for preview
  const boxGeometry = useMemo(() => {
    if (boxCorners.length < 3) return null

    let height = boxHeight
    if (boxPhase === 'height') {
      height = previewHeightRef.current || 0.1
    }

    if (height < 0.01) return null

    // Calculate dimensions
    const edge1 = boxCorners[1].clone().sub(boxCorners[0])
    const edge2 = boxCorners[2].clone().sub(boxCorners[0])
    const width = edge1.length()
    const depth = edge2.length()

    // Calculate center
    const c4 = boxCorners[1].clone().add(boxCorners[2]).sub(boxCorners[0])
    const baseCenter = boxCorners[0].clone()
      .add(boxCorners[1])
      .add(boxCorners[2])
      .add(c4)
      .divideScalar(4)
    const normal = baseNormalRef.current
    const center = baseCenter.clone().add(normal.clone().multiplyScalar(height / 2))

    // Calculate rotation to align box with edges
    const xAxis = edge1.clone().normalize()
    const yAxis = normal.clone().normalize()
    const zAxis = xAxis.clone().cross(yAxis).normalize()

    const rotationMatrix = new THREE.Matrix4()
    rotationMatrix.makeBasis(xAxis, yAxis, zAxis)
    const euler = new THREE.Euler()
    euler.setFromRotationMatrix(rotationMatrix)

    return { width, depth, height, center, rotation: euler }
  }, [boxCorners, boxHeight, boxPhase])

  // Don't render if not in box-fit mode
  if (mode !== 'box-fit') return null
  if (boxPhase === 'fitting' || boxPhase === 'selecting' || boxPhase === 'none') return null

  return (
    <group>
      {/* Corner point markers */}
      {boxCorners.map((corner, i) => (
        <mesh key={i} position={corner}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshBasicMaterial color="#ff0000" />
        </mesh>
      ))}

      {/* Line for 2 corners */}
      {boxCorners.length === 2 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([
                boxCorners[0].x, boxCorners[0].y, boxCorners[0].z,
                boxCorners[1].x, boxCorners[1].y, boxCorners[1].z,
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ffff" linewidth={2} />
        </line>
      )}

      {/* Wireframe base rectangle for 3+ corners */}
      {boxCorners.length >= 3 && corner4 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={5}
              array={new Float32Array([
                boxCorners[0].x, boxCorners[0].y, boxCorners[0].z,
                boxCorners[1].x, boxCorners[1].y, boxCorners[1].z,
                corner4.x, corner4.y, corner4.z,
                boxCorners[2].x, boxCorners[2].y, boxCorners[2].z,
                boxCorners[0].x, boxCorners[0].y, boxCorners[0].z,
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ffff" linewidth={2} />
        </line>
      )}

      {/* Wireframe box preview with height */}
      {boxGeometry && (
        <mesh position={boxGeometry.center} rotation={boxGeometry.rotation}>
          <boxGeometry args={[boxGeometry.width, boxGeometry.height, boxGeometry.depth]} />
          <meshBasicMaterial
            color="#00ffff"
            wireframe
            transparent
            opacity={0.6}
          />
        </mesh>
      )}

      {/* Semi-transparent box fill */}
      {boxGeometry && (
        <mesh position={boxGeometry.center} rotation={boxGeometry.rotation}>
          <boxGeometry args={[boxGeometry.width, boxGeometry.height, boxGeometry.depth]} />
          <meshBasicMaterial
            color="#00ffff"
            transparent
            opacity={0.15}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* Height indicator line */}
      {boxCorners.length >= 3 && boxGeometry && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={(() => {
                const baseCenter = boxCorners[0].clone()
                  .add(boxCorners[1])
                  .add(boxCorners[2])
                  .add(corner4!)
                  .divideScalar(4)
                const normal = baseNormalRef.current
                const topPoint = baseCenter.clone().add(normal.clone().multiplyScalar(boxGeometry.height + 0.5))
                const bottomPoint = baseCenter.clone().sub(normal.clone().multiplyScalar(0.5))
                return new Float32Array([
                  bottomPoint.x, bottomPoint.y, bottomPoint.z,
                  topPoint.x, topPoint.y, topPoint.z,
                ])
              })()}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#ffff00" linewidth={2} />
        </line>
      )}
    </group>
  )
}
