import { useCallback, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useThree, useFrame } from '@react-three/fiber'
import { useSelectionStore } from '../store/selectionStore'
import { useFittingStore, type FittedCylinder } from '../store/fittingStore'
import { usePointCloudStore } from '../store/pointCloudStore'
import { fitCylinders } from '../lib/api'

interface CylinderFitHandlerProps {
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
 * CylinderFitHandler component handles the 3-click workflow for defining
 * a cylinder region and fitting cylinders to the point cloud.
 *
 * Workflow:
 * 1. Click 1: Place center point (raycast to nearest point in cloud)
 * 2. Move mouse: Preview radius (distance from center on a plane)
 * 3. Click 2: Lock radius
 * 4. Move mouse: Preview height along axis
 * 5. Click 3: Lock height and trigger fitting API call
 */
export function CylinderFitHandler({ onCandidatesReady }: CylinderFitHandlerProps) {
  const { camera, raycaster, gl } = useThree()
  const { points, numPoints } = usePointCloudStore()
  const { mode } = useSelectionStore()
  const {
    cylinderPhase,
    cylinderCenter,
    cylinderAxis,
    cylinderRadius,
    cylinderHeight,
    tolerance,
    minInliers,
    setCylinderPhase,
    setCylinderCenter,
    setCylinderAxis,
    setCylinderRadius,
    setCylinderHeight,
    setFittedCylinders,
    resetCylinder,
  } = useFittingStore()

  // Refs for preview during mouse movement
  const previewRadiusRef = useRef<number>(0)
  const previewHeightRef = useRef<number>(0)
  const radiusPlaneRef = useRef<THREE.Plane>(new THREE.Plane())
  const heightPlaneRef = useRef<THREE.Plane>(new THREE.Plane())

  // Initialize cylinder fitting when mode changes to cylinder-fit
  useEffect(() => {
    if (mode === 'cylinder-fit') {
      // Start with center phase if not already in progress
      if (cylinderPhase === 'none') {
        setCylinderPhase('center')
      }
    } else {
      // Reset when leaving cylinder-fit mode
      if (cylinderPhase !== 'none') {
        resetCylinder()
      }
    }
  }, [mode, cylinderPhase, setCylinderPhase, resetCylinder])

  // Handle Escape key to cancel/reset
  useEffect(() => {
    if (mode !== 'cylinder-fit') return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        resetCylinder()
        setCylinderPhase('center')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mode, resetCylinder, setCylinderPhase])

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

  // Handle click events for the 3-click workflow
  const handleClick = useCallback(async (e: MouseEvent) => {
    if (mode !== 'cylinder-fit' || e.button !== 0 || !points) return

    const rect = gl.domElement.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    )
    raycaster.setFromCamera(mouse, camera)

    if (cylinderPhase === 'center') {
      // Click 1: Place center point
      const hit = findClosestPointToRay(raycaster.ray, points, numPoints, 1.0)
      if (hit) {
        setCylinderCenter(hit.position)

        // Default axis is camera up direction
        const cameraUp = new THREE.Vector3(0, 1, 0)
        cameraUp.applyQuaternion(camera.quaternion)
        setCylinderAxis(cameraUp.normalize())

        // Create plane perpendicular to axis for radius adjustment
        radiusPlaneRef.current.setFromNormalAndCoplanarPoint(cameraUp, hit.position)

        setCylinderPhase('radius')
      }
    } else if (cylinderPhase === 'radius') {
      // Click 2: Lock radius
      if (previewRadiusRef.current > 0.01) {
        setCylinderRadius(previewRadiusRef.current)

        // Create plane for height adjustment (perpendicular to view direction, through center)
        if (cylinderCenter && cylinderAxis) {
          const viewDir = new THREE.Vector3()
          camera.getWorldDirection(viewDir)
          heightPlaneRef.current.setFromNormalAndCoplanarPoint(viewDir, cylinderCenter)
        }

        setCylinderPhase('height')
      }
    } else if (cylinderPhase === 'height') {
      // Click 3: Lock height and trigger fitting
      if (previewHeightRef.current > 0.01) {
        setCylinderHeight(previewHeightRef.current)
        setCylinderPhase('fitting')

        // Trigger API call
        if (cylinderCenter && cylinderAxis) {
          try {
            const center: [number, number, number] = [
              cylinderCenter.x,
              cylinderCenter.y,
              cylinderCenter.z,
            ]
            const axis: [number, number, number] = [
              cylinderAxis.x,
              cylinderAxis.y,
              cylinderAxis.z,
            ]

            const candidates = await fitCylinders(
              center,
              axis,
              previewHeightRef.current > 0 ? previewHeightRef.current : cylinderRadius,
              previewHeightRef.current,
              tolerance,
              minInliers
            )

            // Convert API response to FittedCylinder format
            const fittedCylinders: FittedCylinder[] = candidates.map(c => ({
              id: c.id,
              center: new THREE.Vector3(c.center[0], c.center[1], c.center[2]),
              axis: new THREE.Vector3(c.axis[0], c.axis[1], c.axis[2]),
              radius: c.radius,
              height: c.height,
              pointIndices: Array.from(c.pointIndices),
              accepted: true, // Default to accepted
            }))

            setFittedCylinders(fittedCylinders)
            setCylinderPhase('selecting')
            onCandidatesReady()
          } catch (error) {
            console.error('Failed to fit cylinders:', error)
            setCylinderPhase('center')
          }
        }
      }
    }
  }, [
    mode, points, numPoints, camera, raycaster, gl,
    cylinderPhase, cylinderCenter, cylinderAxis, cylinderRadius,
    tolerance, minInliers,
    setCylinderPhase, setCylinderCenter, setCylinderAxis,
    setCylinderRadius, setCylinderHeight, setFittedCylinders,
    onCandidatesReady
  ])

  // Handle mouse move for preview
  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (mode !== 'cylinder-fit') return

    if (cylinderPhase === 'radius' && cylinderCenter) {
      // Calculate radius as distance from center to mouse on the plane
      const intersection = getPlaneIntersection(e, radiusPlaneRef.current)
      if (intersection) {
        previewRadiusRef.current = cylinderCenter.distanceTo(intersection)
      }
    } else if (cylinderPhase === 'height' && cylinderCenter && cylinderAxis) {
      // Calculate height by projecting mouse movement along axis
      const intersection = getPlaneIntersection(e, heightPlaneRef.current)
      if (intersection) {
        // Project the vector from center to intersection onto the axis
        const toIntersection = intersection.clone().sub(cylinderCenter)
        const projectedLength = Math.abs(toIntersection.dot(cylinderAxis))
        previewHeightRef.current = projectedLength * 2 // Double for full height (above and below center)
      }
    }
  }, [mode, cylinderPhase, cylinderCenter, cylinderAxis, getPlaneIntersection])

  // Attach event listeners
  useEffect(() => {
    if (mode !== 'cylinder-fit') return

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

  // Calculate cylinder geometry for preview
  const cylinderGeometry = useMemo(() => {
    if (!cylinderCenter || !cylinderAxis) return null

    let radius = cylinderRadius
    let height = cylinderHeight

    // Use preview values during adjustment phases
    if (cylinderPhase === 'radius') {
      radius = previewRadiusRef.current || 0.1
      height = 0.1 // Minimal height during radius adjustment
    } else if (cylinderPhase === 'height') {
      radius = cylinderRadius || previewRadiusRef.current || 0.1
      height = previewHeightRef.current || 0.1
    }

    if (radius < 0.01 || height < 0.01) return null

    return { radius, height }
  }, [cylinderCenter, cylinderAxis, cylinderRadius, cylinderHeight, cylinderPhase])

  // Calculate rotation to align cylinder with axis
  const cylinderRotation = useMemo(() => {
    if (!cylinderAxis) return new THREE.Euler()

    // Default cylinder is aligned with Y axis, rotate to match our axis
    const defaultAxis = new THREE.Vector3(0, 1, 0)
    const quaternion = new THREE.Quaternion()
    quaternion.setFromUnitVectors(defaultAxis, cylinderAxis.clone().normalize())
    const euler = new THREE.Euler()
    euler.setFromQuaternion(quaternion)
    return euler
  }, [cylinderAxis])

  // Don't render if not in cylinder-fit mode or no center set
  if (mode !== 'cylinder-fit') return null
  if (!cylinderCenter) return null
  if (cylinderPhase === 'fitting' || cylinderPhase === 'selecting' || cylinderPhase === 'none') return null

  return (
    <group position={cylinderCenter}>
      {/* Center point marker */}
      <mesh>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color="#ff0000" />
      </mesh>

      {/* Cylinder wireframe preview */}
      {cylinderGeometry && (
        <mesh rotation={cylinderRotation}>
          <cylinderGeometry args={[
            cylinderGeometry.radius,
            cylinderGeometry.radius,
            cylinderGeometry.height,
            32,
            1,
            true
          ]} />
          <meshBasicMaterial
            color="#00ffff"
            wireframe
            transparent
            opacity={0.6}
          />
        </mesh>
      )}

      {/* Semi-transparent cylinder fill */}
      {cylinderGeometry && (
        <mesh rotation={cylinderRotation}>
          <cylinderGeometry args={[
            cylinderGeometry.radius,
            cylinderGeometry.radius,
            cylinderGeometry.height,
            32
          ]} />
          <meshBasicMaterial
            color="#00ffff"
            transparent
            opacity={0.15}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      )}

      {/* Axis indicator line */}
      {cylinderAxis && cylinderGeometry && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([
                0, -cylinderGeometry.height / 2 - 0.5, 0,
                0, cylinderGeometry.height / 2 + 0.5, 0,
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#ffff00" linewidth={2} />
        </line>
      )}
    </group>
  )
}
