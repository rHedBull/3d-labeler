import { useMemo } from 'react'
import * as THREE from 'three'
import { useSelectionStore } from '../store/selectionStore'
import { useFittingStore } from '../store/fittingStore'

// Color palette for distinguishing candidates
const COLORS = [
  '#ff6b6b',
  '#4ecdc4',
  '#45b7d1',
  '#96ceb4',
  '#ffeaa7',
  '#dfe6e9',
  '#fd79a8',
  '#a29bfe',
  '#00b894',
  '#e17055',
]

/**
 * FittedPrimitivesView renders fitted cylinders and boxes as wireframe meshes
 * in the 3D viewport during the 'selecting' phase of primitive fitting.
 *
 * Each candidate is rendered with:
 * - A distinct color from the palette (cycles if more candidates than colors)
 * - Higher opacity (0.6) if accepted, lower (0.25) if not accepted
 * - Wireframe mesh for clear visibility
 * - Semi-transparent fill for spatial understanding
 */
export function FittedPrimitivesView() {
  const { mode } = useSelectionStore()
  const {
    cylinderPhase,
    boxPhase,
    fittedCylinders,
    fittedBoxes,
  } = useFittingStore()

  // Only show during 'selecting' phase
  const showCylinders = mode === 'cylinder-fit' && cylinderPhase === 'selecting'
  const showBoxes = mode === 'box-fit' && boxPhase === 'selecting'

  // Compute cylinder rotations from axis vectors
  const cylinderRotations = useMemo(() => {
    if (!showCylinders) return []

    return fittedCylinders.map(cylinder => {
      // Default cylinder is aligned with Y axis, rotate to match fitted axis
      const defaultAxis = new THREE.Vector3(0, 1, 0)
      const quaternion = new THREE.Quaternion()
      quaternion.setFromUnitVectors(defaultAxis, cylinder.axis.clone().normalize())
      const euler = new THREE.Euler()
      euler.setFromQuaternion(quaternion)
      return euler
    })
  }, [showCylinders, fittedCylinders])

  if (!showCylinders && !showBoxes) {
    return null
  }

  return (
    <group>
      {/* Render fitted cylinders */}
      {showCylinders && fittedCylinders.map((cylinder, index) => {
        const color = COLORS[index % COLORS.length]
        const wireframeOpacity = cylinder.accepted ? 0.6 : 0.25
        const fillOpacity = cylinder.accepted ? 0.15 : 0.05
        const rotation = cylinderRotations[index]

        return (
          <group key={`cylinder-${cylinder.id}`} position={cylinder.center}>
            {/* Wireframe cylinder */}
            <mesh rotation={rotation}>
              <cylinderGeometry
                args={[
                  cylinder.radius,
                  cylinder.radius,
                  cylinder.height,
                  32,
                  1,
                  true,
                ]}
              />
              <meshBasicMaterial
                color={color}
                wireframe
                transparent
                opacity={wireframeOpacity}
              />
            </mesh>

            {/* Semi-transparent fill */}
            <mesh rotation={rotation}>
              <cylinderGeometry
                args={[
                  cylinder.radius,
                  cylinder.radius,
                  cylinder.height,
                  32,
                ]}
              />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={fillOpacity}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>

            {/* Top and bottom caps for better visibility */}
            <mesh rotation={rotation} position={[0, cylinder.height / 2, 0]}>
              <circleGeometry args={[cylinder.radius, 32]} />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={fillOpacity}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
            <mesh rotation={rotation} position={[0, -cylinder.height / 2, 0]}>
              <circleGeometry args={[cylinder.radius, 32]} />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={fillOpacity}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
          </group>
        )
      })}

      {/* Render fitted boxes */}
      {showBoxes && fittedBoxes.map((box, index) => {
        const color = COLORS[index % COLORS.length]
        const wireframeOpacity = box.accepted ? 0.6 : 0.25
        const fillOpacity = box.accepted ? 0.15 : 0.05

        return (
          <group key={`box-${box.id}`} position={box.center}>
            {/* Wireframe box */}
            <mesh rotation={box.rotation}>
              <boxGeometry args={[box.size.x, box.size.y, box.size.z]} />
              <meshBasicMaterial
                color={color}
                wireframe
                transparent
                opacity={wireframeOpacity}
              />
            </mesh>

            {/* Semi-transparent fill */}
            <mesh rotation={box.rotation}>
              <boxGeometry args={[box.size.x, box.size.y, box.size.z]} />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={fillOpacity}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}
