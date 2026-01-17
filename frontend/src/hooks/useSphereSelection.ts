import { useCallback, useState } from 'react'
import * as THREE from 'three'
import { useThree } from '@react-three/fiber'
import { usePointCloudStore } from '../store/pointCloudStore'

export function useSphereSelection() {
  const { camera, raycaster } = useThree()
  const { points, numPoints, selectedIndices, setSelection } = usePointCloudStore()

  const [sphereCenter, setSphereCenter] = useState<THREE.Vector3 | null>(null)
  const [sphereRadius, setSphereRadius] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  const findClickedPoint = useCallback((event: MouseEvent, canvas: HTMLCanvasElement) => {
    if (!points) return null

    const rect = canvas.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    )

    raycaster.setFromCamera(mouse, camera)

    // Find closest point to ray
    let closestDist = Infinity
    let closestIdx = -1
    const ray = raycaster.ray
    const point = new THREE.Vector3()

    for (let i = 0; i < numPoints; i++) {
      point.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2])
      const dist = ray.distanceToPoint(point)

      if (dist < closestDist && dist < 0.5) { // 0.5 threshold
        closestDist = dist
        closestIdx = i
      }
    }

    if (closestIdx >= 0) {
      return new THREE.Vector3(
        points[closestIdx * 3],
        points[closestIdx * 3 + 1],
        points[closestIdx * 3 + 2]
      )
    }

    return null
  }, [points, numPoints, camera, raycaster])

  const selectPointsInSphere = useCallback((center: THREE.Vector3, radius: number, shiftKey: boolean, ctrlKey: boolean) => {
    if (!points) return

    const newSelection = new Set<number>(shiftKey ? selectedIndices : [])
    const radiusSq = radius * radius

    for (let i = 0; i < numPoints; i++) {
      const dx = points[i * 3] - center.x
      const dy = points[i * 3 + 1] - center.y
      const dz = points[i * 3 + 2] - center.z
      const distSq = dx * dx + dy * dy + dz * dz

      if (distSq <= radiusSq) {
        if (ctrlKey) {
          newSelection.delete(i)
        } else {
          newSelection.add(i)
        }
      }
    }

    setSelection(newSelection)
  }, [points, numPoints, selectedIndices, setSelection])

  return {
    sphereCenter,
    sphereRadius,
    isDragging,
    setSphereCenter,
    setSphereRadius,
    setIsDragging,
    findClickedPoint,
    selectPointsInSphere,
  }
}
