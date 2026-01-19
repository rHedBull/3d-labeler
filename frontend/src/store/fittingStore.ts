import { create } from 'zustand'
import * as THREE from 'three'

export interface FittedCylinder {
  id: number
  center: THREE.Vector3
  axis: THREE.Vector3
  radius: number
  height: number
  pointIndices: number[]
  accepted: boolean
}

export interface FittedBox {
  id: number
  center: THREE.Vector3
  size: THREE.Vector3
  rotation: THREE.Euler
  pointIndices: number[]
  accepted: boolean
}

interface CylinderRegion {
  center: THREE.Vector3
  axis: THREE.Vector3
  radius: number
  height: number
}

interface BoxRegion {
  corners: [THREE.Vector3, THREE.Vector3, THREE.Vector3]
  height: number
}

interface FittingState {
  // Cylinder fitting
  cylinderPhase: 'none' | 'center' | 'radius' | 'height' | 'fitting' | 'selecting'
  cylinderCenter: THREE.Vector3 | null
  cylinderAxis: THREE.Vector3 | null
  cylinderRadius: number
  cylinderHeight: number
  fittedCylinders: FittedCylinder[]

  // Box fitting
  boxPhase: 'none' | 'corner1' | 'corner2' | 'corner3' | 'height' | 'fitting' | 'selecting'
  boxCorners: THREE.Vector3[]
  boxHeight: number
  fittedBoxes: FittedBox[]

  // Settings
  tolerance: number
  minInliers: number

  // Actions
  setCylinderPhase: (phase: FittingState['cylinderPhase']) => void
  setCylinderCenter: (center: THREE.Vector3 | null) => void
  setCylinderAxis: (axis: THREE.Vector3 | null) => void
  setCylinderRadius: (radius: number) => void
  setCylinderHeight: (height: number) => void
  setFittedCylinders: (cylinders: FittedCylinder[]) => void
  toggleCylinderAccepted: (id: number) => void

  setBoxPhase: (phase: FittingState['boxPhase']) => void
  addBoxCorner: (corner: THREE.Vector3) => void
  setBoxHeight: (height: number) => void
  setFittedBoxes: (boxes: FittedBox[]) => void
  toggleBoxAccepted: (id: number) => void

  setTolerance: (tolerance: number) => void
  setMinInliers: (minInliers: number) => void

  resetCylinder: () => void
  resetBox: () => void

  getCylinderRegion: () => CylinderRegion | null
  getBoxRegion: () => BoxRegion | null
}

export const useFittingStore = create<FittingState>((set, get) => ({
  // Cylinder state
  cylinderPhase: 'none',
  cylinderCenter: null,
  cylinderAxis: null,
  cylinderRadius: 0,
  cylinderHeight: 0,
  fittedCylinders: [],

  // Box state
  boxPhase: 'none',
  boxCorners: [],
  boxHeight: 0,
  fittedBoxes: [],

  // Settings
  tolerance: 0.02,
  minInliers: 500,

  // Cylinder actions
  setCylinderPhase: (phase) => set({ cylinderPhase: phase }),
  setCylinderCenter: (center) => set({ cylinderCenter: center }),
  setCylinderAxis: (axis) => set({ cylinderAxis: axis }),
  setCylinderRadius: (radius) => set({ cylinderRadius: radius }),
  setCylinderHeight: (height) => set({ cylinderHeight: height }),
  setFittedCylinders: (cylinders) => set({ fittedCylinders: cylinders }),
  toggleCylinderAccepted: (id) => set((state) => ({
    fittedCylinders: state.fittedCylinders.map(c =>
      c.id === id ? { ...c, accepted: !c.accepted } : c
    )
  })),

  // Box actions
  setBoxPhase: (phase) => set({ boxPhase: phase }),
  addBoxCorner: (corner) => set((state) => ({
    boxCorners: [...state.boxCorners, corner]
  })),
  setBoxHeight: (height) => set({ boxHeight: height }),
  setFittedBoxes: (boxes) => set({ fittedBoxes: boxes }),
  toggleBoxAccepted: (id) => set((state) => ({
    fittedBoxes: state.fittedBoxes.map(b =>
      b.id === id ? { ...b, accepted: !b.accepted } : b
    )
  })),

  // Settings
  setTolerance: (tolerance) => set({ tolerance }),
  setMinInliers: (minInliers) => set({ minInliers }),

  // Reset
  resetCylinder: () => set({
    cylinderPhase: 'none',
    cylinderCenter: null,
    cylinderAxis: null,
    cylinderRadius: 0,
    cylinderHeight: 0,
    fittedCylinders: [],
  }),
  resetBox: () => set({
    boxPhase: 'none',
    boxCorners: [],
    boxHeight: 0,
    fittedBoxes: [],
  }),

  // Get region for API
  getCylinderRegion: () => {
    const { cylinderCenter, cylinderAxis, cylinderRadius, cylinderHeight } = get()
    if (!cylinderCenter || !cylinderAxis) return null
    return { center: cylinderCenter, axis: cylinderAxis, radius: cylinderRadius, height: cylinderHeight }
  },
  getBoxRegion: () => {
    const { boxCorners, boxHeight } = get()
    if (boxCorners.length < 3) return null
    return { corners: boxCorners.slice(0, 3) as [THREE.Vector3, THREE.Vector3, THREE.Vector3], height: boxHeight }
  },
}))
