# FFD CFD (CUDA) + GH Visualizer

CUDA-powered voxel CFD for Rhino/Grasshopper.  
Feed a voxel **occupancy** grid + **domain Brep**, pick **inflow**/**wall** faces, and get **pressure** & **velocity** fields. A visualizer draws vectors and colored slice meshes.
Works with https://github.com/KossPa/VoxelizerPlugin_GH
---

## Features
- CUDA 12.x backend (single GPU)
- Inflow, wall, outflow BCs via face indices
- Axis-aligned voxel grid (aligned to domain AABB)
- Slice meshes (|V| & Pressure) + vector preview
- Iteration controls from Grasshopper

---

## Components

### Solver (`FFD_GHComponent`)
**Inputs**

| Name | Type | Notes |
|---|---|---|
| `O` | `byte[]/bool[]` | Occupancy (1=solid), length `Rx*Ry*Rz` |
| `D` | Brep | Domain used by your voxelizer |
| `Rx,Ry,Rz` | int | Grid resolution |
| `In` | int | Inflow face index (direction = inward face normal) |
| `B` | int[] | Wall (no-slip) face indices |
| `Mu`,`Rho` | double | Viscosity (Pa·s), Density (kg/m³) |
| `S` | double | Inflow speed (m/s) |
| `Ns` | int | Steps (default 20) |
| `Nd` | int | Diffusion iters/step (default 20) |
| `Np` | int | Pressure iters/step (default 20) |

**Outputs**

| Name | Type | Notes |
|---|---|---|
| `P` | `float[]` | Pressure, length `N` |
| `V` | `float[]` | Velocity, length `3N` as `[u | v | w]` |
| `O` | Point3d | Grid origin (world) |
| `dx,dy,dz` | double | Voxel sizes (m) |

> BC flags: **Solid**, **Inflow** (Dirichlet velocity), **Wall** (no-slip), **Outflow** (zero-grad vel, `p=0`).  
> Boundary slabs are aligned to the **domain AABB** (by design for now).

---

### Visualizer (`Velocity Visualizer` / `VelViz`)
**Inputs:** `O`, `V`, `P`, `Rx,Ry,Rz`, `dx,dy,dz`, axis `A` (0/1/2/−1), slice `S`, scale `L`, colormap `C` (0=BlueRed, 1=Plasma).  
**Outputs:** vector **Lines**, **Velocity Slice** mesh (|V|), **Pressure Slice** mesh.

---

## Build

1. **CUDA DLL**: build `FFD_CUDA_Solver` in **Release** (CUDA 12.9+). Produces `FFD_CUDA_Solver.dll`.
2. **.NET**: build `FFD_SolverComponent` & `FFD_VisualizerComponent` (e.g., net48).
3. Copy the `.gha` files **and** `FFD_CUDA_Solver.dll` to your Grasshopper **Libraries** folder.

**Native entrypoint**
```c
int RunCFDSimulation(
  const unsigned char* flags, int Rx,int Ry,int Rz,
  float dx, float dy, float dz,
  float mu, float rho,
  float velX, float velY, float velZ,
  int numSteps, int diffIters, int pressureIters,
  float* pressureOut, float* velocityOut);
```

## TODO

- Implement domain aligned axis for the creation of true domain aligned voxels instead of world axis aligned bounding boxes.
- Use a vertical inflow profile (ABL) instead of a constant inflow.
- Smagorinsky LES term (effective viscosity)
