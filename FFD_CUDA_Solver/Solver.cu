/*  ----- Cuda 12.9 , check also header SolverAPI as its needed ------*/

#include "SolverAPI.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>      // fminf / fmaxf
#include <algorithm>  // std::swap  (host)
#define CUDA_OK(x) { gpuAssert((x),__FILE__,__LINE__); }

 /* ------------------------------------ Set to 1 to synchronize after each kernel in Debug ------------------------------------ */
#ifndef DEBUG_SYNC
#define DEBUG_SYNC 0
#endif

#ifdef _DEBUG
#if DEBUG_SYNC
#define CUDA_CHECK_KERNEL() do {                                          \
        cudaError_t e__ = cudaGetLastError();                                 \
        if (e__ != cudaSuccess)                                               \
            fprintf(stderr, "CUDA kernel error: %s  at  %s:%d\n",             \
                    cudaGetErrorString(e__), __FILE__, __LINE__);             \
        CUDA_OK(cudaDeviceSynchronize());                                     \
    } while(0)
#else
  /*------------------------------------  Non-blocking check: catch errors without stalling the GPU. ------------------------------------ */
#define CUDA_CHECK_KERNEL() do {                                          \
        cudaError_t e__ = cudaGetLastError();                                 \
        if (e__ != cudaSuccess)                                               \
            fprintf(stderr, "CUDA kernel error: %s  at  %s:%d\n",             \
                    cudaGetErrorString(e__), __FILE__, __LINE__);             \
    } while(0)
#endif
#else
#define CUDA_CHECK_KERNEL() do {} while(0)
#endif

/* ------------------------------------  Set >0 to check residual every N Jacobi iterations, 0 to disable (fastest)------------------------------------ . */
#ifndef PRESSURE_RESIDUAL_CHECK_EVERY
#define PRESSURE_RESIDUAL_CHECK_EVERY 0
#endif

/*------------------------------------  utilities ------------------------------------  */

inline void gpuAssert(cudaError_t c, const char* f, int l)
{
    if (c != cudaSuccess)
        fprintf(stderr, "CUDA %s  at  %s:%d\n", cudaGetErrorString(c), f, l);
}
//for this check the CellFlag
enum : unsigned char {
    CF_FLUID = 0,
    CF_SOLID = 1,   // 0001
    CF_INFLOW = 2,   // 0010
    CF_WALL = 4,   // 0100
    CF_OUTFLOW = 8    // 1000
};
__device__ __host__ inline bool isSolid(unsigned char f) { return f & CF_SOLID; }
__device__ __host__ inline bool isInflow(unsigned char f) { return f & CF_INFLOW; }
__device__ __host__ inline bool isWall(unsigned char f) { return f & CF_WALL; }
__device__ __host__ inline bool isOutflow(unsigned char f) { return f & CF_OUTFLOW; }

__device__ __host__ inline int  flatten(int i, int j, int k, int nx, int ny, int)
{
    return i + j * nx + k * nx * ny;
}

template<typename T>
__global__ void setValueKernel(T* p, T v, int N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N) p[id] = v;
}

/*------------------------------------  clamp helpers usable on host + device ------------------------------------ */
__device__ __host__ inline int   clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__          inline float clampF(float v, float lo, float hi) { return fminf(hi, fmaxf(lo, v)); }

/* ------------------------------------ boundary kernel to flag the cells on domain faces------------------------------------*/
__global__ void BoundaryVelocityKernel(float* u, float* v, float* w,
    const unsigned char* flag,
    float inX, float inY, float inZ,
    int N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N) return;
    unsigned char f = flag[id];
    if (isInflow(f)) { u[id] = inX; v[id] = inY; w[id] = inZ; }
    else if (isWall(f) || isSolid(f)) { u[id] = v[id] = w[id] = 0.f; }
}

/* ------------------------------------ ═ sampler helpers ------------------------------------  */
__device__ inline float at(const float* s, int i, int j, int k,
    int nx, int ny, int nz)
{
    i = clampInt(i, 0, nx - 1); j = clampInt(j, 0, ny - 1); k = clampInt(k, 0, nz - 1);
    return s[flatten(i, j, k, nx, ny, nz)];
}
__device__ inline unsigned char atFlag(const unsigned char* s, int i, int j, int k,
    int nx, int ny, int nz)
{
    i = clampInt(i, 0, nx - 1); j = clampInt(j, 0, ny - 1); k = clampInt(k, 0, nz - 1);
    return s[flatten(i, j, k, nx, ny, nz)];
}
__device__ float sampleLinear(const float* s, float gx, float gy, float gz,
    int nx, int ny, int nz)
{
    gx = clampF(gx, 0.f, nx - 1.001f);
    gy = clampF(gy, 0.f, ny - 1.001f);
    gz = clampF(gz, 0.f, nz - 1.001f);

    int i0 = (int)gx, j0 = (int)gy, k0 = (int)gz;
    int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
    float tx = gx - i0, ty = gy - j0, tz = gz - k0;

    float c000 = at(s, i0, j0, k0, nx, ny, nz), c100 = at(s, i1, j0, k0, nx, ny, nz);
    float c010 = at(s, i0, j1, k0, nx, ny, nz), c110 = at(s, i1, j1, k0, nx, ny, nz);
    float c001 = at(s, i0, j0, k1, nx, ny, nz), c101 = at(s, i1, j0, k1, nx, ny, nz);
    float c011 = at(s, i0, j1, k1, nx, ny, nz), c111 = at(s, i1, j1, k1, nx, ny, nz);

    float c00 = c000 * (1 - tx) + c100 * tx, c10 = c010 * (1 - tx) + c110 * tx;
    float c01 = c001 * (1 - tx) + c101 * tx, c11 = c011 * (1 - tx) + c111 * tx;

    float c0 = c00 * (1 - ty) + c10 * ty, c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
}


/* ------------------------------------  advection------------------------------------  */
__global__ void AdvectKernel(float* dst, const float* src,
    const float* u, const float* v, const float* w,
    const unsigned char* flag,
    int nx, int ny, int nz,
    float dt, float hx, float hy, float hz)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nx * ny * nz) return;

    if (isSolid(flag[id]) || isWall(flag[id])) { dst[id] = 0.f; return; }

    int k = id / (nx * ny); int j = (id - k * nx * ny) / nx; int i = id - k * nx * ny - j * nx;

    float px = (i + 0.5f) * hx, py = (j + 0.5f) * hy, pz = (k + 0.5f) * hz;
    float velX = u[id], velY = v[id], velZ = w[id];

    float bx = px - velX * dt, by = py - velY * dt, bz = pz - velZ * dt;
    float gx = bx / hx - 0.5f, gy = by / hy - 0.5f, gz = bz / hz - 0.5f;

    dst[id] = sampleLinear(src, gx, gy, gz, nx, ny, nz);
}


/* ------------------------------------  diffusion ------------------------------------  */
__global__ void DiffuseJacobiKernel(float* dst, const float* src,
    const unsigned char* flag,
    int nx, int ny, int nz,
    float ax, float ay, float az, float rBeta)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nx * ny * nz) return;
    if (isSolid(flag[id]) || isWall(flag[id])) { dst[id] = 0.f; return; }

    int k = id / (nx * ny);
    int j = (id - k * nx * ny) / nx;
    int i = id - k * nx * ny - j * nx;

    // neighbor sums per axis
    float sx = at(src, i + 1, j, k, nx, ny, nz) + at(src, i - 1, j, k, nx, ny, nz);
    float sy = at(src, i, j + 1, k, nx, ny, nz) + at(src, i, j - 1, k, nx, ny, nz);
    float sz = at(src, i, j, k + 1, nx, ny, nz) + at(src, i, j, k - 1, nx, ny, nz);

    // Jacobi update with anisotropic weights
    dst[id] = (src[id] + ax * sx + ay * sy + az * sz) * rBeta;
}


/* ------------------------------------  divergence ------------------------------------  */
__global__ void DivergenceKernel(const float* u, const float* v, const float* w,
    const unsigned char* flag, float* div,
    int nx, int ny, int nz,
    float hx, float hy, float hz)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nx * ny * nz) return;

    int k = id / (nx * ny);
    int j = (id - k * nx * ny) / nx;
    int i = id - k * nx * ny - j * nx;

    // No divergence in solid/wall cells
    unsigned char f0 = flag[id];
    if (isSolid(f0) || isWall(f0)) { div[id] = 0.f; return; }

    // Neighbor flags (inside-domain) used for one-sided BC at walls/solids
    unsigned char fl = (i > 0) ? flag[flatten(i - 1, j, k, nx, ny, nz)] : f0;
    unsigned char fr = (i < nx - 1) ? flag[flatten(i + 1, j, k, nx, ny, nz)] : f0;
    unsigned char fb = (j > 0) ? flag[flatten(i, j - 1, k, nx, ny, nz)] : f0;
    unsigned char ft = (j < ny - 1) ? flag[flatten(i, j + 1, k, nx, ny, nz)] : f0;
    unsigned char ff = (k > 0) ? flag[flatten(i, j, k - 1, nx, ny, nz)] : f0;
    unsigned char fn = (k < nz - 1) ? flag[flatten(i, j, k + 1, nx, ny, nz)] : f0;

    bool leftSolid = isSolid(fl) || isWall(fl);
    bool rightSolid = isSolid(fr) || isWall(fr);
    bool backSolid = isSolid(fb) || isWall(fb);
    bool topSolid = isSolid(ft) || isWall(ft);
    bool frontSolid = isSolid(ff) || isWall(ff);
    bool nearSolid = isSolid(fn) || isWall(fn);

    // Center values
    float ui = u[id], vi = v[id], wi = w[id];

    // Helpers to read inbounds neighbor (no clamping)
    auto at_nb = [&](const float* s, int ii, int jj, int kk) -> float
        {
            if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) return 0.f; // this remeains unused mostly unless guard fails
            return s[flatten(ii, jj, kk, nx, ny, nz)];
        };

    // ∂u/∂x
    float dudx;
    if (i == 0)                            dudx = (at_nb(u, i + 1, j, k) - ui) / hx;           // forward diff at -X boundary
    else if (i == nx - 1)                  dudx = (ui - at_nb(u, i - 1, j, k)) / hx;           // backward diff at +X boundary
    else if (rightSolid && !leftSolid)   dudx = (0.f - ui) / hx;                              // wall on + side
    else if (leftSolid && !rightSolid)  dudx = (ui - 0.f) / hx;                              // wall on - side
    else                                   dudx = (at_nb(u, i + 1, j, k) - at_nb(u, i - 1, j, k)) / (2.f * hx);

    // ∂v/∂y
    float dvdy;
    if (j == 0)                            dvdy = (at_nb(v, i, j + 1, k) - vi) / hy;           // forward at -Y
    else if (j == ny - 1)                  dvdy = (vi - at_nb(v, i, j - 1, k)) / hy;           // backward at +Y
    else if (topSolid && !backSolid)    dvdy = (0.f - vi) / hy;                              // wall on +Y
    else if (backSolid && !topSolid)     dvdy = (vi - 0.f) / hy;                              // wall on -Y
    else                                   dvdy = (at_nb(v, i, j + 1, k) - at_nb(v, i, j - 1, k)) / (2.f * hy);

    // ∂w/∂z
    float dwdz;
    if (k == 0)                            dwdz = (at_nb(w, i, j, k + 1) - wi) / hz;           // forward at -Z
    else if (k == nz - 1)                  dwdz = (wi - at_nb(w, i, j, k - 1)) / hz;           // backward at +Z
    else if (nearSolid && !frontSolid)   dwdz = (0.f - wi) / hz;                              // wall on +Z
    else if (frontSolid && !nearSolid)   dwdz = (wi - 0.f) / hz;                              // wall on -Z
    else                                   dwdz = (at_nb(w, i, j, k + 1) - at_nb(w, i, j, k - 1)) / (2.f * hz);

    div[id] = dudx + dvdy + dwdz;
}

/* ------------------------------------ pressure Poisson (Jacobi)   ------------------------------------ 
here each thread finds its cell checks if its boundary or not if so apply boun. condition if not read neighbour pressures combine with divergence then compute new p and write it in a new array.*/
__global__ void JacobiPressureKernel(
    float* __restrict__ pN,          // new pressure
    const float* __restrict__ pO,    // old pressure
    const float* __restrict__ div,   // divergence
    const unsigned char* __restrict__ flag,
    int nx, int ny, int nz,
    float ax, float ay, float az,    // 1/hx^2, 1/hy^2, 1/hz^2
    float rBeta,                     // 1 / (2*(ax+ay+az))
    float bScale)                    // rho / dt
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nx * ny * nz) return;

    int k = id / (nx * ny);
    int j = (id - k * nx * ny) / nx;
    int i = id - k * nx * ny - j * nx;

    unsigned char f = flag[id];

    // Keep p in solids/walls (Neumann); set p=0 at outflow (Dirichlet)
    if (isSolid(f) || isWall(f)) { pN[id] = pO[id]; return; }
    if (isOutflow(f)) { pN[id] = 0.f;    return; }

    float sx = at(pO, i + 1, j, k, nx, ny, nz) + at(pO, i - 1, j, k, nx, ny, nz);
    float sy = at(pO, i, j + 1, k, nx, ny, nz) + at(pO, i, j - 1, k, nx, ny, nz);
    float sz = at(pO, i, j, k + 1, nx, ny, nz) + at(pO, i, j, k - 1, nx, ny, nz);

    float rhs = bScale * div[id];  // (rho / dt) * divergence
    pN[id] = (ax * sx + ay * sy + az * sz - rhs) * rBeta;
}
/* ═══ pressure residual – single-thread compute (no atomics/syncthreads) ═══ */
__global__ void ResidualComputeKernel(
    const float* __restrict__ p,       // current pressure
    const float* __restrict__ pPrev,   // previous pressure
    const unsigned char* __restrict__ flag,
    int N,
    float* __restrict__ outRes,        // sum of squared updates
    int* __restrict__ outCnt)        // number of interior cells
{
    // Launch with <<<1,1>>>. Do the reduction serially on device (simple & robust).
    float sum = 0.f;
    int   cnt = 0;
    for (int id = 0; id < N; ++id)
    {
        unsigned char f = flag[id];
        // interior = not solid, not wall, not outflow (Dirichlet)
        if (!(isSolid(f) || isWall(f) || isOutflow(f)))
        {
            float du = p[id] - pPrev[id];
            sum += du * du;
            cnt += 1;
        }
    }
    *outRes = sum;
    *outCnt = cnt;
}

/* ------------------------------------ projection (subtract grad p) ------------------------------------ */
__global__ void PressureGradKernel(const float* p, const unsigned char* flag,
    float* u, float* v, float* w,
    int nx, int ny, int nz,
    float hx, float hy, float hz, float scale)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nx * ny * nz) return;
    if (isSolid(flag[id]) || isWall(flag[id])) return;

    int k = id / (nx * ny); int j = (id - k * nx * ny) / nx; int i = id - k * nx * ny - j * nx;

    float dpdx = (at(p, i + 1, j, k, nx, ny, nz) - at(p, i - 1, j, k, nx, ny, nz)) / (2 * hx);
    float dpdy = (at(p, i, j + 1, k, nx, ny, nz) - at(p, i, j - 1, k, nx, ny, nz)) / (2 * hy);
    float dpdz = (at(p, i, j, k + 1, nx, ny, nz) - at(p, i, j, k - 1, nx, ny, nz)) / (2 * hz);

    u[id] -= scale * dpdx; v[id] -= scale * dpdy; w[id] -= scale * dpdz;
}
/* ------------------------------------ outflow BC: zero normal gradient on velocity (copy interior) ------------------------------------ */
__global__ void OutflowZeroGradKernel(
    float* u, float* v, float* w,
    const unsigned char* flag,
    int nx, int ny, int nz)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nx * ny * nz) return;

    if (!isOutflow(flag[id])) return;

    int k = id / (nx * ny);
    int j = (id - k * nx * ny) / nx;
    int i = id - k * nx * ny - j * nx;

    // Choose the interior neighbor along the outward normal.
    // Priority order handles edges/corners deterministically.
    int ii = i, jj = j, kk = k;
    if (i == 0)           ii = 1;         // -X slab → copy from +X
    else if (i == nx - 1) ii = nx - 2;    // +X slab → copy from -X
    else if (j == 0)      jj = 1;         // -Y slab → copy from +Y
    else if (j == ny - 1) jj = ny - 2;    // +Y slab → copy from -Y
    else if (k == 0)      kk = 1;         // -Z slab → copy from +Z
    else if (k == nz - 1) kk = nz - 2;    // +Z slab → copy from -Z
    else return; // Not a domain-exterior slab (shouldn't happen if flagged as Outflow)

    int nid = flatten(ii, jj, kk, nx, ny, nz);
    u[id] = u[nid];
    v[id] = v[nid];
    w[id] = w[nid];
}
/* ------------------------------------ host entry ------------------------------------ */
extern "C" __declspec(dllexport)
int RunCFDSimulation(const unsigned char* hFlag,
    int   nx, int  ny, int  nz,
    float dx, float dy, float dz,
    float mu, float rho,
    float inX, float inY, float inZ,
    int   numSteps,        // NEW
    int   diffIters,       // NEW
    int   pressureIters,   // NEW
    float* hP,
    float* hUVW)
{
    /* ------------------------------------ constants --------------------------------------------------- */
    const int   N = nx * ny * nz;
    const float hx = dx, hy = dy, hz = dz;

    // --- Physically meaningful time step ---
    const float hmin = fminf(hx, fminf(hy, hz));
    const float vin = sqrtf(inX * inX + inY * inY + inZ * inZ);
    const float CFL = 0.5f;                 // semi-Lagrangian: CFL for accuracy
    const float epsV = 1e-6f;

    float dt_adv = (vin > epsV) ? (CFL * hmin / vin) : 0.5f;
    const float nu = mu / fmaxf(rho, 1e-12f);
    float dt_diff = (nu > 1e-12f) ? (0.25f * hmin * hmin / nu) : 0.5f;

    float dt = fminf(dt_adv, dt_diff);
    dt = fmaxf(dt, 1e-4f);
    dt = fminf(dt, 0.5f);

    // NEW: iteration counts from caller (clamped to >=1)
    const int   NUM_STEPS = (numSteps > 0) ? numSteps : 20;
    const int   DIFF_ITERS = (diffIters > 0) ? diffIters : 20;                    ////////Pressure and Diffussion are called in every step ////// 20 is enough to get a result especially for small grids //////
    const int   PRESSURE_ITERS = (pressureIters > 0) ? pressureIters : 20;                

    /* ------------------------------------ diffusion coefficients (implicit Jacobi) ------------------- */
    const float aDx = dt * nu / (hx * hx);
    const float aDy = dt * nu / (hy * hy);
    const float aDz = dt * nu / (hz * hz);
    const float rBD = 1.f / (1.f + 2.f * (aDx + aDy + aDz));

    /* ------------------------------------ pressure Poisson coefficients: ∇²p = (ρ/Δt) ∇·u ------------ */
    const float aPx = 1.f / (hx * hx);
    const float aPy = 1.f / (hy * hy);
    const float aPz = 1.f / (hz * hz);
    const float rBP = 1.f / (2.f * (aPx + aPy + aPz));
    const float bScale = rho / dt;

    /* ------------------------------------ device buffers --------------------------------------------- */
    float* u;   CUDA_OK(cudaMalloc(&u, N * sizeof(float)));
    float* v;   CUDA_OK(cudaMalloc(&v, N * sizeof(float)));
    float* w;   CUDA_OK(cudaMalloc(&w, N * sizeof(float)));
    float* u1;  CUDA_OK(cudaMalloc(&u1, N * sizeof(float)));
    float* v1;  CUDA_OK(cudaMalloc(&v1, N * sizeof(float)));
    float* w1;  CUDA_OK(cudaMalloc(&w1, N * sizeof(float)));
    float* p;   CUDA_OK(cudaMalloc(&p, N * sizeof(float)));
    float* p1;  CUDA_OK(cudaMalloc(&p1, N * sizeof(float)));
    float* div; CUDA_OK(cudaMalloc(&div, N * sizeof(float)));

    unsigned char* dFlag; CUDA_OK(cudaMalloc(&dFlag, N * sizeof(unsigned char)));
    CUDA_OK(cudaMemcpy(dFlag, hFlag, N * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Scalars for residual early-stop
    float* dRes; CUDA_OK(cudaMalloc(&dRes, sizeof(float)));
    int* dCnt; CUDA_OK(cudaMalloc(&dCnt, sizeof(int)));

    dim3 blk(256), grd((N + blk.x - 1) / blk.x);

    /* ------------------------------------ zero initialise everything + set inflow/wall conditions ------------------------------------ */
    setValueKernel << <grd, blk >> > (u, 0.f, N);
    setValueKernel << <grd, blk >> > (v, 0.f, N);
    setValueKernel << <grd, blk >> > (w, 0.f, N);
    setValueKernel << <grd, blk >> > (u1, 0.f, N);
    setValueKernel << <grd, blk >> > (v1, 0.f, N);
    setValueKernel << <grd, blk >> > (w1, 0.f, N);
    setValueKernel << <grd, blk >> > (p, 0.f, N);
    setValueKernel << <grd, blk >> > (p1, 0.f, N);
    setValueKernel << <grd, blk >> > (div, 0.f, N);
    CUDA_CHECK_KERNEL();

    BoundaryVelocityKernel << <grd, blk >> > (u, v, w, dFlag, inX, inY, inZ, N);
    CUDA_CHECK_KERNEL();

    // Reference pressure scale for tolerance (dynamic pressure, floored)
    const float pRef = fmaxf(0.5f * rho * vin * vin, 1e-2f); // Pa
    const float tolP = 1e-4f * pRef;                         // RMS update target

    /* ------------------------------------ simulation loop follows Jos stam stable fluids approach with a few kinks here and there ------------------------------------ */
    for (int step = 0; step < NUM_STEPS; ++step)
    {
        /* 1 ─ advection ---------------------------------------------- */
        AdvectKernel << <grd, blk >> > (u1, u, u, v, w, dFlag, nx, ny, nz, dt, hx, hy, hz);
        AdvectKernel << <grd, blk >> > (v1, v, u, v, w, dFlag, nx, ny, nz, dt, hx, hy, hz);
        AdvectKernel << <grd, blk >> > (w1, w, u, v, w, dFlag, nx, ny, nz, dt, hx, hy, hz);
        float* tmp;
        tmp = u; u = u1; u1 = tmp;
        tmp = v; v = v1; v1 = tmp;
        tmp = w; w = w1; w1 = tmp;
        CUDA_CHECK_KERNEL();

        /* 2 ─ viscosity (implicit Jacobi) ---------------------------- */
        for (int it = 0; it < DIFF_ITERS; ++it)
        {
            DiffuseJacobiKernel << <grd, blk >> > (u1, u, dFlag, nx, ny, nz, aDx, aDy, aDz, rBD);
            DiffuseJacobiKernel << <grd, blk >> > (v1, v, dFlag, nx, ny, nz, aDx, aDy, aDz, rBD);
            DiffuseJacobiKernel << <grd, blk >> > (w1, w, dFlag, nx, ny, nz, aDx, aDy, aDz, rBD);
            tmp = u; u = u1; u1 = tmp;
            tmp = v; v = v1; v1 = tmp;
            tmp = w; w = w1; w1 = tmp;
        }
        BoundaryVelocityKernel << <grd, blk >> > (u, v, w, dFlag, inX, inY, inZ, N);
        CUDA_CHECK_KERNEL();

        /* 3 ─ divergence --------------------------------------------- */
        DivergenceKernel << <grd, blk >> > (u, v, w, dFlag, div, nx, ny, nz, hx, hy, hz);
        CUDA_CHECK_KERNEL();

        /* 4 ─ pressure Poisson (Jacobi) with early-stop -------------- */
        for (int it = 0; it < PRESSURE_ITERS; ++it)
        {
            JacobiPressureKernel << <grd, blk >> > (p1, p, div, dFlag,
                nx, ny, nz, aPx, aPy, aPz, rBP, bScale);
            tmp = p; p = p1; p1 = tmp;
            CUDA_CHECK_KERNEL();

            // Check every 4 iters to limit sync overhead
            #if PRESSURE_RESIDUAL_CHECK_EVERY > 0
            if (((it + 1) % PRESSURE_RESIDUAL_CHECK_EVERY) == 0 || it == PRESSURE_ITERS - 1)
            {
                ResidualComputeKernel << <1, 1 >> > (p, p1, dFlag, N, dRes, dCnt);
                CUDA_CHECK_KERNEL();

                float hRes = 0.f; int hCnt = 0;
                CUDA_OK(cudaMemcpy(&hRes, dRes, sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemcpy(&hCnt, dCnt, sizeof(int), cudaMemcpyDeviceToHost));
                float rms = (hCnt > 0) ? sqrtf(hRes / (float)hCnt) : 0.f;
                if (rms < tolP) break;
            }
            #endif
        }

        /* 5 ─ projection --------------------------------------------- */
        PressureGradKernel << <grd, blk >> > (p, dFlag, u, v, w,
            nx, ny, nz, hx, hy, hz,
            dt / rho);
        CUDA_CHECK_KERNEL();

        /* 6 ─ enforce open boundary on Outflow ----------------------- */
        OutflowZeroGradKernel << <grd, blk >> > (u, v, w, dFlag, nx, ny, nz);
        CUDA_CHECK_KERNEL();

        /* 7 ─ re-enforce inflow/walls -------------------------------- */
        BoundaryVelocityKernel << <grd, blk >> > (u, v, w, dFlag, inX, inY, inZ, N);
        CUDA_CHECK_KERNEL();
    }
    

    /* --- copy back --------------------------------------------------- */
    CUDA_OK(cudaMemcpy(hP, p, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hUVW, u, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hUVW + N, v, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hUVW + 2 * N, w, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* --- cleanup ----------------------------------------------------- */
    cudaFree(u);  cudaFree(v);  cudaFree(w);
    cudaFree(u1); cudaFree(v1); cudaFree(w1);
    cudaFree(p);  cudaFree(p1); cudaFree(div);
    cudaFree(dRes); cudaFree(dCnt);
    cudaFree(dFlag);

    return 0;
}