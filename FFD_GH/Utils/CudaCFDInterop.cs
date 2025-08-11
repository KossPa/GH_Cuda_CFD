using System.Runtime.InteropServices;

namespace FFD_SolverComponent.Utils
{
    /// <summary>
    /// Thin P/Invoke layer – _signature MUST stay identical_ to <c>SolverAPI.h</c>  <c>Solver.cu</c>.
    /// </summary>
    internal static class CudaCFDInterop
    {
        [DllImport("FFD_CUDA_Solver.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int RunCFDSimulation(
    /* ---- flags & grid ---- */
    [In] byte[] flagArray,   /* length = N                      */
    int resX,
    int resY,
    int resZ,
    float dx,           /* voxel size (m) – X              */
    float dy,           /* voxel size (m) – Y              */
    float dz,           /* voxel size (m) – Z              */

    /* ---- fluid properties & inflow ---- */
    float mu,           /* dynamic viscosity  (Pa·s)       */
    float rho,          /* density (kg/m³)                 */
    float velX,         /* inflow velocity components      */
    float velY,
    float velZ,

    /* ---- iteration controls (NEW) ---- */
    int numSteps,
    int diffIters,
    int pressureIters,

    /* ---- outputs ---- */
    [Out] float[] pressureOut,/* N                               */
    [Out] float[] velocityOut /* 3·N (u-slab | v-slab | w-slab)  */
    );
    }
}
