#pragma once
#ifdef _WIN32
#   define DLL_EXPORT __declspec(dllexport)
#else
#   define DLL_EXPORT
#endif

    /*  N = resX × resY × resZ (just the grid size)
     *  velocityOut layout = [u0 … uN-1 | v0 … vN-1 | w0 … wN-1]       */
extern "C" DLL_EXPORT int RunCFDSimulation(
    const unsigned char* flagArray,   // CellFlag per voxel            
    int   resX, int   resY, int   resZ,
    float dx, float dy, float dz,     // voxel size in m               
    float mu, float rho,              //viscosity & density           
    float velX, float velY, float velZ, // inflow velocity (m/s)       
    int   numSteps,                   // simulation steps (>=1)        
    int   diffIters,                  // viscosity Jacobi iters (>=1)  
    int   pressureIters,              // pressure Jacobi iters (>=1)   
    float* pressureOut,               // N floats                      
    float* velocityOut                // 3·N floats                    
);