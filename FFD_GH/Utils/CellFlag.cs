using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FFD_SolverComponent.Utils
{
    /// <summary>
    /// Bit-flags for the GPU , not to be changed under any circumstances.
    /// </summary>
    [System.Flags]
    public enum CellFlag : byte
    {
        Fluid = 0,        // 0000
        Solid = 1 << 0,   // 0001
        Inflow = 1 << 1,   // 0010
        Wall = 1 << 2,   // 0100
        Outflow = 1 << 3    // 1000
    }
}
