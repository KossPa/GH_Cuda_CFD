using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using FFD_SolverComponent.Utils;
using System.Linq;
using System.Drawing;
namespace FFD_SolverComponent
{
    public class FFD_GHComponent : GH_Component
    {
        /// <summary>
        /// Each implementation of GH_Component must provide a public 
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear, 
        /// Subcategory the panel. If you use non-existing tab or panel names, 
        /// new tabs/panels will automatically be created.
        /// </summary>
        public FFD_GHComponent()
          : base("FFD_GHComponent", "Nickname",
            "Description",
            "Category", "Subcategory")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Occupancy", "O",
         "bool[ ] or byte[ ] from the voxeliser (1 = solid)", GH_ParamAccess.item);
            pManager.AddBrepParameter("Domain", "D",
                "Axis-aligned Brep identical to the voxeliser domain", GH_ParamAccess.item);

            pManager.AddIntegerParameter("ResX", "Rx", "Grid resolution X", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResY", "Ry", "Grid resolution Y", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResZ", "Rz", "Grid resolution Z", GH_ParamAccess.item);

            pManager.AddIntegerParameter("Inflow Face", "In", "Brep face index used as inflow", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Wall Faces", "B", "Indexes of wall / no-slip faces", GH_ParamAccess.list);

            pManager.AddNumberParameter("Viscosity μ", "Mu", "Dynamic viscosity [Pa·s]", GH_ParamAccess.item, 1.8e-5);
            pManager.AddNumberParameter("Density ρ", "Rho", "Density [kg/m³]", GH_ParamAccess.item, 1.225);
            pManager.AddNumberParameter("Speed", "S", "Inflow speed [m/s] (direction taken from face normal)",
                                 GH_ParamAccess.item, 2.0);

            //iteration controls (defaults 20)
            pManager.AddIntegerParameter("Steps", "Ns", "Number of simulation steps", GH_ParamAccess.item, 20);
            pManager.AddIntegerParameter("Diffusion Iters", "Nd", "Jacobi iterations for viscosity", GH_ParamAccess.item, 20);
            pManager.AddIntegerParameter("Pressure Iters", "Np", "Jacobi iterations for pressure", GH_ParamAccess.item, 20);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Pressure", "P", "float[ ] pressure per voxel", GH_ParamAccess.item);
            pManager.AddGenericParameter("Velocity", "V", "float[ ] 3·N flattened velocity", GH_ParamAccess.item);
            pManager.AddPointParameter("Origin", "O", "Grid origin (world-space)", GH_ParamAccess.item);
            pManager.AddNumberParameter("dx", "dx", "Voxel size X [m]", GH_ParamAccess.item);
            pManager.AddNumberParameter("dy", "dy", "Voxel size Y [m]", GH_ParamAccess.item);
            pManager.AddNumberParameter("dz", "dz", "Voxel size Z [m]", GH_ParamAccess.item);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            /* ----- raw inputs ----- */
            byte[] occ = null; if (!DA.GetData(0, ref occ)) return;
            Brep dom = null; if (!DA.GetData(1, ref dom)) return;

            int Rx = 0, Ry = 0, Rz = 0;
            if (!DA.GetData(2, ref Rx)) return;
            if (!DA.GetData(3, ref Ry)) return;
            if (!DA.GetData(4, ref Rz)) return;

            int inflowFace = -1; DA.GetData(5, ref inflowFace);
            var wallFaces = new List<int>(); DA.GetDataList(6, wallFaces);

            double mu = 0, rho = 0; DA.GetData(7, ref mu);
            DA.GetData(8, ref rho);

            double speed = 0.0; DA.GetData(9, ref speed);
            speed = Math.Max(0.0, speed);

            // NEW: iteration controls
            int numSteps = 20, diffIters = 20, pressureIters = 20;
            DA.GetData(10, ref numSteps);
            DA.GetData(11, ref diffIters);
            DA.GetData(12, ref pressureIters);
            numSteps = Math.Max(1, numSteps);
            diffIters = Math.Max(1, diffIters);
            pressureIters = Math.Max(1, pressureIters);

            int N = Rx * Ry * Rz;
            if (occ.Length != N)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                    $"Occupancy length {occ.Length} ≠ {N} (grid size)");
                return;
            }

            /* ----- CellFlag map ----- */
            var flags = new byte[N];

            for (int k = 0; k < Rz; k++)
                for (int j = 0; j < Ry; j++)
                    for (int i = 0; i < Rx; i++)
                    {
                        int id = i + j * Rx + k * Rx * Ry;
                        flags[id] = occ[id] != 0
                                  ? (byte)CellFlag.Solid
                                  : (byte)CellFlag.Fluid;
                    }

            var faceVox = FaceVoxelizer.GetFaceVoxelSets(dom, Rx, Ry, Rz);
            var sb = new System.Text.StringBuilder("Face voxel counts: ");
            for (int f = 0; f < faceVox.Count; ++f)
                sb.Append($"[{f}:{faceVox[f].Length}] ");
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, sb.ToString());

            // A) Mark WALLS first (fluid only)
            for (int idx = 0; idx < wallFaces.Count; ++idx)
            {
                int fi = wallFaces[idx];
                if (fi >= 0 && fi < faceVox.Count)
                {
                    var arr = faceVox[fi];
                    for (int t = 0; t < arr.Length; ++t)
                    {
                        int v = arr[t];
                        if ((flags[v] & (byte)CellFlag.Solid) != 0) continue;
                        flags[v] |= (byte)CellFlag.Wall;
                        flags[v] = (byte)(flags[v] & ~(byte)CellFlag.Inflow);
                        flags[v] = (byte)(flags[v] & ~(byte)CellFlag.Outflow);
                    }
                }
            }

            // B) Mark INFLOW (fluid only)
            if (inflowFace >= 0 && inflowFace < faceVox.Count)
            {
                var arr = faceVox[inflowFace];
                for (int t = 0; t < arr.Length; ++t)
                {
                    int v = arr[t];
                    if ((flags[v] & (byte)CellFlag.Solid) != 0) continue;
                    flags[v] = (byte)(flags[v] & ~(byte)CellFlag.Wall);
                    flags[v] = (byte)(flags[v] & ~(byte)CellFlag.Outflow);
                    flags[v] |= (byte)CellFlag.Inflow;
                }
            }
            else
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Inflow face index invalid.");
                return;
            }

            // C) Outflow = other boundary faces that are not inflow and not walls (fluid only)
            for (int f = 0; f < faceVox.Count; ++f)
            {
                if (f == inflowFace) continue;
                if (wallFaces != null && wallFaces.Contains(f)) continue;
                var arr = faceVox[f];
                if (arr == null || arr.Length == 0) continue;

                for (int t = 0; t < arr.Length; ++t)
                {
                    int v = arr[t];
                    if ((flags[v] & (byte)CellFlag.Solid) != 0) continue;
                    if ((flags[v] & (byte)CellFlag.Inflow) != 0) continue;
                    if ((flags[v] & (byte)CellFlag.Wall) != 0) continue;
                    flags[v] |= (byte)CellFlag.Outflow;
                }
            }

            /* ----- grid metrics ----- */
            BoundingBox bb = dom.GetBoundingBox(true);
            double dx = bb.Diagonal.X / Rx;
            double dy = bb.Diagonal.Y / Ry;
            double dz = bb.Diagonal.Z / Rz;

            /* ----- inflow vector from face inward normal ----- */
            Vector3d V;
            {
                var face = dom.Faces[inflowFace];
                Plane fPln;
                if (!face.TryGetPlane(out fPln, 1e-6))
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Inflow face not planar.");
                    return;
                }
                var du = face.Domain(0);
                var dv = face.Domain(1);
                Point3d cFace = face.PointAt(0.5 * (du.T0 + du.T1), 0.5 * (dv.T0 + dv.T1));
                Vector3d n = fPln.Normal; n.Unitize();
                Vector3d toCenter = (bb.Center - cFace);
                if (toCenter * n < 0.0) n = -n; // inward
                V = n * speed;
            }

            /* ----- DEBUG ----- */
            int solidCount = 0, wallCount = 0, inflowCount = 0, outflowCount = 0;
            for (int id = 0; id < N; ++id)
            {
                byte f = flags[id];
                if ((f & (byte)CellFlag.Solid) != 0) solidCount++;
                if (((f & (byte)CellFlag.Wall) != 0) && (f & (byte)CellFlag.Solid) == 0) wallCount++;
                if (((f & (byte)CellFlag.Inflow) != 0) && (f & (byte)CellFlag.Solid) == 0) inflowCount++;
                if (((f & (byte)CellFlag.Outflow) != 0) && (f & (byte)CellFlag.Solid) == 0) outflowCount++;
            }
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                $"Flags: Wall={wallCount}, Inflow={inflowCount}, Outflow={outflowCount}, Solids={solidCount}");
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                $"Inflow vector sent to CUDA: ({V.X:F4}, {V.Y:F4}, {V.Z:F4})  speed={speed:F4}");
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                $"Iters: steps={numSteps}, diff={diffIters}, pressure={pressureIters}");

            /* ----- allocate & call CUDA ----- */
            var P = new float[N];
            var Vel = new float[3 * N];

            int err = CudaCFDInterop.RunCFDSimulation(
                flags,
                Rx, Ry, Rz,
                (float)dx, (float)dy, (float)dz,
                (float)mu, (float)rho,
                (float)V.X, (float)V.Y, (float)V.Z,
                numSteps, diffIters, pressureIters,  // NEW
                P, Vel);

            if (err != 0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"CUDA solver error code {err}");
                return;
            }

            /* ----- outputs ----- */
            DA.SetData(0, P);
            DA.SetData(1, Vel);
            DA.SetData(2, new Point3d(bb.Min));
            DA.SetData(3, dx);
            DA.SetData(4, dy);
            DA.SetData(5, dz);
        }

        /// <summary>
        /// Provides an Icon for every component that will be visible in the User Interface.
        /// Icons need to be 24x24 pixels.
        /// You can add image files to your project resources and access them like this:
        /// return Resources.IconForThisComponent;
        /// </summary>
        protected override System.Drawing.Bitmap Icon => null;

        /// <summary>
        /// Each component must have a unique Guid to identify it. 
        /// It is vital this Guid doesn't change otherwise old ghx files 
        /// that use the old ID will partially fail during loading.
        /// </summary>
        public override Guid ComponentGuid => new Guid("2151c7b1-ec38-4712-ab57-58528f910946");
    }
}