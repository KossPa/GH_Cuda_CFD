using FFD_VisualizerComponent.Utils;
using Grasshopper;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace FFD_VisualizerComponent
{
    public class FFD_VisualizerComponent : GH_Component
    {
        /// <summary>
        /// Each implementation of GH_Component must provide a public 
        /// constructor without any arguments.
        /// Category represents the Tab in which the component will appear, 
        /// Subcategory the panel. If you use non-existing tab or panel names, 
        /// new tabs/panels will automatically be created.
        /// </summary>
        public FFD_VisualizerComponent()
          : base("Velocity Visualizer",              
                   "VelViz",                           
                   "Draws CFD velocity vectors coloured by wind speed.",
                   "Wind Simulation", "Visualization")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Origin", "O", "Voxel (0,0,0) world-space", GH_ParamAccess.item);
            pManager.AddGenericParameter("Velocity field", "V", "float[]: U slab, V slab, W slab", GH_ParamAccess.item);
            pManager.AddGenericParameter("Pressure field", "P", "float[]: pressure per voxel", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResX", "Rx", "Voxel resolution X", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResY", "Ry", "Voxel resolution Y", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ResZ", "Rz", "Voxel resolution Z", GH_ParamAccess.item);
            pManager.AddNumberParameter("dx", "dx", "Voxel size X  [m]", GH_ParamAccess.item);
            pManager.AddNumberParameter("dy", "dy", "Voxel size Y  [m]", GH_ParamAccess.item);
            pManager.AddNumberParameter("dz", "dz", "Voxel size Z  [m]", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Section axis", "A", "0=X  1=Y  2=Z  –1=all (vectors only)", GH_ParamAccess.item, -1);
            pManager.AddIntegerParameter("Slice index", "S", "Index along axis", GH_ParamAccess.item, 0);
            pManager.AddNumberParameter("Scale", "L", "Vector length scale (m per m/s)", GH_ParamAccess.item, 1.0);
            pManager.AddIntegerParameter("Colormap", "C", "0 = BlueRed (default), 1 = Plasma", GH_ParamAccess.item, 0);
        }
        private readonly List<Line> _lines = new List<Line>();
        private readonly List<Color> _col = new List<Color>();
        private readonly List<int> _width = new List<int>();


        protected override void BeforeSolveInstance()
        {
            _lines.Clear(); _col.Clear(); _width.Clear();
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddLineParameter("Lines", "L", "Preview lines", GH_ParamAccess.list);
            pManager.AddMeshParameter("Velocity Slice", "MV", "Colored mesh slice of |V|", GH_ParamAccess.item);
            pManager.AddMeshParameter("Pressure Slice", "MP", "Colored mesh slice of pressure", GH_ParamAccess.item);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object can be used to retrieve data from input parameters and 
        /// to store data in output parameters.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            /* ── input ──────────────────────────────────── */
            Point3d origin = Point3d.Unset; if (!DA.GetData(0, ref origin)) return;

            float[] V = null; if (!DA.GetData(1, ref V)) return;
            float[] P = null; if (!DA.GetData(2, ref P)) return;

            int Rx = 0, Ry = 0, Rz = 0;
            if (!DA.GetData(3, ref Rx)) return;
            if (!DA.GetData(4, ref Ry)) return;
            if (!DA.GetData(5, ref Rz)) return;

            int N = Rx * Ry * Rz;
            if (V == null || V.Length != 3 * N)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Velocity array length {(V == null ? 0 : V.Length)} ≠ 3·{N}");
                return;
            }
            if (P == null || P.Length != N)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Pressure array length {(P == null ? 0 : P.Length)} ≠ {N}");
                return;
            }

            double dx = 1, dy = 1, dz = 1;
            DA.GetData(6, ref dx); DA.GetData(7, ref dy); DA.GetData(8, ref dz);

            int axis = -1; DA.GetData(9, ref axis);                // –1 → vectors only
            int slice = 0; DA.GetData(10, ref slice);
            double scale = 1.0; DA.GetData(11, ref scale);
            scale = Math.Max(1e-9, scale);

            int cmapIdx = 0; DA.GetData(12, ref cmapIdx);
            if (cmapIdx < 0) cmapIdx = 0; if (cmapIdx > 1) cmapIdx = 1;
            var cmap = (Utils.ColorMap)cmapIdx;

            if (axis >= 0)
            {
                int max = (axis == 0) ? Rx - 1 : (axis == 1) ? Ry - 1 : Rz - 1;
                slice = Math.Max(0, Math.Min(max, slice));
            }

            /* ── helpers ────────────────────────────────── */
            Func<int, Vector3d> Vel = id => new Vector3d(V[id], V[id + N], V[id + 2 * N]);
            Func<int, double> Speed = id => Vel(id).Length;

            /* ── colour range for VECTORS (shown region) ───────────────────── */
            double minS = double.MaxValue, maxS = 0.0;
            for (int k = 0; k < Rz; ++k)
                for (int j = 0; j < Ry; ++j)
                    for (int i = 0; i < Rx; ++i)
                    {
                        if (axis >= 0 &&
                           ((axis == 0 && i != slice) ||
                            (axis == 1 && j != slice) ||
                            (axis == 2 && k != slice))) continue;

                        double s = Speed(i + j * Rx + k * Rx * Ry);
                        if (s < 1e-9) continue;
                        if (s < minS) minS = s;
                        if (s > maxS) maxS = s;
                    }
            if (!(minS < maxS)) { minS = 0; maxS = 1; }

            /* ── build preview lines ───────────────────────────── */
            for (int k = 0; k < Rz; ++k)
                for (int j = 0; j < Ry; ++j)
                    for (int i = 0; i < Rx; ++i)
                    {
                        if (axis >= 0 &&
                           ((axis == 0 && i != slice) ||
                            (axis == 1 && j != slice) ||
                            (axis == 2 && k != slice))) continue;

                        int id = i + j * Rx + k * Rx * Ry;
                        Vector3d v = Vel(id);
                        double s = v.Length;
                        if (s < 1e-9) continue;

                        Point3d p = new Point3d(
                            origin.X + (i + 0.5) * dx,
                            origin.Y + (j + 0.5) * dy,
                            origin.Z + (k + 0.5) * dz);

                        Line ln = new Line(p, v * scale);

                        _lines.Add(ln);
                        _col.Add(VisHelpers.Ramp(s, minS, maxS, cmap));
                        _width.Add(1);
                    }

            DA.SetDataList(0, _lines);

            /* ── slice meshes (only when a single axis slice is selected) ──── */
            Mesh mVel = null, mPrs = null;

            if (axis >= 0)
            {
                // Velocity magnitude slice
                double vMin, vMax;
                VisHelpers.FindSliceMinMax(axis, slice, Rx, Ry, Rz, Speed, out vMin, out vMax);
                if (!(vMin < vMax)) { vMin = 0; vMax = 1; }
                mVel = VisHelpers.BuildSliceMesh(axis, slice, Rx, Ry, Rz, dx, dy, dz, origin,
                                                 Speed, vMin, vMax, cmap);

                // Pressure slice
                Func<int, double> Pval = id => P[id];
                double pMin, pMax;
                VisHelpers.FindSliceMinMax(axis, slice, Rx, Ry, Rz, Pval, out pMin, out pMax);
                if (!(pMin < pMax)) { pMin -= 0.5; pMax += 0.5; }
                mPrs = VisHelpers.BuildSliceMesh(axis, slice, Rx, Ry, Rz, dx, dy, dz, origin,
                                                 Pval, pMin, pMax, cmap);
            }

            DA.SetData(1, mVel);
            DA.SetData(2, mPrs);
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
        public override Guid ComponentGuid => new Guid("643eef8c-7720-48b7-aaaa-5c456c63521d");
    }
}