using Grasshopper.Kernel;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace FFD_VisualizerComponent
{
    public class PressureSurfaceVisualizer : GH_Component
    {
        public PressureSurfaceVisualizer()
          : base("Pressure Visualiser", "PrsViz",
                 "Colours a Brep’s faces by voxel pressure field.",
                 "Wind Simulation", "Visualization")
        { }

        protected override void RegisterInputParams(GH_InputParamManager p)
        {
            p.AddBrepParameter("Geometry", "G", "Obstacle / façade Brep", GH_ParamAccess.item);
            p.AddGenericParameter("Pressure", "P", "float[] from solver", GH_ParamAccess.item);
            p.AddIntegerParameter("ResX", "Rx", "Voxel X", GH_ParamAccess.item);
            p.AddIntegerParameter("ResY", "Ry", "Voxel Y", GH_ParamAccess.item);
            p.AddIntegerParameter("ResZ", "Rz", "Voxel Z", GH_ParamAccess.item);
            p.AddPointParameter("Origin", "O", "Domain min corner", GH_ParamAccess.item);
            p.AddNumberParameter("dx", "dx", "Voxel size X", GH_ParamAccess.item);
            p.AddNumberParameter("dy", "dy", "Voxel size Y", GH_ParamAccess.item);
            p.AddNumberParameter("dz", "dz", "Voxel size Z", GH_ParamAccess.item);
        }
        protected override void RegisterOutputParams(GH_OutputParamManager p)
        {
            p.AddMeshParameter("Mesh", "M", "Coloured mesh", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            Brep geo = null; float[] P = null;
            int Rx = 0, Ry = 0, Rz = 0; Point3d origin = Point3d.Unset;
            double dx = 1, dy = 1, dz = 1;

            if (!DA.GetData(0, ref geo)) return;
            if (!DA.GetData(1, ref P)) return;
            if (!DA.GetData(2, ref Rx)) return;
            if (!DA.GetData(3, ref Ry)) return;
            if (!DA.GetData(4, ref Rz)) return;
            if (!DA.GetData(5, ref origin)) return;
            if (!DA.GetData(6, ref dx)) return;
            if (!DA.GetData(7, ref dy)) return;
            if (!DA.GetData(8, ref dz)) return;

            int N = Rx * Ry * Rz;
            if (P.Length != N)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Pressure array length mismatch.");
                return;
            }

            /* 1. triangulate object */
            var meshes = Mesh.CreateFromBrep(geo, new MeshingParameters { SimplePlanes = true });
            var m = new Mesh(); foreach (var s in meshes) m.Append(s); m.Normals.ComputeNormals();

            /* 2. sample pressure per vertex */
            float pMin = float.MaxValue, pMax = float.MinValue;
            var samples = new float[m.Vertices.Count];
            for (int i = 0; i < m.Vertices.Count; i++)
            {
                var v = m.Vertices[i];
                int ix = (int)Math.Floor((v.X - origin.X) / dx);
                int iy = (int)Math.Floor((v.Y - origin.Y) / dy);
                int iz = (int)Math.Floor((v.Z - origin.Z) / dz);
                ix = Math.Max(0, Math.Min(Rx - 1, ix));
                iy = Math.Max(0, Math.Min(Ry - 1, iy));
                iz = Math.Max(0, Math.Min(Rz - 1, iz));
                int id = ix + iy * Rx + iz * Rx * Ry;
                float p = P[id];
                samples[i] = p;
                if (p < pMin) pMin = p;
                if (p > pMax) pMax = p;
            }
            float span = Math.Max(1e-6f, pMax - pMin);
            m.VertexColors.CreateMonotoneMesh(Color.Black);
            for (int i = 0; i < samples.Length; i++)
            {
                float t = (samples[i] - pMin) / span;          // 0…1
                Color col = Color.FromArgb(255,
                    (int)(t * 255), 0, (int)((1 - t) * 255));   // blue→red
                m.VertexColors[i] = col;
            }
            DA.SetData(0, m);
        }

        public override Guid ComponentGuid => new Guid("a9ed1e3c-1d64-46f1-99f3-11d9e5a2eed1");
        protected override Bitmap Icon => null;
    }
}
