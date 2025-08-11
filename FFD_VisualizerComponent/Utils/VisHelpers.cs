using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace FFD_VisualizerComponent.Utils
{
    internal static class VisHelpers
    {
        /// <summary>Blue→Red ramp over [sMin, sMax].</summary>
        public static Color Ramp(double s, double sMin, double sMax)
        {
            if (double.IsNaN(s) || double.IsInfinity(s)) return Color.Black;
            double t = (s - sMin) / (sMax - sMin);
            if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
            int r = (int)(t * 255.0);
            int b = (int)((1.0 - t) * 255.0);
            return Color.FromArgb(255, r, 0, b);
        }

        /// <summary>Find scalar min/max on a single slice (axis 0=X,1=Y,2=Z).</summary>
        public static void FindSliceMinMax(int axis, int slice, int Rx, int Ry, int Rz,
                                           Func<int, double> scalar,
                                           out double sMin, out double sMax)
        {
            sMin = double.MaxValue; sMax = -double.MaxValue;

            if (axis == 2) // Z → XY slice at k
            {
                int k = slice;
                for (int j = 0; j < Ry; ++j)
                    for (int i = 0; i < Rx; ++i)
                    {
                        int id = i + j * Rx + k * Rx * Ry;
                        double s = scalar(id);
                        if (s < sMin) sMin = s;
                        if (s > sMax) sMax = s;
                    }
            }
            else if (axis == 1) // Y → XZ slice at j
            {
                int j = slice;
                for (int k = 0; k < Rz; ++k)
                    for (int i = 0; i < Rx; ++i)
                    {
                        int id = i + j * Rx + k * Rx * Ry;
                        double s = scalar(id);
                        if (s < sMin) sMin = s;
                        if (s > sMax) sMax = s;
                    }
            }
            else // axis == 0, X → YZ slice at i
            {
                int i = slice;
                for (int k = 0; k < Rz; ++k)
                    for (int j = 0; j < Ry; ++j)
                    {
                        int id = i + j * Rx + k * Rx * Ry;
                        double s = scalar(id);
                        if (s < sMin) sMin = s;
                        if (s > sMax) sMax = s;
                    }
            }
        }

        /// <summary>
        /// Build a per-cell colored quad mesh for a slice. Colors are set per-vertex.
        /// </summary>
        public static Mesh BuildSliceMesh(int axis, int slice,
                                          int Rx, int Ry, int Rz,
                                          double dx, double dy, double dz,
                                          Point3d origin,
                                          Func<int, double> scalar,
                                          double sMin, double sMax)
        {
            var mesh = new Mesh();

            if (axis == 2) // Z → XY plane at k
            {
                double z = origin.Z + (slice + 0.5) * dz;
                int k = slice;
                for (int j = 0; j < Ry; ++j)
                {
                    double y0 = origin.Y + j * dy;
                    double y1 = y0 + dy;
                    for (int i = 0; i < Rx; ++i)
                    {
                        double x0 = origin.X + i * dx;
                        double x1 = x0 + dx;

                        int id = i + j * Rx + k * Rx * Ry;
                        Color c = Ramp(scalar(id), sMin, sMax);

                        int v0 = mesh.Vertices.Add(new Point3d(x0, y0, z));
                        int v1 = mesh.Vertices.Add(new Point3d(x1, y0, z));
                        int v2 = mesh.Vertices.Add(new Point3d(x1, y1, z));
                        int v3 = mesh.Vertices.Add(new Point3d(x0, y1, z));

                        mesh.Faces.AddFace(v0, v1, v2, v3);
                        mesh.VertexColors.SetColor(v0, c);
                        mesh.VertexColors.SetColor(v1, c);
                        mesh.VertexColors.SetColor(v2, c);
                        mesh.VertexColors.SetColor(v3, c);
                    }
                }
            }
            else if (axis == 1) // Y → XZ plane at j
            {
                double y = origin.Y + (slice + 0.5) * dy;
                int j = slice;
                for (int k = 0; k < Rz; ++k)
                {
                    double z0 = origin.Z + k * dz;
                    double z1 = z0 + dz;
                    for (int i = 0; i < Rx; ++i)
                    {
                        double x0 = origin.X + i * dx;
                        double x1 = x0 + dx;

                        int id = i + j * Rx + k * Rx * Ry;
                        Color c = Ramp(scalar(id), sMin, sMax);

                        int v0 = mesh.Vertices.Add(new Point3d(x0, y, z0));
                        int v1 = mesh.Vertices.Add(new Point3d(x1, y, z0));
                        int v2 = mesh.Vertices.Add(new Point3d(x1, y, z1));
                        int v3 = mesh.Vertices.Add(new Point3d(x0, y, z1));

                        mesh.Faces.AddFace(v0, v1, v2, v3);
                        mesh.VertexColors.SetColor(v0, c);
                        mesh.VertexColors.SetColor(v1, c);
                        mesh.VertexColors.SetColor(v2, c);
                        mesh.VertexColors.SetColor(v3, c);
                    }
                }
            }
            else // axis == 0 → X, YZ plane at i
            {
                double x = origin.X + (slice + 0.5) * dx;
                int i = slice;
                for (int k = 0; k < Rz; ++k)
                {
                    double z0 = origin.Z + k * dz;
                    double z1 = z0 + dz;
                    for (int j = 0; j < Ry; ++j)
                    {
                        double y0 = origin.Y + j * dy;
                        double y1 = y0 + dy;

                        int id = i + j * Rx + k * Rx * Ry;
                        Color c = Ramp(scalar(id), sMin, sMax);

                        int v0 = mesh.Vertices.Add(new Point3d(x, y0, z0));
                        int v1 = mesh.Vertices.Add(new Point3d(x, y1, z0));
                        int v2 = mesh.Vertices.Add(new Point3d(x, y1, z1));
                        int v3 = mesh.Vertices.Add(new Point3d(x, y0, z1));

                        mesh.Faces.AddFace(v0, v1, v2, v3);
                        mesh.VertexColors.SetColor(v0, c);
                        mesh.VertexColors.SetColor(v1, c);
                        mesh.VertexColors.SetColor(v2, c);
                        mesh.VertexColors.SetColor(v3, c);
                    }
                }
            }

            mesh.Normals.ComputeNormals();
            mesh.Compact();
            return mesh;
        }
    }
}
