using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace FFD_VisualizerComponent.Utils
{
        internal enum ColorMap
        {
            BlueRed = 0,  // simple blue→red (legacy)
            Plasma = 1   // higher definition (Matplotlib-like)
        }

        internal static class VisHelpers
        {
            private static double Normalize(double s, double sMin, double sMax)
            {
                double t = (s - sMin) / (sMax - sMin);
                if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                return t;
            }

            /// <summary>Legacy ramp (blue→red) kept for compatibility.</summary>
            public static Color Ramp(double s, double sMin, double sMax)
            {
                double t = Normalize(s, sMin, sMax);
                int r = (int)(t * 255.0);
                int b = (int)((1.0 - t) * 255.0);
                return Color.FromArgb(255, r, 0, b);
            }

            /// <summary>New ramp with selectable colormap.</summary>
            public static Color Ramp(double s, double sMin, double sMax, ColorMap map)
            {
                double t = Normalize(s, sMin, sMax);
                switch (map)
                {
                    case ColorMap.Plasma: return PlasmaColor(t);
                    case ColorMap.BlueRed:
                    default:
                        int r = (int)(t * 255.0);
                        int b = (int)((1.0 - t) * 255.0);
                        return Color.FromArgb(255, r, 0, b);
                }
            }

            /// <summary>
            /// Plasma colormap (Matplotlib-like), sampled via piecewise-linear
            /// interpolation of anchor colors. t ∈ [0,1].
            /// </summary>
            private static Color PlasmaColor(double t)
            {
                // Anchor points (t, #RRGGBB) from Matplotlib plasma samples.
                // 0.00: #0d0887, 0.13: #42039e, 0.25: #6a00a8, 0.38: #8f0da4,
                // 0.50: #b12a90, 0.63: #cc4778, 0.75: #e16462, 0.88: #f2844b, 1.00: #f0f921
                int[] stops = {
                0x0d0887, 0x42039e, 0x6a00a8, 0x8f0da4,
                0xb12a90, 0xcc4778, 0xe16462, 0xf2844b, 0xf0f921
            };
                double[] pos = { 0.00, 0.13, 0.25, 0.38, 0.50, 0.63, 0.75, 0.88, 1.00 };

                if (t <= pos[0]) return FromHex(stops[0]);
                if (t >= pos[pos.Length - 1]) return FromHex(stops[stops.Length - 1]);

                // find segment
                int i = 0;
                for (int k = 1; k < pos.Length; ++k)
                {
                    if (t <= pos[k]) { i = k - 1; break; }
                }
                double t0 = pos[i], t1 = pos[i + 1];
                double u = (t - t0) / (t1 - t0);

                Color c0 = FromHex(stops[i]);
                Color c1 = FromHex(stops[i + 1]);

                int R = (int)Math.Round(c0.R + u * (c1.R - c0.R));
                int G = (int)Math.Round(c0.G + u * (c1.G - c0.G));
                int B = (int)Math.Round(c0.B + u * (c1.B - c0.B));
                if (R < 0) R = 0; if (R > 255) R = 255;
                if (G < 0) G = 0; if (G > 255) G = 255;
                if (B < 0) B = 0; if (B > 255) B = 255;
                return Color.FromArgb(255, R, G, B);
            }

            private static Color FromHex(int rgb)
            {
                int R = (rgb >> 16) & 0xFF;
                int G = (rgb >> 8) & 0xFF;
                int B = rgb & 0xFF;
                return Color.FromArgb(255, R, G, B);
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
            /// Default colormap: BlueRed (legacy).
            /// </summary>
            public static Mesh BuildSliceMesh(int axis, int slice,
                                              int Rx, int Ry, int Rz,
                                              double dx, double dy, double dz,
                                              Point3d origin,
                                              Func<int, double> scalar,
                                              double sMin, double sMax)
            {
                return BuildSliceMesh(axis, slice, Rx, Ry, Rz, dx, dy, dz, origin, scalar, sMin, sMax, ColorMap.BlueRed);
            }

            /// <summary>
            /// Build a per-cell colored quad mesh for a slice with selectable colormap.
            /// </summary>
            public static Mesh BuildSliceMesh(int axis, int slice,
                                              int Rx, int Ry, int Rz,
                                              double dx, double dy, double dz,
                                              Point3d origin,
                                              Func<int, double> scalar,
                                              double sMin, double sMax,
                                              ColorMap map)
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
                            Color c = Ramp(scalar(id), sMin, sMax, map);

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
                            Color c = Ramp(scalar(id), sMin, sMax, map);

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
                            Color c = Ramp(scalar(id), sMin, sMax, map);

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
