using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FFD_SolverComponent.Utils
{
    internal static class FaceVoxelizer
    {
        public static List<int[]> GetFaceVoxelSets(Brep brep, int rx, int ry, int rz)
        {
            var lists = new List<int>[brep.Faces.Count];
            for (int f = 0; f < brep.Faces.Count; f++)
                lists[f] = new List<int>(Math.Max(256, rx * ry / 8));

            // Flatten identical to solver/visualizer
            Func<int, int, int, int> Flatten = (x, y, z) => x + y * rx + z * rx * ry;

            // Grid AABB (the solver’s origin = aabb.Min; voxel sizes from this too)
            BoundingBox aabb = brep.GetBoundingBox(true);

            for (int f = 0; f < brep.Faces.Count; f++)
            {
                var face = brep.Faces[f];

                // Face center (UV mid)
                Interval du = face.Domain(0);
                Interval dv = face.Domain(1);
                Point3d c = face.PointAt(0.5 * (du.T0 + du.T1), 0.5 * (dv.T0 + dv.T1));

                // Try to get a reliable inward normal
                Vector3d nInward;
                Plane pl;
                if (face.TryGetPlane(out pl, 1e-6))
                {
                    nInward = pl.Normal;
                    nInward.Unitize();
                    // flip to point toward the domain center
                    Vector3d toCenter = aabb.Center - c;
                    if (toCenter * nInward < 0.0) nInward = -nInward;
                }
                else
                {
                    // Fallback: use direction from face center to domain center
                    nInward = aabb.Center - c;
                    if (nInward.IsTiny(1e-12)) { lists[f] = new List<int>(0); continue; }
                    nInward.Unitize();
                }

                // Dominant axis of inward direction
                double ax = Math.Abs(nInward.X);
                double ay = Math.Abs(nInward.Y);
                double az = Math.Abs(nInward.Z);

                int side = -1; // 0:-X 1:+X 2:-Y 3:+Y 4:-Z 5:+Z
                if (ax >= ay && ax >= az)
                {
                    side = (nInward.X > 0.0) ? 0 : 1;
                }
                else if (ay >= az)
                {
                    side = (nInward.Y > 0.0) ? 2 : 3;
                }
                else
                {
                    side = (nInward.Z > 0.0) ? 4 : 5;
                }

                // Emit the voxel indices for that slab
                switch (side)
                {
                    case 0: // -X → i = 0
                        for (int j = 0; j < ry; ++j)
                            for (int k = 0; k < rz; ++k)
                                lists[f].Add(Flatten(0, j, k));
                        break;

                    case 1: // +X → i = rx - 1
                        for (int j = 0; j < ry; ++j)
                            for (int k = 0; k < rz; ++k)
                                lists[f].Add(Flatten(rx - 1, j, k));
                        break;

                    case 2: // -Y → j = 0
                        for (int i = 0; i < rx; ++i)
                            for (int k = 0; k < rz; ++k)
                                lists[f].Add(Flatten(i, 0, k));
                        break;

                    case 3: // +Y → j = ry - 1
                        for (int i = 0; i < rx; ++i)
                            for (int k = 0; k < rz; ++k)
                                lists[f].Add(Flatten(i, ry - 1, k));
                        break;

                    case 4: // -Z → k = 0
                        for (int i = 0; i < rx; ++i)
                            for (int j = 0; j < ry; ++j)
                                lists[f].Add(Flatten(i, j, 0));
                        break;

                    case 5: // +Z → k = rz - 1
                        for (int i = 0; i < rx; ++i)
                            for (int j = 0; j < ry; ++j)
                                lists[f].Add(Flatten(i, j, rz - 1));
                        break;
                }
            }

            var result = new List<int[]>(brep.Faces.Count);
            for (int f = 0; f < brep.Faces.Count; f++) result.Add(lists[f].ToArray());
            return result;
        }
    }
}
