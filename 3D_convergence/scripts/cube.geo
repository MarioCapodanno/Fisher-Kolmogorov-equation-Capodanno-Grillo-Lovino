// Usage example:
//    gmsh -3 -setnumber h 16 cube.geo -o mesh-cube-16.msh
SetFactory("OpenCASCADE");

// 1) number of segments per edge

N = h + 1;

// 2) build the geometry with the built-in box
Box(1) = {0, 0, 0, 1, 1, 1};

// 3) tag the volume (optional, but often useful)
Physical Volume("Cube") = {1};

// 4) transfinite (structured) mesh constraints
Transfinite Line    {1:12}       = N;            // split each of the 12 edges into N-1 segments
Transfinite Surface {1:6};                        // inherit the line splits onto each of the 6 faces
Transfinite Volume  {1}          = {N, N, N};    // carve the volume into N×N×N blocks

// 5) mesh options
Mesh.ElementOrder  = 1;      // linear tets
Mesh.Algorithm3D   = 1;      // Delaunay tetrahedralization

// 6) actually generate the 3D mesh
Mesh 3;