// Usage example: gmsh scripts/brain_script.geo -3 -o mesh/brain-h3.0.msh

// Import the STL file
Merge "../mesh/brain-h3.0.stl";

// Run a coherence step to merge duplicate nodes
Coherence;

// Define a surface loop from the imported surface(s)
// If the STL contains a single closed surface, you can simply use its ID (commonly 1)
// If it contains multiple surfaces, list them separated by commas (e.g., {1,2,3,...})
Surface Loop(1) = {1};

// Create a volume from the surface loop
Volume(1) = {1};

// Instruct Gmsh to perform 3D mesh generation
Mesh 3;
