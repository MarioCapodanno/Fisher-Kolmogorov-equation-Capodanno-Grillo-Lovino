# Fisher–Kolmogorov Equation

### Numerical Methods for Partial Differential Equations – Project 2024/25 - Politecnico di Milano

| Students |
|-------- |
| [Capodanno Mario](https://github.com/MarioCapodanno) |
| [Grillo Valerio](https://github.com/Valegrl) |
| [Lovino Emanuele](https://github.com/EmanueleLovino) |

* Professor: _Alfio Maria Quarteroni_
* Assistant Professor: _Michele Bucelli_

## Full Repository Structure

```text
common/                           <- Shared CMake settings and utilities
└── cmake-common.cmake            <- Shared CMake macros (deal.II configuration)

1D/                               <- 1D prion-like spreading solution study
├── CMakeLists.txt                
├── src/
│   ├── FisherKolmogorov1D.hpp    <- Declaration of the 1D problem class
│   ├── FisherKolmogorov1D.cpp    <- Implementation: setup, assembly, solver, output
│   └── main_1D.cpp               <- Entry point, constructs problem
└── scripts/
    └── plot-solution.py          <- Python script for plotting VTK results

3D/                               <- Brain-specific 3D study
├── CMakeLists.txt                <- Build configuration
├── src/
│   ├── FisherKolmogorov3D.hpp    <- Declaration of the 3D problem class
│   ├── FisherKolmogorov3D.cpp    <- Implementation: setup, assembly, solver, diffusion tensors, output
│   ├── main_3D.cpp               <- Entry point, constructs problem
│   ├── ParameterReader.hpp       <- Parses `parameters.prm`, selects diffusion tensor
│   └── DiffusionTensor.hpp       <- Defines isotropic/anisotropic tensors
└── scripts/
    └── brain_script.geo          <- Gmsh script for `.msh` conversion

3D_convergence/                   <- Convergence study on unit cube
├── CMakeLists.txt                <- Build configuration
├── src/
│   ├── FisherKolmogorov3D.hpp    <- Declaration of 3D convergence problem class
│   ├── FisherKolmogorov3D.cpp    <- Implementation: setup, assembly, solver, errors, output
│   ├── main_3D.cpp               <- Entry point, constructs problem
│   └── ParameterReader.hpp       <- ParameterHandler wrapper
└── scripts/
    ├── cube.geo                  <- Gmsh geometry for unit cube
    └── plot-convergence.py       <- Python script for convergence plots
```

## Prerequisites

* **deal.II** (≥ 9.0) with MPI support.
* **C++** compiler (C++11 or later).
* **CMake** ≥ 3.12.
* **Gmsh** ≥ 4.0.
* **Python** 3.6+ with:

  * `numpy`, `pandas`, `matplotlib`

<br></br>
Before starting, ensure that the required environment modules are loaded:
```bash
module load gcc-glibc dealii
```

## 1D Solution Study

Folder: `1D/`

```text
1D/                               
├── CMakeLists.txt                
├── src/
│   ├── FisherKolmogorov1D.hpp   
│   ├── FisherKolmogorov1D.cpp    
│   └── main_1D.cpp               
└── scripts/
    └── plot-solution.py         
```

### Running the 1D Study

1. **Build** (from `1D/`):

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```
2. **Run**:
   ```bash
   ./main_1D [d] [α]
   ```
   Default values are `d = 0.0001` and `α = 1.0`
3. **Plot**:

   ```bash
   python <script-path>/plot-solution.py <vtk-files-path>
   ```
   Default:

   ```bash
   python ../scripts/plot-solution.py ../build/
   ```

   Generates `solution.pdf` in `1D/build/` with the concentration profile over time.


## Brain‐Specific 3D Study

Folder: `3D/`

```text
3D/                             
├── CMakeLists.txt               
├── src/
│   ├── FisherKolmogorov3D.hpp   
│   ├── FisherKolmogorov3D.cpp   
│   ├── main_3D.cpp               
│   ├── ParameterReader.hpp      
│   └── DiffusionTensor.hpp      
└── scripts/
    └── brain_script.geo                       
```

### Preparing the Brain Mesh

1. **Create** the folder for meshes (from `3D/`):

   ```bash
   mkdir mesh
   ```

2. **Convert** your mesh from `.stl` to `.msh` if needed:

   * Place your STL file inside the `3D/mesh/` directory (**STL must be a closed surface.** If not, `Volume` creation will fail).

   * Modify the `.geo` file to specify the right `.stl` file.
      ```geo
      Merge "../mesh/brain-h3.0.stl";
      ```
   * Run Gmsh with the script:
      ```bash
      gmsh scripts/brain_script.geo -3 -o <save-mesh-path>/output.msh
      ```
      In our case:
      ```bash
      gmsh scripts/brain_script.geo -3 -o mesh/brain-h3.0.msh
      ```

3. **Place** your mesh file (e.g., `brain-h3.0.msh`) in the `3D/mesh/` directory (if you have not already done so) before running.

### Running the 3D Study

4. **Build** (from `3D/`):

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

5. **Configuring the 3D Study**

   In the working directory you can find `parameters.prm` file. This file lets you tune mesh, physical, time‐stepping, solver, and diffusion‐tensor settings:

   - **Mesh & geometry parameters** (`Mesh & geometry parameters`)
      - `Degree`  
         Polynomial degree for the finite‐element discretization (e.g. 1 or 2).

   - **Physical constants** (`Physical constants`)
      - `Dext`  
         Extra‐cellular diffusion coefficient.
      - `Daxn`  
         Axonal diffusion coefficient.
      - `Alpha coefficient`  
         Reaction rate coefficient.

   - **Time stepping parameters** (`Time stepping parameters`)
      - `T`  
         Final simulation time.
      - `deltat`  
         Time step size.

   - **Solver parameters** (`Solver parameters`)
      - `Max Newton iterations`  
         Maximum Newton‐solver iterations per timestep.
      - `Newton tolerance`  
         Convergence tolerance for the Newton solver.
      - `Max CG iterations`  
         Maximum conjugate‐gradient iterations.
      - `CG tolerance factor`  
         Relative tolerance factor for the CG solver.

   - **Diffusion tensor parameters** (`Diffusion tensor parameters`)
      - `Diffusion tensor type`  
         Isotropic | Radial | Circumferential.
      - `Center X`, `Center Y`, `Center Z`  
         Coordinates defining the center for diffusion.

6. **Run**:

   ```bash
   mpirun -np <N-processes> ./main_3D parameters.prm
   ```

   The code will load `mesh/brain-h3.0.msh`, simulate, and write output files in the working directory.

## 3D Convergence Study

Folder: `3D_convergence/`

```text
3D_convergence/                 
├── CMakeLists.txt               
├── src/
│   ├── FisherKolmogorov3D.hpp    
│   ├── FisherKolmogorov3D.cpp    
│   ├── main_3D.cpp               
│   └── ParameterReader.hpp       
└── scripts/
    ├── cube.geo                  
    └── plot-convergence.py                                             
```

### Generating Uniform Cube Meshes

1. **Create** mesh folder (from `3D_convergence/`):

   ```bash
   mkdir mesh
   ```
2. **Generate** meshes:

   ```bash
   for H in 2 4 8 16 32; do
      tag=$(echo $H | sed 's/\./p/')
      gmsh -3 -setnumber h $H scripts/cube.geo -o mesh/mesh-cube-$tag.msh
   done
   ```
   or once at a time:

   ```bash
   gmsh -3 -setnumber h <h-size> scripts/cube.geo -o mesh/output.msh
   ```

   Where `h-size` is the number of segments per edge. Example:

   ```bash
   gmsh -3 -setnumber h 16 scripts/cube.geo -o mesh/mesh-cube-16.msh
   ```

### Running the Convergence Study

3. **Build** (from `3D_convergence/`):

   ```bash
   mkdir -p build && cd build
   cmake ..
   make
   ```

4. **Configuring the Convergence Study**

   In the working directory you can find `parameters.prm` file. It controls mesh, physical, time‐stepping, and solver options for each cube mesh:

   - **Mesh & geometry parameters** (`Mesh & geometry parameters`)
      - `Degree`  
         Polynomial degree for all tests.

   - **Physical constants** (`Physical constants`)
      - `Dext`  
         Extra‐cellular diffusion coefficient.
      - `Daxn`  
         Axonal diffusion coefficient.
      - `Alpha coefficient`  
         Reaction rate coefficient.

   - **Time stepping parameters** (`Time stepping parameters`)
      - `T`  
         Final time for each simulation.
      - `deltat`  
         Time step size.
      - `Theta`  
         Implicitness factor for the time‐integration scheme (1.0 = fully implicit).

   - **Solver parameters** (`Solver parameters`)
      - `Max Newton iterations`  
         Maximum Newton‐solver iterations per run.
      - `Newton tolerance`  
         Convergence tolerance for Newton steps.
      - `Max CG iterations`  
         Maximum conjugate‐gradient iterations.
      - `CG tolerance factor`  
         Relative tolerance factor for the CG solver.

5. **Run**:

   ```bash
   mpirun -np <N-processes> ./main_3D parameters.prm
   ```

6. **Plot**:

   ```bash
   python ../scripts/plot-convergence.py convergence.csv
   ```

   Produces `convergence.pdf` showing error vs. `h` and reference slopes.