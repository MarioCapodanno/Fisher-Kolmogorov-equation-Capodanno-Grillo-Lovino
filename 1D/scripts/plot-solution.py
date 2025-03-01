#!/usr/bin/env python
import glob
import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def sort_key(filename):
    """
    Extract the progressive number from a file name that follows the pattern:
    "output-0.100000_#.vtu". Returns the numeric value after the underscore.
    """
    base = os.path.basename(filename)  # e.g. "output-0.100000_3.vtu"
    parts = base.split('_')
    if len(parts) < 2:
        return 0
    num_str = os.path.splitext(parts[-1])[0]
    try:
        return float(num_str)
    except ValueError:
        return 0

# Read .vtu files from the current directory
vtu_files = sorted(glob.glob("*.vtu"), key=sort_key)
if not vtu_files:
    raise FileNotFoundError("No .vtu files found in the current directory.")

# Read each file, extract the x-coordinates and the scalar field (u)
x_coords = None
solutions = []

for filename in vtu_files:
    mesh = pv.read(filename)
    x = mesh.points[:, 0]  # use the first coordinate as space
    scalar = mesh["u"]     # scalar field name

    # Sort by x so that curves line up consistently
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    scalar_sorted = scalar[sort_idx]

    if x_coords is None:
        x_coords = x_sorted
    else:
        if not np.allclose(x_coords, x_sorted):
            raise ValueError("Inconsistent x-coordinates among time steps.")

    solutions.append(scalar_sorted)

solutions = np.array(solutions)  # shape: (Nt, Nx)
Nt, Nx = solutions.shape

# Create a "time" array (using file index as time here)
time = np.arange(Nt)

fig, ax = plt.subplots(figsize=(8, 5))

# Use the reversed "jet" colormap by adding "_r"
cmap = plt.get_cmap("jet_r")
norm = plt.Normalize(vmin=time.min(), vmax=time.max())

# Plot from the last time step to the first, so the first is on top
for t, sol in zip(time[::-1], solutions[::-1]):
    color = cmap(norm(t))
    # Fill the area under the curve from y=0 to y=sol
    ax.fill_between(x_coords, 0, sol, color=color)

ax.set_xlabel("Space coordinate (x)")
ax.set_ylabel("Solution/Concentration (c)")

# Add a colorbar to indicate time mapping
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Time step")

plt.savefig("solution.pdf")