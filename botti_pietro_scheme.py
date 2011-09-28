#!/usr/bin/env python
"""
A prototype implementation of the Navier Stokes scheme proposed in

  Lorenzo Botti and Daniele A. Di Pietro,
  "A pressure-correction scheme for convection-dominated incompressible
  flows with discontinuous velocity and continuous pressure"
  Journal of Computational Physics 230 (2011)
"""

from dolfin import *

# Parameters
d, m = 2, 10
T0, T1 = 0.0, 0.2
dt = 0.1

# Define mesh
if d == 2:
    mesh = UnitSquare(m, m)
elif d == 3:
    mesh = UnitCube(m, m, m)

# TODO: Define subdomains

# Get UFL geometric quantities
cell = mesh.ufl_cell()
x = cell.x
n = cell.n

# TODO: Define function spaces


# TODO: Define functions


# TODO: Define forms


# TODO: Define boundary conditions


# Run time loop
t = T0
while t < T1:
    # TODO: Solve advection-diffusion equations

    # TODO: Solve pressure equation

    t += dt

# TODO: Postprocessing
