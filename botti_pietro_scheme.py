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

# Define geometry
if d == 2:
    mesh = UnitSquare(m, m)
elif d == 3:
    mesh = UnitCube(m, m, m)
cell = mesh.ufl_cell()
x = cell.x
n = cell.n

# Define function spaces
