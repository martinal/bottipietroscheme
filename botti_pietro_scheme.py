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
d, m = 2, 20
dt = 0.1
T0, T1 = 0.0, dt/2 # Just one step for now

# Define mesh
if d == 2:
    mesh = UnitSquare(m, m)
elif d == 3:
    mesh = UnitCube(m, m, m)

# TODO: Define subdomains

# Get UFL geometric quantities
cell = mesh.ufl_cell()
x = cell.x
n = FacetNormal(mesh)

# Define function spaces
k = 1
V = VectorFunctionSpace(mesh, "DG", k)
P = FunctionSpace(mesh, "CG", 1)

# Define functions
uh = Function(V)
ph = Function(P)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(P)
q = TestFunction(P)
nu = Constant(1.0, cell=cell) # TODO: Value?
eta = Constant(3.0, cell=cell)

# TODO: Define forms
h_T = CellSize(mesh)
h_dT = FacetArea(mesh)
h_p = h_T('+') / h_dT('+')
h_m = h_T('-') / h_dT('-')
h_F = conditional(lt(h_p, h_m), h_p, h_m) # h_F = h_p < h_m ? h_p: h_m

h_F = avg(h_T)
nm = n('-')
np = n('+')

i, j = indices(2)

a_h  = inner(grad(u), grad(v))*dx
a_h += (eta('+')*k**2/h_F)*dot(jump(u),jump(v))*dS
a_h -= dot( avg(grad(u))*np, jump(v) )*dS
a_h -= dot( avg(grad(v))*np, jump(u) )*dS
a_h += (eta*k**2/h_T)*dot(u,v)*ds
a_h -= dot(grad(u)*n, v)*ds
a_h -= dot(grad(v)*n, u)*ds

b_h = (eta*k**2/h_T)*dot(uh,v)*ds
b_h -= dot(grad(v)*n, uh)*ds


#a_h -= ((avg(u[j].dx(i)) * nm[i]) * jump(v[j]))*dS # FIXME: FFC bug!
#a_h -= ((avg(v[j].dx(i)) * nm[i]) * jump(u[j]))*dS

#a_h -= dot( avg(grad(u)) * nm, jump(v) )*dS # FIXME: FFC bug!
#a_h -= dot( avg(grad(v)) * nm, jump(u) )*dS

#a_h -= dot( avg(grad(u))*np, jump(v) )*dS # FIXME: FFC bug!
#a_h -= dot( avg(grad(v))*np, jump(u) )*dS

#a_h -= (dot(avg(grad(u[j])), n) * jump(v[j]))*dS # TODO: Support this in ufl?




# TODO: Define boundary conditions



# Assemble matrices
uh.assign(project(as_vector((x[0]**2, 0)), V))
A_int = assemble(a_h)
b_int = assemble(b_h)
solve(A_int, uh.vector(), b_int)

# Run time loop
t = T0
while t < T1:
    # TODO: Solve advection-diffusion equations

    # TODO: Solve pressure equation

    t += dt

# TODO: Postprocessing
plot(uh)
plot(ph)
interactive()
