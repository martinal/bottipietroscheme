#!/usr/bin/env python
"""
A prototype implementation of the Navier Stokes scheme proposed in

  Lorenzo Botti and Daniele A. Di Pietro,
  "A pressure-correction scheme for convection-dominated incompressible
  flows with discontinuous velocity and continuous pressure"
  Journal of Computational Physics 230 (2011)
"""

import sys
from dolfin import *

if 1:
    # Ugly but sure way to set quadrature rule
    ffc_opt = {}
    ffc_opt['representation'] = 'quadrature'
    #ffc_opt['quadrature_degree'] = 7
    def assemble(*args, **kwargs):
        kwargs['form_compiler_parameters'] = ffc_opt
        return dolfin.assemble(*args, **kwargs)

# Parameters
d = 2
m = int(sys.argv[1]) if len(sys.argv) == 2 else 16
k = 1
dtvalue = 1e-1
steps = 20
T0, T1 = 0.0, dtvalue*(steps+0.1)

yscale = 10
aspectratio = 1.0

# Define mesh
if d == 2:
    mesh = UnitSquare(m, int(m*yscale/aspectratio))
    mesh.coordinates()[:,1] *= yscale
elif d == 3:
    mesh = UnitCube(m, m, m)

# Define subdomains
eps = 0.01 / m
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < eps and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > (yscale - eps) and on_boundary

boundaries = MeshFunction("uint", mesh, d-1)
boundaries.set_all(0)
Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)

# Attach boundary indicators to a boundary integration measure
dsb = ds[boundaries]

# Get UFL geometric quantities
cell = mesh.ufl_cell()
x = cell.x # TODO: Implement the FacetNormal trick in PyDOLFIN, so we can write SpatialCoordinates(mesh)
def SpatialCoordinates(mesh):
    return mesh.ufl_cell().x
n = FacetNormal(mesh)

# Define function spaces
V = VectorFunctionSpace(mesh, "DG", k)
Vf = VectorFunctionSpace(mesh, "CG", 3)
P = FunctionSpace(mesh, "CG", 1)

# Define functions
uh = Function(V)
uhp = Function(V)
ph = Function(P)
php = Function(P)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(P)
q = TestFunction(P)
eta = Constant(5.0, cell=cell)
f = Function(Vf)
dt = Constant(dtvalue, cell=cell)
dti = 1.0 / dt

# Boundary condition functions for u
# (setting one of these to 0,0 is equivalent to
#  removing the integral in this formulation)
uz = as_vector((0, 0))
ub0, ub1, ub2 = uz, uz, uz # Zero as default
ub1 = as_vector((0, x[0]*(1.0-x[0]))) # Parabolic profile in y direction
#ub2 = as_vector((0, x[0]*(1.0-x[0])))

# Define penalty terms
nm = n('-')
np = n('+')
h_T = CellSize(mesh)
h_dT = FacetArea(mesh)

# FIXME: Remove when dolfin gets a fix for the facetarea bug
h_dT = 0.5/m**2 # Hardcoded area of 1.0/16 length perfect triangle

h_F_ext = h_T / h_dT
if 1:
    h_p = h_T('+')
    h_m = h_T('-')
    h_T_min = conditional(lt(h_p, h_m), h_p, h_m)
    # h_T_min = h_p < h_m ? h_p: h_m
    h_F_int = h_T_min / h_dT #('+')
else:
    h_F_int = avg(h_T)

# Penalty term
pen_ext = eta * k**2 / h_F_ext
pen_int = eta('+') * k**2 / h_F_int

# Free indices for use with implicit summation
i, j = indices(2)

# Left hand side for u equations
a_u  = inner(grad(u), grad(v))*dx
a_u += ( pen_int*dot(jump(u),jump(v))
        - dot( avg(grad(u))*np, jump(v) )
        - dot( avg(grad(v))*np, jump(u) ) ) * dS
a_u += ( pen_ext*dot(u,v)
        - dot(grad(u)*n, v)
        - dot(grad(v)*n, u) ) * ds
a_u += dti*dot(u, v)*dx # Time derivative term

# Right hand side for u equations
b_u = -dot(f,v)*dx                       # Forcing term
b_u += dti*dot(uhp,v)*dx                 # Time derivative term
b_u -= dot(v, 2*grad(ph) - grad(php))*dx # Pressure coupling term

if 0: # FIXME: What is the right formulation for this term?
    b_u += pen_ext*dot(ub0,v)*dsb(0)
    b_u += pen_ext*dot(ub1,v)*dsb(1)
    b_u += pen_ext*dot(ub2,v)*dsb(2)
else:
    b_u += pen_ext*dot(ub0,n)*dot(v,n)*dsb(0)
    b_u += pen_ext*dot(ub1,n)*dot(v,n)*dsb(1)
    b_u += pen_ext*dot(ub2,n)*dot(v,n)*dsb(2)

b_u -= dot(grad(v)*n, ub0)*dsb(0)
b_u -= dot(grad(v)*n, ub1)*dsb(1)
b_u -= dot(grad(v)*n, ub2)*dsb(2)

# Pressure Poisson equation
a_p = dot(grad(p),grad(q))*dx
b_p  = dot(grad(php), grad(q))*dx
b_p -= dti*div(uh)*q*dx
b_p -= dti('+')*dot(np, jump(uh))*avg(q)*dS
b_p += dti*dot(n, uh-ub0)*q*dsb(0)
b_p += dti*dot(n, uh-ub1)*q*dsb(1)
b_p += dti*dot(n, uh-ub2)*q*dsb(2)

# BC for pressure, zero on outlet boundary
pbc = DirichletBC(P, 0, Outlet())

# Setup analytical solution and corresponding forcing term
if 0: # From testing at an earlier stage, not compatible with current problem setup
    u0 = as_vector((0, 0))
    ub0.assign(project(u0, V))
    ub1.assign(project(u0, V))
    ub2.assign(project(u0, V))

    f0 = as_vector((0, 0))
    f.assign(project(f0, Vf))

# Assemble time independent matrices
A_u = assemble(a_u)
A_p = assemble(a_p)

# Storage of results
times = []
errors = []
uchanges = []
ufile = File('u.pvd')
pfile = File('p.pvd')
ufile << uh
pfile << ph

# Run time loop
t = T0
tn = 0
while t < T1:
    # Solve advection-diffusion equations
    bh_u = assemble(b_u)
    if 1:
        solve(A_u, uh.vector(), bh_u, "lu")
    else:
        solve(A_u, uh.vector(), bh_u,
              "gmres",
              solver_parameters={
                'relative_tolerance': 1e-15,
                'monitor_convergence': True,
                'gmres': { 'restart': 100 },
                })

    # Compute error, need to have defined an analytical solution u0 for this to work
    if 0:
        ue = uh-u0
        e = sqrt(assemble(dot(ue,ue)*dx))
        #e1 = sqrt(assemble(ue[0]**2*dx))
        #e2 = sqrt(assemble(ue[1]**2*dx))
        print "t, m, unknowns, error, e1, e2: ", t, m, uh.vector().size(), e #, e1, e2
        times.append(t)
        errors.append(e)
        if e < 1e-10:
            print "Reached analytical solution, breaking out of time loop."
            break

    # Solve pressure equation
    php.assign(ph)
    bh_p = assemble(b_p)
    pbc.apply(A_p, bh_p)
    solve(A_p, ph.vector(), bh_p) # -> pn+1

    # Compute change in u from last timestep
    uchange = assemble(dot((uh-uhp),(uh-uhp))*dx)

    # Store results
    uchanges.append(uchange)
    print "uchange:", uchange
    ufile << uh
    pfile << ph

    # Prepare for next iteration
    uhp.assign(uh)
    t += dtvalue
    tn += 1

# Postprocessing
if 0:
    plot(uh, title='u')
    #plot(ue)
    plot(ph, title='p')
    interactive()

if 0:
    import pylab
    pylab.plot(times, errors)
    pylab.plot(times, uchanges)
    pylab.show()

