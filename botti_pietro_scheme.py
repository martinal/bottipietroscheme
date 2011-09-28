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

if 0:
    # Ugly but sure way to set quadrature rule
    ffc_opt = { 'representation': 'quadrature', 'quadrature_degree': 7 }
    def assemble(*args, **kwargs):
        kwargs['form_compiler_parameters'] = ffc_opt
        return dolfin.assemble(*args, **kwargs)

# Parameters
d = 2
m = int(sys.argv[1])
dtvalue = 1e-1
T0, T1 = 0.0, dtvalue*10.1 # Just a few steps for now

# Define mesh
if d == 2:
    mesh = UnitSquare(m, m)
elif d == 3:
    mesh = UnitCube(m, m, m)
mesh.order()

# TODO: Define subdomains

# Get UFL geometric quantities
cell = mesh.ufl_cell()
x = cell.x
n = FacetNormal(mesh)

# Define function spaces
k = 1
Vs = [VectorFunctionSpace(mesh, "DG", ks) for ks in [1,2,3]]
V = Vs[k-1]
V_cg = VectorFunctionSpace(mesh, "CG", 3)
P = FunctionSpace(mesh, "CG", 1)
P2 = FunctionSpace(mesh, "CG", 2)

# Define functions
uh = Function(V)
uhp = Function(V)
ub = Function(V)
ph = Function(P)
php = Function(P)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(P)
q = TestFunction(P)
nu = Constant(1.0, cell=cell) # TODO: Value?
eta = Constant(5.0, cell=cell)
f = Function(V_cg)
g = Function(P2)
dt = Constant(dtvalue, cell=cell)

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

# Left hand side for u equations
a_h  = inner(grad(u), grad(v))*dx
a_h += ( (eta('+')*k**2/h_F)*dot(jump(u),jump(v))
        - dot( avg(grad(u))*np, jump(v) )
        - dot( avg(grad(v))*np, jump(u) ) ) * dS
a_h += ((eta*k**2/h_T)*dot(u,v)
        - dot(grad(u)*n, v)
        - dot(grad(v)*n, u) ) * ds

# Right hand side for u equations
b_h = (eta*k**2/h_T)*dot(ub,v)*ds
b_h -= dot(grad(v)*n, ub)*ds
b_h -= dot(f,v)*dx

# Add time derivatives to forms
a_h += (1/dt)*dot(u, v)*dx
b_h += (1/dt)*dot(uhp,v)*dx

# Alternative formulations exposing bugs in FEniCS:
#a_h -= ((avg(u[j].dx(i)) * nm[i]) * jump(v[j]))*dS # BUG in FFC, undefined 'direction' variable
#a_h -= dot( avg(grad(u)) * nm, jump(v) )*dS # BUG in FFC, np works but not nm...
#a_h -= (dot(avg(grad(u[j])), n) * jump(v[j]))*dS # BUG in UFL, mixing tensor/index notation

# Pressure Poisson equation
a_h_p = dot(grad(p),grad(q))*dx
b_h_p = div(f - uh)*q*dx # TODO


# Assemble matrices
if 1:
    u0 = as_vector((x[0]**2, x[1]**2))
    ub.assign(project(u0, V))
    f0 = as_vector((2, 2))
    f.assign(project(f0, V_cg))

#f.assign(project(div(grad(u0)), V_cg)) # BUG in UFL, listtensor assumption failed

A_u = assemble(a_h)
A_p = assemble(a_h_p)

times = []
errors = []

# Run time loop
t = T0
while t < T1:
    # TODO: Solve advection-diffusion equations
    b_u = assemble(b_h)
    if 1:
        solve(A_u, uh.vector(), b_u, "lu")
    else:
        solve(A_u, uh.vector(), b_u,
              "gmres",
              solver_parameters={
                'relative_tolerance': 1e-15,
                'monitor_convergence': True,
                'gmres': { 'restart': 300 },
                })

    if 1:
        ue = uh-ub
        e = sqrt(assemble(dot(ue,ue)*dx))
        #e1 = sqrt(assemble(ue[0]**2*dx))
        #e2 = sqrt(assemble(ue[1]**2*dx))
        print "t, cells, unknowns, error, e1, e2: ", t, (2*m**2), uh.vector().size(), e #, e1, e2
        times.append(t)
        errors.append(e)

        if e < 1e-8:
            break

    # TODO: Solve pressure equation
    g.assign(project(x[0]**2, P2))
    b_p = assemble(b_h_p)
    solve(A_p, ph.vector(), b_p)

    uchange = assemble(dot((uh-uhp),(uh-uhp))*dx)
    print "diff:", uchange
    uhp.assign(uh)
    t += dtvalue


# TODO: Postprocessing
if 1:
    plot(uh)
    plot(ue)
    plot(ph)
    interactive()

if 0:
    import pylab
    pylab.plot(times, errors)
    pylab.show()
