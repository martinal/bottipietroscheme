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
m = int(sys.argv[1])
k = 1
dtvalue = 1e-1
T0, T1 = 0.0, dtvalue*50.1 # Just a few steps for now

yscale = 10

# Define mesh
if d == 2:
    mesh = UnitSquare(m, int(m*yscale/5))
    mesh.coordinates()[:,1] *= yscale
elif d == 3:
    mesh = UnitCube(m, m, m)
eps = 0.1/m

# Define subdomains
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

if 0:
    ba = boundaries.array()
    from collections import defaultdict
    dd = defaultdict(int)
    for i in range(ba.shape[0]):
        print i, ba[i]
        dd[ba[i]] += 1
    print dd
    sys.exit(0)

# Get UFL geometric quantities
cell = mesh.ufl_cell()
x = cell.x
n = FacetNormal(mesh)

# Define function spaces
Vs = [VectorFunctionSpace(mesh, "DG", ks) for ks in [1,2,3]]
V = Vs[k-1]
V_cg = VectorFunctionSpace(mesh, "CG", 3)
P = FunctionSpace(mesh, "CG", 1)
P2 = FunctionSpace(mesh, "CG", 2)

# Define boundary conditions
#class UB(Expression):
#    def eval(self, values, x):
#        values[0] = 0.0
#        values[1] = 1.0 if x[1] < DOLFIN_EPS else 0.0

#ub = Expression(("0.0", "x[1] < DOLFIN_EPS ? 1.0: 0.0"),
#                element=VectorElement("CG", cell, 1))

# Define functions
uh = Function(V)
uhp = Function(V)
ph = Function(P)
php = Function(P)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(P)
q = TestFunction(P)
nu = Constant(1.0, cell=cell) # TODO: Value?
eta = Constant(5.0, cell=cell)
f = Function(V_cg)
#g = Function(P2)
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
ub = as_vector((0, x[0]*(1.0-x[0])))
ub2 = -ub
ub0 = as_vector((0,0))
dsb = ds[boundaries]
b_h = (eta*k**2/h_T)*dot(ub,v)*dsb(1) + (eta*k**2/h_T)*dot(ub0,v)*dsb(0)
b_h -= dot(grad(v)*n, ub)*dsb(1) \
     + dot(grad(v)*n, ub0)*dsb(0) \
     + dot(grad(v)*n, ub2)*dsb(2)
b_h -= dot(f,v)*dx
b_h -= dot(v, 2*grad(ph) - grad(php))*dx # pressure coupling

# Add time derivatives to forms
a_h += (1/dt)*dot(u, v)*dx
b_h += (1/dt)*dot(uhp,v)*dx

# Alternative formulations exposing bugs in FEniCS:
#a_h -= ((avg(u[j].dx(i)) * nm[i]) * jump(v[j]))*dS # BUG in FFC, undefined 'direction' variable
#a_h -= dot( avg(grad(u)) * nm, jump(v) )*dS # BUG in FFC, np works but not nm...
#a_h -= (dot(avg(grad(u[j])), n) * jump(v[j]))*dS # BUG in UFL, mixing tensor/index notation

# Pressure Poisson equation
a_h_p = dot(grad(p),grad(q))*dx
b_h_p  = -(1/dt)*div(uh)*q*dx
b_h_p -= (1/dt('+'))*dot(np, jump(uh))*avg(q)*dS
b_h_p += (1/dt)*dot(n, uh-ub)*q*dsb(1)  \
      +  (1/dt)*dot(n, uh-ub0)*q*dsb(0) \
      +  (1/dt)*dot(n, uh-ub2)*q*dsb(2)
b_h_p += dot(grad(php), grad(q))*dx

pbc = DirichletBC(P, 0, Outlet())

# Assemble matrices
if 0:
    u0 = as_vector((x[0]**2, x[1]**2))
    #ub.assign(project(UB(), V))

    f0 = as_vector((2, 2))
    f.assign(project(f0, V_cg))

#f.assign(project(div(grad(u0)), V_cg)) # BUG in UFL, listtensor assumption failed

A_u = assemble(a_h)
A_p = assemble(a_h_p)

times = []
errors = []

ufile = File('u.pvd')
pfile = File('p.pvd')
ufile << uh
pfile << ph

# Run time loop
t = T0
tn = 0
while t < T1:
    # Solve advection-diffusion equations
    b_u = assemble(b_h) # pn, pn-1
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

    if 0:
        ue = uh-u0
        e = sqrt(assemble(dot(ue,ue)*dx))
        #e1 = sqrt(assemble(ue[0]**2*dx))
        #e2 = sqrt(assemble(ue[1]**2*dx))
        print "t, cells, unknowns, error, e1, e2: ", t, (2*m**2), uh.vector().size(), e #, e1, e2
        times.append(t)
        errors.append(e)

        if e < 1e-8:
            break

    # Solve pressure equation
    php.assign(ph)
    b_p = assemble(b_h_p)
    #pbc.apply(A_p, b_p)
    solve(A_p, ph.vector(), b_p) # -> pn+1

    uchange = assemble(dot((uh-uhp),(uh-uhp))*dx)
    print "diff:", uchange
    uhp.assign(uh)
    t += dtvalue
    tn += 1

    ufile << uh
    pfile << ph


# TODO: Postprocessing
if 1:
    plot(uh, title='u')
    #plot(ue)
    plot(ph, title='p')
    interactive()

if 0:
    import pylab
    pylab.plot(times, errors)
    pylab.show()
