from ufl import *
n = triangle.n
V = VectorElement('DG', triangle, 1)
v = TestFunction(V)
u = TrialFunction(V)
av = v[j]('+') + v[j]('-')
au = u.dx(i)[j]('-') + u.dx(i)[j]('+')
integrand = (n('-')[i] * au) * av
a = integrand * dS

from ufl.algorithms import tree_format
print
print str(a)
print
print tree_format(a)
print

from ffc import jit
aj = jit(a)

