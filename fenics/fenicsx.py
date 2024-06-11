import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.io import XDMFFile
import numpy as np
from basix.ufl import element

# Define the domain
Lx, Ly = 2.0, 1.0
nx, ny = 32, 16
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [nx, ny])
gdim = domain.topology.dim

# Function spaces
P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim, ))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
V = fem.functionspace(domain,P2)
Q = fem.functionspace(domain,P1)

#

fdim = gdim - 1
# Boundary conditions
def inlet_boundary(x):
    return np.isclose(x[0],0.0)

def outlet_boundary(x):
    return np.isclose(x[0], Lx)

def wall_boundary(x):
    return np.isclose(x[1],0.0) or np.isclose(x[1],Ly)

inlet_velocity = fem.Constant(domain, [1.0, 0.0])
bc_inlet = fem.dirichletbc(inlet_velocity, fem.locate_dofs_geometrical(V, inlet_boundary))
bc_wall = fem.dirichletbc(fem.Constant(domain, [0.0, 0.0]), fem.locate_dofs_geometrical(V, wall_boundary))
bc_outlet = fem.dirichletbc(fem.Constant(domain, 0.0), fem.locate_dofs_geometrical(Q, outlet_boundary))

# Equations
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)

nu = 0.1  # Viscosity

a1 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
L1 = -ufl.inner(ufl.dot(u, ufl.grad(u)), v) * ufl.dx

a2 = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
L2 = -ufl.inner(ufl.div(u), q) * ufl.dx

# Assemble
A1 = fem.form(a1)
b1 = fem.form(L1)
A2 = fem.form(a2)
b2 = fem.form(L2)

# Solve
solver1 = fem.petsc.LinearProblem(A1, b1, bcs=[bc_inlet, bc_wall])
u_sol = solver1.solve()

solver2 = fem.petsc.LinearProblem(A2, b2, bcs=[bc_outlet])
p_sol = solver2.solve()

# Output
with XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_sol)

with XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_sol)