from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista

from dolfinx.fem import Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.io import VTXWriter
from dolfinx.mesh import create_rectangle
from dolfinx.plot import vtk_mesh
from basix.ufl import element
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)

# Define the domain
Lx, Ly = 2.0, 1.0
nx, ny = 32, 16
domain = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [nx, ny])
gdim = domain.topology.dim

t = 0
T = 10
num_steps = 500
dt = T / num_steps

# Function spaces
P2 = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim, ))
P1 = element("Lagrange", domain.topology.cell_name(), 1)
V = functionspace(domain,P2)
Q = functionspace(domain,P1)

#

fdim = gdim - 1
# Boundary conditions
def inlet_boundary(x):
    return np.isclose(x[0],0.0)

def outlet_boundary(x):
    return np.isclose(x[0], Lx)

def wall_boundary(x):
    return np.logical_or(np.isclose(x[1],0.0) , np.isclose(x[1],Ly))

inlet_velocity = Constant(domain, [1.0, 0.0])
bc_inlet = dirichletbc(inlet_velocity, locate_dofs_geometrical(V, inlet_boundary),V)
bc_wall = dirichletbc(Constant(domain, [0.0, 0.0]), locate_dofs_geometrical(V, wall_boundary),V)
bc_outlet = dirichletbc(Constant(domain, 0.0), locate_dofs_geometrical(Q, outlet_boundary),Q)


# Equations
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)


u_n = Function(V)
u_n.name = "u_n"
U = 0.5 * (u_n + u)
n = FacetNormal(domain)
f = Constant(domain, PETSc.ScalarType((0, 0)))
k = Constant(domain, PETSc.ScalarType(dt))
mu = Constant(domain, PETSc.ScalarType(1))
rho = Constant(domain, PETSc.ScalarType(1))

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor


def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))


# Define the variational problem for the first step
p_n = Function(Q)
p_n.name = "p_n"
F1 = rho * dot((u - u_n) / k, v) * dx
F1 += rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
F1 += inner(sigma(U, p_n), epsilon(v)) * dx
F1 += dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds
F1 -= dot(f, v) * dx
a1 = form(lhs(F1))
L1 = form(rhs(F1))

A1 = assemble_matrix(a1, bcs=[bc_wall])
A1.assemble()
b1 = create_vector(L1)

# Define variational problem for step 2
u_ = Function(V)
a2 = form(dot(nabla_grad(p), nabla_grad(q)) * dx)
L2 = form(dot(nabla_grad(p_n), nabla_grad(q)) * dx - (rho / k) * div(u_) * q * dx)
A2 = assemble_matrix(a2, bcs=[bc_inlet,bc_outlet])
A2.assemble()
b2 = create_vector(L2)

# Define variational problem for step 3
p_ = Function(Q)
a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_, v) * dx - k * dot(nabla_grad(p_ - p_n), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)

# Solver for step 1
solver1 = PETSc.KSP().create(domain.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.HYPRE)
pc1.setHYPREType("boomeramg")

# Solver for step 2
solver2 = PETSc.KSP().create(domain.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.BCGS)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(domain.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)

from pathlib import Path
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)
vtx_u = VTXWriter(domain.comm, folder / "poiseuille_u.bp", u_n, engine="BP4")
vtx_p = VTXWriter(domain.comm, folder / "poiseuille_p.bp", p_n, engine="BP4")
vtx_u.write(t)
vtx_p.write(t)

def u_exact(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (1.0 - x[1])
    return values


u_ex = Function(V)
u_ex.interpolate(u_exact)

L2_error = form(dot(u_ - u_ex, u_ - u_ex) * dx)

for i in range(num_steps):
    # Update current time step
    t += dt

    # Step 1: Tentative veolcity step
    with b1.localForm() as loc_1:
        loc_1.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [[bc_wall]])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, [bc_wall])
    solver1.solve(b1, u_.vector)
    u_.x.scatter_forward()

    # Step 2: Pressure corrrection step
    with b2.localForm() as loc_2:
        loc_2.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [[bc_inlet,bc_outlet]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2,  [bc_inlet,bc_outlet])
    solver2.solve(b2, p_.vector)
    p_.x.scatter_forward()

    # Step 3: Velocity correction step
    with b3.localForm() as loc_3:
        loc_3.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.vector)
    u_.x.scatter_forward()
    # Update variable with solution form this time step
    u_n.x.array[:] = u_.x.array[:]
    p_n.x.array[:] = p_.x.array[:]

    # Write solutions to file
    vtx_u.write(t)
    vtx_p.write(t)

    # Compute error at current time-step
    error_L2 = np.sqrt(domain.comm.allreduce(assemble_scalar(L2_error), op=MPI.SUM))
    error_max = domain.comm.allreduce(np.max(u_.vector.array - u_ex.vector.array), op=MPI.MAX)
    # Print error only every 20th step and at the last step
    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")
# Close xmdf file
vtx_u.close()
vtx_p.close()
b1.destroy()
b2.destroy()
b3.destroy()
solver1.destroy()
solver2.destroy()
solver3.destroy()

# nu = 0.1  # Viscosity

# a1 = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(u, v) * ufl.dx
# L1 = -ufl.inner(ufl.dot(u, ufl.grad(u)), v) * ufl.dx

# a2 = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
# L2 = -ufl.inner(ufl.div(u), q) * ufl.dx

# # Assemble
# # A1 = fem.form(ufl.lhs(a1))
# # b1 = fem.form(ufl.rhs(L1))
# # A2 = fem.form(ufl.lhs(a2))
# # b2 = fem.form(ufl.rhs(L2))

# # Solve
# solver1 = dolfinx.fem.petsc.LinearProblem(a1, L1, bcs=[bc_inlet, bc_wall])
# u_sol = solver1.solve()

# solver2 = dolfinx.fem.petsc.LinearProblem(a2, L2, bcs=[bc_outlet])
# p_sol = solver2.solve()

# Output
# with XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(u_sol)

# with XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(p_sol)