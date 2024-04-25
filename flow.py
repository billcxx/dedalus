#%% 
import pathlib
import subprocess
import h5py
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3


%matplotlib inline
%config InlineBackend.figure_format = 'retina'
#%%

# Calculate penalized solution
ϵ = 0.1
Nx = [128,128]

xb0 = de.Chebyshev('x0',Nx[0],interval=(-5,0))
xb1 = de.Chebyshev('x1',Nx[1],interval=(0,1))
xbasis = de.Compound('x',[xb0,xb1])
domain = de.Domain([xbasis], grid_dtype=np.float64)
x = xbasis.grid(*domain.dealias)

Γ = domain.new_field(name='Γ',scales=domain.dealias)
Γ['g'] = 1.*(x<ϵ)

poiseuille = de.LBVP(domain, variables=['u','ux'])
poiseuille.meta[:]['x']['dirichlet'] = True
poiseuille.parameters['Γ'] = Γ
poiseuille.parameters['ε'] = ϵ
poiseuille.add_equation("dx(ux) - (Γ/ε**2)*u = -2")
poiseuille.add_equation("ux - dx(u) = 0")
poiseuille.add_bc("left(ux) = 0")
poiseuille.add_bc("right(u) = 0")

poiseuille_solver = poiseuille.build_solver()
poiseuille_solver.solve()

u, ux = poiseuille_solver.state['u'], poiseuille_solver.state['ux']