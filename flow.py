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
Nx = [128,128]

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.Chebyshev(coords['x'], size=Nx[0], bounds=(0, 1),dealias=3/2)
ybasis = d3.Chebyshev(coords['y'], size=Nx[1], bounds=(0, 1))

# fields
# x = dist.Field(bases=(xbasis,ybasis))
x = dist.local_grid(xbasis)

# xb0 = de.Chebyshev('x0',Nx[0],interval=(-5,0))
# xb1 = de.Chebyshev('x1',Nx[1],interval=(0,1))
# xbasis = de.Compound('x',[xb0,xb1])
# domain = de.Domain([xbasis], grid_dtype=np.float64)
# x = xbasis.grid(*domain.dealias)

# Γ = domain.new_field(name='Γ',scales=domain.dealias)
Γ = dist.Field(name='Γ',bases=(xbasis,ybasis))
ϵ = dist.Field(name='ϵ')
ϵ['g'] = 0.1
Γ['g'] = 1.*(x<0.1)

u = dist.Field(name='u',bases=(xbasis,ybasis))
ux = d3.Differentiate(u, coords['x'])
dx = lambda A: d3.Differentiate(A, coords['x'])

#%%
# poiseuille = de.LBVP(domain, variables=['u','ux'])
poiseuille = d3.LBVP([u,ux],namespace=locals())

# poiseuille.meta[:]['x']['dirichlet'] = True
# poiseuille.parameters['Γ'] = Γ
# poiseuille.parameters['ε'] = ϵ
poiseuille.add_equation("dx(ux) = -2")
poiseuille.add_equation("u(x='left') = 0")
poiseuille.add_equation("u(x='right') = 0")

poiseuille_solver = poiseuille.build_solver()
poiseuille_solver.solve()

u, ux = poiseuille_solver.state['u'], poiseuille_solver.state['ux']

#%%

fig, ax = plt.subplots()
ax.plot(x,u['g'],color='C1',label='Penalized')
ax.plot(x[x>0],x[x>0]*(1-x[x>0]),'k--',label='Reference')
ax.fill_between(x[x<0],0,10,color='lightgray')
ax.set(aspect=1,xlim=[-1,1],ylim=[0,1],xlabel='$x$',ylabel='$u$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()