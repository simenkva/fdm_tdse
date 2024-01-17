import numpy as np 
import scipy as sp
from matplotlib import pyplot as plt
from poisson_solvers import PoissonSpherical 

r_max = 30
nr = 5000
r = np.linspace(0, r_max, nr+2)

poisson = PoissonSpherical(r, l_max=3)

psi = np.exp(-r**2/2) #Groundstate of the harmonic oscillator with omega = 1
psi /= np.sqrt(sp.integrate.simps(psi**2*r**2, r)) #Normalize

rho = psi[1:-1]*psi[1:-1]/np.sqrt(4*np.pi) #The factor 4pi is due to the integraton over the spherical harmonics
b = -4*np.pi*r[1:-1]*rho
w_rmax = np.sqrt(4*np.pi)*sp.integrate.simps(psi**2*r**2, r) #Boundary condition at r_max when l=m=0.


w00_r = poisson.solve(l=0, f=b, w_rmax=w_rmax)


w00_exact_H = np.sqrt(4*np.pi)*(1-(r+1)*np.exp(-2*r))
w00_exact_HO = np.sqrt(4*np.pi) * sp.special.erf(r)

w00_exact = w00_exact_HO

plt.figure()
plt.subplot(211)
plt.plot(r[1:-1], w00_r, color='red', linestyle='dashed', label=r"$w^{FDM}(r)$")
plt.plot(r[1:-1], w00_exact[1:-1], color='black', label=r"$w^{exact}(r)$")
plt.legend()
plt.subplot(212)
plt.plot(r[1:-1], np.abs(w00_r-w00_exact[1:-1]), color='black', label=r"$|w^{FDM}(r)-w^{exact}(r)|$")
plt.legend()
plt.show()