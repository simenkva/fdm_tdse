import numpy as np
import scipy as sp
import time
from poisson_solvers import PoissonSpherical

"""
Solve the Restricted Hartree-Fock equations for the Helium atom 
in spherical coordinates using the finite difference method. 
The direct potential is computed by solving the Poisson equation.
"""

r_max = 10
nr = 799
r = np.linspace(0, r_max, nr + 2)
dr = r[1] - r[0]
Z = 2

h_diag = 1 / dr**2 - Z / r[1:-1]
h_offdiag = -1 / (2 * dr**2)

H = (
    np.diag(h_diag)
    + np.diag(h_offdiag * np.ones(nr - 1), k=1)
    + np.diag(h_offdiag * np.ones(nr - 1), k=-1)
)

eps, C = np.linalg.eigh(H)
psi0 = C[:, 0] / r[1:-1]
psi0 /= np.sqrt(sp.integrate.simps(psi0**2 * r[1:-1] ** 2, r[1:-1]))


poisson = PoissonSpherical(r, l_max=3)

print()
for i in range(10):

    rho = np.abs(psi0) ** 2 / np.sqrt(4 * np.pi)
    b = -4 * np.pi * r[1:-1] * rho
    W_rmax = np.sqrt(4 * np.pi)  # Boundary condition at r_max when l=m=0.

    tic = time.time()
    W00 = poisson.solve(l=0, f=b, w_rmax=W_rmax)
    toc = time.time()
    time_poisson = toc - tic

    W_direct = W00 / r[1:-1]
    F = H + np.diag(
        W_direct / np.sqrt(4 * np.pi)
    )  # The factor 1/(4pi) is due to the integraton over the spherical harmonics

    tic = time.time()
    eps, C = np.linalg.eigh(F)
    toc = time.time()
    time_diag = toc - tic

    psi0 = C[:, 0] / r[1:-1]
    psi0 /= np.sqrt(sp.integrate.simps(psi0**2 * r[1:-1] ** 2, r[1:-1]))

    # Compute the Hartree-Fock energy
    EH = sp.integrate.simps(
        W_direct * psi0**2 * r[1:-1] ** 2, r[1:-1]
    ) / np.sqrt(4 * np.pi)
    e_hf = 2 * eps[0] - EH

    print(f"Time for Poisson solver: {time_poisson:.1e} s")
    print(f"Time for diagonalization: {time_diag:.3f} s")
    print(f"EHF: {e_hf:.4f}, eps0: {eps[0]:.4f}")
    print()


from matplotlib import pyplot as plt

plt.figure()
plt.subplot(211)
plt.plot(r[1:-1], np.abs(psi0)**2, label=r"$|\psi_0(r)|^2$")
plt.legend()
plt.subplot(212)
plt.plot(r[1:-1], W_direct, label=r"$W(r)$")
plt.legend()
plt.show()