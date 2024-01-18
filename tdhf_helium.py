import numpy as np
import scipy as sp
import time
from poisson_solvers import PoissonSpherical
from matplotlib import pyplot as plt

gs_state = np.load("helium_groundstate.npz")

Z = 2
r_max = 10
l_max = 5

###############################################################################
"""
The gaunt coefficients are the integral over three spherical harmonics
    \int Y_{l1,m1} Y_{l2,m2} Y_{l3,m3} dOmega
"""
from sympy.physics.wigner import gaunt
from sympy import N

tic = time.time()
gaunt_coeffs = np.zeros((l_max, l_max, l_max))
for l1 in range(l_max):
    for l2 in range(l_max):
        for l3 in range(l_max):
            gaunt_coeffs[l1, l2, l3] = N(gaunt(l1, l2, l3, 0, 0, 0))
toc = time.time()
print(f"Time for computing gaunt coefficients: {toc - tic:.3f} s")
###############################################################################

nr = 799
r = np.linspace(0, r_max, nr + 2)
dr = r[1] - r[0]

poisson = PoissonSpherical(r, l_max=l_max)

u = np.zeros((l_max, nr), dtype=np.complex128)
u[0] = np.complex128(gs_state["u0"])

tic = time.time()
ug = np.einsum("IJK,Ki->IJi", gaunt_coeffs, u)
rho = np.einsum("IJi,Ji->Ii", ug, np.conj(u))
toc = time.time()
print(f"Time for computing rho: {toc - tic:.3f} s")

W = np.zeros((l_max, nr), dtype=np.complex128)
tic = time.time()
for l in range(l_max):
    b = -4 * np.pi / r[1:-1] * rho[l]
    if l == 0:
        w_rmax = np.sqrt(4 * np.pi)  # Boundary condition at r_max when l=0.
    else:
        w_rmax = 0
    w_l = poisson.solve(l=l, f=b, w_rmax=w_rmax)
    W[l] = w_l / r[1:-1]
toc = time.time()
print(f"Time for computing W: {toc - tic:.3f} s")

EH = (
    sp.integrate.simps(W[0] * np.abs(u[0] ** 2), r[1:-1])
    * gaunt_coeffs[0, 0, 0]
)
print(f"EH: {EH:.4f}")


def Tu(u):

    Tu = np.zeros(u.shape, u.dtype)
    l_max = u.shape[0]

    for l in range(l_max):
        Tu[l, 0] = -0.5 * (-2 * u[l, 0] + u[l, 1]) / dr**2
        Tu[l, 1:-1] = -0.5 * (u[l, 0:-2] - 2 * u[l, 1:-1] + u[l, 2:]) / dr**2
        Tu[l, -1] = -0.5 * (u[l, -2] - 2 * u[l, -1]) / dr**2
        Tu[l] += l * (l + 1) / (2 * r[1:-1] ** 2) * u[l]

    return Tu


tic = time.time()
Hu = Tu(u)
Hu -= Z * np.einsum("K, lK->lK", 1 / r[1:-1], u)
Wu = np.einsum("IJi, Ji->Ii", ug, W)
Fu = Hu + Wu
toc = time.time()
print(f"Time for computing Fu: {toc - tic:.3f} s")

eps = sp.integrate.simps(np.conj(u[0]) * Fu[0], r[1:-1]).real
e_hf = 2 * eps - EH.real
print(f"orbital energy {eps:.4f}, Hartree-Fock energy: {e_hf:.4f}")
