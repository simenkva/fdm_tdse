import numpy as np
import scipy as sp
import time
from poisson_solvers import PoissonSpherical
from matplotlib import pyplot as plt
import tqdm

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

dt = 0.1
t_diag = np.ones(nr) / dr**2
t_subdiag = -0.5 * np.ones(nr) / dr**2
t_upperdiag = -0.5 * np.ones(nr) / dr**2
t_upperdiag[0] = 0
t_subdiag[-1] = 0
v_diag = -Z / r[1:-1]

I_p_1jH0_diag_l = np.zeros((l_max, nr), dtype=np.complex128)
for l in range(l_max):
    I_p_1jH0_diag_l[l] = np.ones(nr) + 1j * dt / 2 * (
        t_diag + v_diag + l * (l + 1) / (2 * r[1:-1] ** 2)
    )
I_p_1jH0_subdiag = 1j * dt / 2 * t_subdiag
I_p_1jH0_upperdiag = 1j * dt / 2 * t_upperdiag

###############################################################################
poisson = PoissonSpherical(r, l_max=l_max)


def compute_W(rho):
    W = np.zeros((l_max, nr), dtype=np.complex128)
    for l in range(l_max):
        b = -4 * np.pi / r[1:-1] * rho[l]
        if l == 0:
            w_rmax = np.sqrt(4 * np.pi)  # Boundary condition at r_max when l=0.
        else:
            w_rmax = 0
        w_l = poisson.solve(l=l, f=b, w_rmax=w_rmax)
        W[l] = w_l / r[1:-1]
    return W


###############################################################################
def Tu_mult(u):

    Tu = np.zeros(u.shape, u.dtype)
    l_max = u.shape[0]

    for l in range(l_max):
        Tu[l, 0] = -0.5 * (-2 * u[l, 0] + u[l, 1]) / dr**2
        Tu[l, 1:-1] = -0.5 * (u[l, 0:-2] - 2 * u[l, 1:-1] + u[l, 2:]) / dr**2
        Tu[l, -1] = -0.5 * (u[l, -2] - 2 * u[l, -1]) / dr**2
        Tu[l] += l * (l + 1) / (2 * r[1:-1] ** 2) * u[l]

    return Tu


###############################################################################

ut = np.zeros((l_max, nr), dtype=np.complex128)
ut[0] = np.complex128(gs_state["u0"])


num_steps = 1000
t_final = num_steps * dt
print(f"Final time: {t_final:.3f}")

eps = np.zeros(num_steps, dtype=np.complex128)
EH = np.zeros(num_steps, dtype=np.complex128)
norm = np.zeros(num_steps, dtype=np.complex128)
time_points = np.zeros(num_steps)

print()
for n in tqdm.tqdm(range(num_steps - 1)):
    time_points[n] = n * dt
    # print(ut[0, 0])
    # tic = time.time()
    ut_g = np.einsum("IJK,Ki->IJi", gaunt_coeffs, ut)
    rho = np.einsum("IJi,Ji->Ii", ut_g, np.conj(ut))
    # toc = time.time()
    # print(f"Time for computing rho: {toc - tic:.3f} s")

    # tic = time.time()
    W = compute_W(rho)
    # toc = time.time()
    # print(f"Time for computing W: {toc - tic:.3f} s")

    norm[n] = sp.integrate.simps(np.abs(ut[0] ** 2), r[1:-1])
    EH[n] = (
        sp.integrate.simps(W[0] * np.abs(ut[0] ** 2), r[1:-1])
        * gaunt_coeffs[0, 0, 0]
    )

    # tic = time.time()
    Tu = Tu_mult(ut)
    Vu = -Z * np.einsum("K, lK->lK", 1 / r[1:-1], ut)
    Wu = np.einsum("IJi, Ji->Ii", ut_g, W)
    Fu = Tu + Vu + Wu
    # toc = time.time()
    # print(f"Time for computing Fu: {toc - tic:.3f} s")

    eps[n] = sp.integrate.simps(np.conj(ut[0]) * Fu[0], r[1:-1]).real

    ut_tilde = ut - 1j * dt / 2 * Fu

    tic = time.time()
    fj = np.zeros((l_max, nr), dtype=np.complex128)
    for l in range(l_max):
        fj[l] = sp.linalg.solve_banded(
            (1, 1),
            np.array(
                [I_p_1jH0_upperdiag, I_p_1jH0_diag_l[l], I_p_1jH0_subdiag]
            ),
            ut_tilde[l],
        )
    ut = fj.copy()

    max_iters = 100
    iters = 1
    while np.linalg.norm(fj.ravel()) > 1e-10 and iters < max_iters:
        fj_g = np.einsum("IJK,Ki->IJi", gaunt_coeffs, fj)
        Vext_fj = -1j * dt / 2 * np.einsum("IJi,Ji->Ii", fj_g, W)
        for l in range(l_max):
            fj[l] = sp.linalg.solve_banded(
                (1, 1),
                np.array(
                    [I_p_1jH0_upperdiag, I_p_1jH0_diag_l[l], I_p_1jH0_subdiag]
                ),
                Vext_fj[l],
            )
        ut += fj
        iters += 1
    toc = time.time()
    # print(f"Number of iterations: {iters}")
    # print(f"Time for computing ut_new: {toc - tic:.3f} s")
    # print()


e_hf = 2 * eps - EH.real

plt.figure()
plt.subplot(311)
plt.plot(time_points[:-1], e_hf[:-1].real - e_hf[0].real, label=r"$E_{HF}$")
# plt.plot(time_points, eps.real, label=r"$\epsilon$")
plt.subplot(312)
plt.plot(time_points[:-1], EH[:-1].real - EH[0].real, label=r"$E_H$")
plt.subplot(313)
plt.plot(time_points[:-1], eps[:-1].real - eps[0].real, label=r"$\epsilon$")

plt.figure()
plt.plot(time_points[:-1], 1 - norm[:-1].real, label=r"$\int |\psi|^2$")

plt.show()
