import numpy as np
import scipy as sp
import tqdm
from poisson_solvers import PoissonSpherical

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
u0 = C[:, 0]
u0 /= np.sqrt(sp.integrate.simps(u0**2, r[1:-1]))

poisson = PoissonSpherical(r, l_max=3)
dt = 0.1
num_steps = 100

orbital_energy = np.zeros(num_steps)
hartree_energy = np.zeros(num_steps)

rho = u0**2 / r[1:-1] ** 2 * (1 / np.sqrt(4 * np.pi))
b = -4 * np.pi * r[1:-1] * rho
W_rmax = np.sqrt(4 * np.pi)  # Boundary condition at r_max when l=m=0.

W00 = poisson.solve(l=0, f=b, w_rmax=W_rmax)
W_direct = W00 / r[1:-1]

orbital_energy[0] = sp.integrate.simps(u0 * np.dot(H, u0), r[1:-1])
hartree_energy[0] = sp.integrate.simps(W_direct * u0**2, r[1:-1]) / np.sqrt(
    4 * np.pi
)


I = np.eye(nr)

for n in tqdm.tqdm(range(num_steps - 1)):

    F = H + np.diag(
        W_direct / np.sqrt(4 * np.pi)
    )  # The factor 1/(4pi) is due to the integraton over the spherical harmonics

    A_p = I + 0.5 * dt * F
    A_m = I - 0.5 * dt * F

    z = np.dot(A_m, u0)
    u0 = np.linalg.solve(A_p, z)
    u0 /= np.sqrt(sp.integrate.simps(u0**2, r[1:-1]))

    rho = u0**2 / r[1:-1] ** 2 * (1 / np.sqrt(4 * np.pi))
    b = -4 * np.pi * r[1:-1] * rho
    W_rmax = np.sqrt(4 * np.pi)  # Boundary condition at r_max when l=m=0.

    W00 = poisson.solve(l=0, f=b, w_rmax=W_rmax)
    W_direct = W00 / r[1:-1]

    orbital_energy[n + 1] = sp.integrate.simps(u0 * np.dot(F, u0), r[1:-1])
    hartree_energy[n + 1] = sp.integrate.simps(
        W_direct * u0**2, r[1:-1]
    ) / np.sqrt(4 * np.pi)

print(
    f"Orbital energy: {orbital_energy[-1]:.4f}, Hartree energy: {hartree_energy[-1]:.4f}, Hartree-Fock energy: {(2*orbital_energy[-1] - hartree_energy[-1]):.4f}"
)


from matplotlib import pyplot as plt

plt.figure()
plt.subplot(311)
plt.plot(orbital_energy + hartree_energy, label=r"$E_{RHF}$")
plt.legend()
plt.subplot(312)
plt.plot(orbital_energy, label=r"$\epsilon_0$")
plt.legend()
plt.subplot(313)
plt.plot(hartree_energy, label=r"$E_H$")
plt.legend()
plt.show()
