import numpy as np
import matplotlib.pyplot as plt
from cylinder_fdm_3d import CylinderFDM
from fft_tdse.simulator import LaserPulse
from fft_tdse.subspace_imag_prop import ImagProp
from erfgau import ErfgauPotential
from icecream import ic
from scipy.signal import detrend
from rich.progress import Progress


ic.configureOutput(prefix="")

# ---------------------------------------------------------
# A solver for the Hydrogen atom using the erfgau potential.
# We solve the TDSE in cylinder coordinates for m=0, with
# a laser field polarized along the z-axis. We compute
# the induced dipole moment as function of time, and finally
# compute the HHG spectrum.
# ---------------------------------------------------------

#
# Set up grid, potentials, and solver
#

"""
Hei. Her er HHG data for Hydrogen. Jeg har kjørt med r_max=320 a.u., N=640 Lobatto punkter og l_max=1,...,30. Det elektriske feltet er gitt  som

      E(t) = sin(pi*t/T)^2 * sin(omega_L*t),

der T=n_c * (2*pi/omega_L). 
Jeg har brukt n_c=3 og omega_L = 0.057 a.u. (800 nm). Total kjøretid er T. Jeg gjør Crank-Nicholson løst med bicgstab med en konvergenstoleranse på 1e-10 og dt=0.2 a.u.
"""


n_r = 1000
n_z = 1000
r_max = 320.0
z_max = 320.0
solver = CylinderFDM(r_max, z_max, n_r, n_z, n_m=1)
tt, rr, zz = solver.get_trz_meshgrid()
erfgau = ErfgauPotential(mu=0.5)
omega0 = 0.057
E0 = 0.03
t_c = 2 * np.pi / omega0
n_cycles = 3
T = n_cycles * t_c
ic(T)
laser = LaserPulse(omega=omega0, E0=E0, T=T, t0=0.0, phi = 0.5)
solver.set_realspace_potential(erfgau.potential_radial((rr**2 + zz**2) ** 0.5))


#
# Compute ground state wavefunction.
#
ic("Computing ground state ...")
solver.setup_splitting_scheme(-1j*.02)
psi_guess = (rr**.5 * np.exp(-((rr**2 + zz**2) ** 0.5))).reshape((solver.n_dof, 1))
ip = ImagProp(solver.n_dof, lambda u: solver.propagate_crank_nicolson(u.reshape(solver.shape), 0.0).reshape(-1))
psi_init, error = ip.imag_prop(psi_guess, 1e-6, 1000)
#E_init, psi_init = solver.compute_ground_state_via_diagonalization()
psi_init = psi_init.reshape(solver.shape)
E_init = np.sum(psi_init.conj() * solver.apply_hamiltonian(psi_init))
ic(E_init)


#
# Add time dependent potential
#
solver.set_td_potential_modulator(laser)
solver.set_td_potential(zz)



def get_dipole_moment(psi):
    phi = rr**(-.5) * solver.m_to_theta(psi)
    rho = (phi.conj() * phi).real
    return np.sum(rho * solver.D) / np.sum(rho)


#
# Set up time propagation
#
t_final = T
dt = 0.05
t_range = np.arange(0, t_final + dt, dt)
n_steps = len(t_range) - 1
dipole_moments = np.zeros(n_steps + 1)

# #
# # Plot laser
# #
# plt.figure()
# plt.plot(t_range, laser(t_range))
# plt.xlabel("Time")
# plt.title("Laser pulse")
# plt.show()




psi = psi_init
solver.setup_splitting_scheme(dt)
dipole_moments[0] = get_dipole_moment(psi)



#
# Propagate
#
with Progress() as progress:
    task = progress.add_task("[cyan]Propagating...", total=n_steps)

    for i in range(n_steps):
        t = t_range[i]
        psi = solver.propagate_crank_nicolson(psi, t)
        dipole_moments[i + 1] = get_dipole_moment(psi)

        progress.update(task, advance=1, description=f"t = {t:.2f}", completed=i)


# plt.figure()
# plt.plot(t_range, dipole_moments)
# plt.xlabel("Time")
# plt.title("Induced dipole moment")


#
# Compute HHG spectrum
#
def compute_hhg_spectrum(time_points, dipole_moment, hann_window=False):

    dip = detrend(dipole_moment, type="constant")
    if hann_window:
        Px = (
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        dip * np.sin(np.pi * time_points / time_points[-1]) ** 2
                    )
                )
            )
            ** 2
        )
    else:
        Px = np.abs(np.fft.fftshift(np.fft.fft(dip))) ** 2

    dt = time_points[1] - time_points[0]

    omega = (
        np.fft.fftshift(np.fft.fftfreq(len(time_points)))
        * 2
        * np.pi
        / dt
    )

    return omega, Px

# window = np.vectorize(laser.envelope_sin2)(t_range)
# omega = np.fft.fftfreq(len(t_range), d=dt)
# dipa = np.abs(omega**2 * np.fft.fft(dipole_moments * window))

omega, dipa = compute_hhg_spectrum(t_range, dipole_moments, hann_window=True)

#
# Save t_range, dipole_moments, and dipa, and omega
#
np.savez(
    f"cylinder_fdm_r_max_{r_max}_N_{n_r}_dt_{dt}.npz",
    t_range=t_range,
    dipole_moments=dipole_moments,
    dipa=dipa,
    omega=omega/omega0,
)


#
# Plot
#
plt.figure()
plt.semilogy(omega / omega0, dipa)
plt.xlim(0, 30)
#plt.semilogy(omega[: n_steps // 2] / omega0, dipa[: n_steps // 2])
plt.xlabel("Frequency")
plt.title("HHG spectrum")
plt.grid(True)
plt.show()
