import numpy as np
import matplotlib.pyplot as plt
from cylinder_fdm_3d import CylinderFDM
from fft_tdse.simulator import LaserPulse
from erfgau import ErfgauPotential
from icecream import ic
from scipy.signal import detrend
import rich 
from rich.progress import Progress
from rich.live import Live
from rich.table import Table

ic.configureOutput(prefix='')

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

n_r = 400
n_z = 400
r_max = 320.0
z_max = 16.0
solver = CylinderFDM(r_max, z_max, n_r, n_z, n_m=1)
tt, rr, zz = solver.get_trz_meshgrid()
erfgau = ErfgauPotential(mu=1.0)
omega0 = 0.056
E0 = 0.06
t_c = 2*np.pi/omega0
n_cycles = 6
T = n_cycles*t_c
ic(T)
laser = LaserPulse(omega=omega0, E0=E0, T=T, t0=0.0)
solver.set_td_potential_modulator(laser)
solver.set_td_potential(zz)
solver.set_realspace_potential(erfgau.potential_radial((rr**2+zz**2)**.5))



#
# Compute ground state wavefunction.
#
ic('Computing ground state ...')
E_init, psi_init = solver.compute_ground_state_via_diagonalization()
ic(E_init)


def get_dipole_moment(psi):
    phi = solver.m_to_theta(psi)
    rho = (phi.conj() * phi).real
    return np.sum(rho* solver.D) / np.sum(rho)

#
# Set up time propagation
# 
t_final = T
dt = 0.1
t_range = np.arange(0, t_final+dt, dt)
n_steps = len(t_range) - 1
dipole_moments = np.zeros(n_steps+1)

psi = psi_init
solver.setup_splitting_scheme(dt)
dipole_moments[0] = get_dipole_moment(psi)

#
# Plot laser
#
plt.figure()
plt.plot(t_range, laser(t_range))
plt.xlabel('Time')
plt.title('Laser pulse')



#
# Propagate
#
with Progress() as progress:
    task = progress.add_task("[cyan]Propagating...", total=n_steps)
    
    for i in range(n_steps):
        t = t_range[i]
        psi = solver.propagate_crank_nicolson(psi, t)
        dipole_moments[i+1] = get_dipole_moment(psi)
        
        progress.update(task, advance=1, description=f"t = {t:.2f}", completed=i)
        
    
plt.figure()
plt.plot(t_range, dipole_moments)
plt.xlabel('Time')
plt.title('Induced dipole moment')


#
# Compute dipole acceleration
# 

    
window = np.vectorize(laser.envelope_sin2)(t_range)
omega = np.fft.fftfreq(len(t_range), d=dt)
dipa = np.abs(omega**2 * np.fft.fft(dipole_moments * window))
plt.figure()
plt.semilogy(omega[:n_steps//2]/omega0, dipa[:n_steps//2])
plt.xlabel('Frequency')
plt.title('HHG spectrum')
plt.grid(True)
plt.show()
