from cylinder_fdm_3d import CylinderFDM
from icecream import ic
import numpy as np
from scipy.sparse.linalg import eigsh
from time import time
from matplotlib import pyplot as plt
from fft_tdse.simulator import LaserPulse
from erfgau import ErfgauPotential
from rich.console import Console
from rich.table import Table


#
# Convergence of eigenvalues of Harmonic oscillator
#


def build_harmonic_oscillator(n_r, n_z, n_m, nev=10):
    # Set up solver object    
    r_max = 20
    z_max = 20
    solver = CylinderFDM(r_max = r_max, z_max = z_max, n_r = n_r, n_z = n_z, n_m = n_m)

    tt, rr, zz = solver.get_trz_meshgrid()
    ic(tt.shape, rr.shape, zz.shape)
    xx = rr*np.cos(tt)
    yy = rr*np.sin(tt)

    V = 0.5*(xx**2 + yy**2 + zz**2)
    
    # Assign potential
    solver.set_realspace_potential(V, rotation_symmetric=False)
    
    # Compute sparse CSR matrix version of Hamiltonian.
    H_mat_sparse = solver.get_sparse_matrix_fast()

    # Compute lowest eigenvalues and eigenvectors.
    start = time()
    # Compute eigenvalues and eigenvectors.
    ic('Computing eigenvalues')
    E, U = eigsh(H_mat_sparse, k=nev, sigma = 1.5)
    idx = np.argsort(E)
    E = E[idx]
    U = U[:,idx]
    print(time()-start)
    ic(E)
    
    E_exact = np.round(E-1.5) + 1.5
    ic(E_exact)
    
    E_error = E - E_exact
       
    # Get the numerical eigenfunction
    psi_numeric = rr**(-.5) * np.fft.ifft(U[:,0].reshape(solver.shape), axis=0, norm='ortho')
    psi_numeric /= np.linalg.norm(psi_numeric)
    psi_exact =  np.exp(-0.5*(rr**2 + zz**2))
    psi_exact /= np.linalg.norm(psi_exact)

    error = psi_numeric - psi_exact*np.sum(psi_numeric*psi_exact.conj())
    delta_psi = np.linalg.norm(error)

    return E, E_exact, delta_psi


def eigenvalue_study():
    nev = 10
    n_range = np.array([100, 200, 300, 400])
    E_error = np.zeros((len(n_range), nev))
    psi_error = np.zeros((len(n_range)))

    for k, n in enumerate(n_range):
        ic(k, n)
        E, E_exact, delta_psi = build_harmonic_oscillator(n, n, 1, nev=nev)
        E_error[k, :] = E - E_exact
        psi_error[k] = delta_psi
        

    alpha = []
    beta = []
    for i in range(nev):
        alpha0, beta0 = np.polyfit(np.log(1/n_range), np.log(np.abs(E_error[:,i])), 1)
        alpha.append(alpha0)
        beta.append(beta0)  
        

    plt.figure()
    plt.loglog(1/n_range, np.abs(E_error), 'o-')
    plt.legend([f'E={E_exact[i]}, alpha = {alpha[i]:.2f}' for i in range(nev)])
    plt.xlabel('1/n')
    plt.title('Error in eigenvalues of HO')
    plt.show()




def propagation_timing(n_r, n_z, n_m, n_steps = 20):
    solver = CylinderFDM(r_max = 20, z_max = 20, n_r = n_r, n_z = n_z, n_m = n_m)
    tt, rr, zz = solver.get_trz_meshgrid()
    V = 0.5*(rr**2 + zz**2)
    solver.set_realspace_potential(V, rotation_symmetric=False)
    laser = LaserPulse(E0=0.1, omega=0.057, t0=0, T=100)
    solver.set_td_potential_modulator(laser)
    solver.set_td_potential(zz)
    
    dt = 0.01
    solver.setup_splitting_scheme(dt)
    psi = np.exp(-0.5*(rr**2 + (zz-1)**2)) * rr**.5
    
    t = np.linspace(0, n_steps*dt, n_steps+1)
    
    time_taken = []
    

    for k in range(n_steps):
        print('Time step', k)
        start = time()
        psi = solver.propagate_crank_nicolson(psi, t[k])
        time_taken.append(time() - start)

    ic(time_taken, np.mean(time_taken), np.std(time_taken))
    
    
    return np.mean(time_taken), np.std(time_taken)


def propagation_timing_study():

    
    n_m_list = np.array([1, 8, 16, 32, 64])
    n_list = np.array([256, 512, 1024])
    
    mean_time = np.zeros((len(n_list), len(n_m_list)))
    std_time = np.zeros((len(n_list), len(n_m_list)))
    
    for k, n_m in enumerate(n_m_list):
        for j, n in enumerate(n_list):
            mean_time[j,k], std_time[j,k] = propagation_timing(n, n, n_m, n_steps=4)

    plt.figure(figsize=(10, 8))
    for k, n_m in enumerate(n_m_list):
        alpha, beta = np.polyfit(np.log(n_list), np.log(mean_time[:,k]), 1)
        line = lambda x: np.exp(beta)*x**alpha
        plt.errorbar(n_list, mean_time[:,k], yerr=std_time[:,k], fmt='o-', color=f'C{k}', label=f'n_m = {n_m}')
        plt.loglog(n_list, line(n_list), '--', color=f'C{k}', label=f'alpha = {alpha:.2f}')

    plt.legend()
    plt.title('Time per time step')
    plt.xlabel('n=n_r=n_z')
    plt.ylabel('Time (s)')
    plt.savefig('propagation_timing.png')

    plt.figure(figsize=(10, 8))
    for j, n in enumerate(n_list):
        alpha, beta = np.polyfit(np.log(n_m_list), np.log(mean_time[j,:]), 1)
        line = lambda x: np.exp(beta)*x**alpha
        plt.errorbar(n_m_list, mean_time[j,:], yerr=std_time[j,:], fmt='o-', color=f'C{j}', label=f'n = {n}')
        plt.loglog(n_m_list, line(n_m_list), '--', color=f'C{j}', label=f'alpha = {alpha:.2f}')

    plt.legend()
    plt.title('Time per time step')
    plt.xlabel('n_m')
    plt.ylabel('Time (s)')
    plt.savefig('propagation_timing2.png')


    plt.show()
    

    console = Console(record=True)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("n", style="dim")
    for n_m in n_m_list:
        table.add_column(f"n_m = {n_m}")

    for j, n in enumerate(n_list):
        row = [str(n)]
        for k, n_m in enumerate(n_m_list):
            row.append(f"{mean_time[j, k]:.2f} ± {std_time[j, k]:.2f}")
        table.add_row(*row)

    console.print(table)
    console.print(f'Times per time step (s) for different n and n_m')
    
    
    with open('propagation_timing.html', 'w') as f:
        f.write(console.export_html())

if __name__=="__main__":
    propagation_timing_study()
        
    



