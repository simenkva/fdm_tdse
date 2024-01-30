from polar_fdm_2d import *
import numpy as np
import matplotlib.pyplot as plt


def solve_model(r_max, n_r, m_max):
    # Set up solver object    
    solver = polar_fdm_2d(r_max = r_max, n_r = n_r, m_max = 0)
    
    # # x0 = 0.1
    # # y0 = 0.2
    # # alpha = x0 - 1j*y0
    # V_m = []
    # V_m.append(-alpha*0.5*solver.r_inner)
    # V_m.append(0.5*solver.r_inner**2 + 0.5*np.abs(alpha)**2)
    # V_m.append(-alpha.conjugate()*0.5*solver.r_inner)
    V_m = []
    V_m.append(0.5*solver.r_inner**2)
    solver.set_potential(V_m)

    # Compute sparse CSR matrix version of Hamiltonian.
    H_mat_sparse = solver.get_sparse_matrix()

    # Compute lowest eigenvalues and eigenvectors.
    start = time()
    # Compute eigenvalues and eigenvectors.
    E, U = eigsh(H_mat_sparse, k=6, sigma = 1.0)
    print(time()-start)

    psi_numeric = solver.G_neumann @ solver.reduced_to_full(U[:,0].reshape(solver.shape))
    psi_numeric /= psi_numeric[0]
    psi_exact = np.exp(-0.5*np.abs(solver.r0)**2).reshape(psi_numeric.shape)
    print(psi_numeric.shape, psi_exact.shape)
    
    # Return lowest eigenvalue and eigenfunction, converted to non-reduced form
    return E[0], solver.r0, psi_numeric, psi_exact


# Do a sequence of simulations with increasing number of grid points.
r_max = 10
n_r_list = np.array([100, 200, 400, 800, 1600, 3200, 6400])
h_list = r_max / (n_r_list + 1)
E_list = []
psi_numeric_list = []
psi_exact_list = []
psi_err_list = []
r_list = []
for n_r in n_r_list:
    E, r0, psi_numeric, psi_exact = solve_model(r_max = r_max, n_r = n_r, m_max = 0)
    E_list.append(E)
    psi_numeric_list.append(psi_numeric)
    psi_exact_list.append(psi_exact)
    r_list.append(r0)
    
    psi_err_list.append( np.linalg.norm(psi_exact - psi_numeric) )
    


    
alpha, beta = np.polyfit(np.log(h_list), np.log(np.array(E_list) - 1.0), 1)
alpha2, beta2 = np.polyfit(np.log(h_list), np.log(np.array(psi_err_list)), 1)

plt.figure()
plt.loglog(h_list, np.array(E_list) - 1.0, '*-', label=r'$\alpha_E$ = {:.2f}'.format(alpha))
plt.loglog(h_list, np.array(psi_err_list), '*-', label=r'$\alpha_\psi$= {:.2f}'.format(alpha2))
plt.legend()
plt.title('Rate of convergence of $E_0$ and $\psi_0$')
plt.show()

plt.figure()
for i in range(len(n_r_list)):
    n_r = n_r_list[i]
    print(i, n_r)
    x = r_list[i]
    y = np.abs(psi_exact_list[i] - psi_numeric_list[i])
    print(x.shape, y.shape)
    plt.semilogy(x, y, label=f'n_r = {n_r}')
    
plt.title('Absolute error of eigenfunction')
plt.ylim(1e-32, 1e-0)
plt.legend()
plt.show()
