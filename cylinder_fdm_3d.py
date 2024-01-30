from icecream import ic
import numpy as np
from polar_fdm_2d import radial_fdm_laplacian
from scipy.sparse import spdiags, csr_matrix, lil_matrix
from scipy.sparse.linalg import LinearOperator, eigsh
import matplotlib.pyplot as plt

def laplace_stencil_1d(order=2):
    # Construct the 1D Laplacian operator stencil
    
    if order == 2:
        data = np.zeros((3,1))
        data[0, :] = 1
        data[1, :] = -2
        data[2, :] = 1
        offsets = [-1, 0, 1]
    elif order == 4:
        data = np.zeros((5,1))
        data[0, :] = 1 / 12
        data[1, :] = -16 / 12
        data[2, :] = 30 / 12
        data[3, :] = -16 / 12
        data[4, :] = 1 / 12
        offsets = [-2, -1, 0, 1, 2]
    elif order == 6:
        data = np.zeros((7,1))
        data[0, :] = 1 / 90
        data[1, :] = -3 / 20
        data[2, :] = 3 / 2
        data[3, :] = -49 / 18
        data[4, :] = 3 / 2
        data[5, :] = -3 / 20
        data[6, :] = 1 / 90
        offsets = [-3, -2, -1, 0, 1, 2, 3]
    else:
        raise ValueError("Invalid order of FDM.")
    
    return data, offsets   


# def fdm_laplacian_1d_nonuniform(x):
    
#     n = len(x)
#     L = np.zeros((len(x), len(x)))
    
#     for i in range(1, n-1):
#         h1 = x[i] - x[i-1]
#         h2 = x[i+1] - x[i]
#         L[i ,i-1] = 2 / (h1 * (h1 + h2))
#         L[i, i] = -2 / (h1 * h2)
#         L[i, i+1] = 2 / (h2 * (h1 + h2))

#     return L[1:-1, 1:-1], x[1:-1]
        
    

def fdm_laplacian_1d(x_min, x_max, n_inner, order=2):
    """Construct a 1D discretization of the Laplacian operator with Dirichlet
    boundary conditions, using 2, 4 or 6 order FDM, and n_inner inner grid points.
    
    Parameters
    ----------
    x_min : float
        The left boundary value.
    x_max : float
        The right boundary value.
    n_inner : int
        The number of inner grid points.
    order : int, optional
        The order of the FDM. The default is 2.

    Returns
    -------
    scipy.sparse.csr_matrix
        The 1D Laplacian operator.
    numpy.ndarray
        The grid points, including end points.
    """

    # # Grid spacing
    # dx = (x_max - x_min) / (n_inner + 1)

    # Grid points
    x = np.linspace(x_min, x_max, n_inner + 2)
    dx = x[1] - x[0]

    # Construct the 1D Laplacian operator
    data, offsets = laplace_stencil_1d(order=order)
    data = np.repeat(data, n_inner, axis=1)
    # Construct CSR matrix    
    L = spdiags(data, offsets, n_inner, n_inner) / dx**2

    # Construct inner-to-full grid transformation matrix
    G = lil_matrix((n_inner + 2, n_inner))
    G[1:-1,:] = np.eye(n_inner)
    
    return L, x, csr_matrix(G)


class cylinder_fdm_3d:
    
    def __init__(self, r_max, z_max, n_r, n_z, n_m):
        """Set up the cylinder coordinate grid. The wavefunction is decomposed into (2*n_m + 1) partial waves, and
        each partial wave is discretized on a grid with n_r x n_z *inner* grid points. The domain is 0 <= r <= r_max, 
        -z_max <= z <= z_max.
        
        Args:
            r_max (float): right endpoint for radial domain, 0 <= r <= r_max
            z_max (float): right endpoint for z domain, -z_max <= z <= z_max
            n_r (int): number of inner grid points
            n_z (int): number of inner grid points
            n_m (int): maximum angular momentum, i.e., -n_m <= m <= n_m
        """
        
        # Set up grid parameters
        self.r_max = r_max
        self.z_max = z_max
        self.n_r = n_r
        self.n_z = n_z
        self.n_m = n_m
        
        # Compute radial Laplacians and grid
        L_r_neumann, r, G_r_neumann = radial_fdm_laplacian(r_max, n_r, left_bc = 'neumann')
        L_r_dirichlet, r, G_r_dirichlet = radial_fdm_laplacian(r_max, n_r, left_bc = 'neumann')
        self.r = r
        self.L_r_neumann = L_r_neumann
        self.L_r_dirichlet = L_r_dirichlet
        self.G_r_neumann = G_r_neumann
        self.G_r_dirichlet = G_r_dirichlet
        self.r_inner = self.r[1:-1]
        
        # Compute vertical Laplacians and grid
        L_z, z, G_z = fdm_laplacian_1d(-z_max, z_max, n_z, order=2)
        self.L_z = L_z
        self.z = z
        self.G_z = G_z
        self.z_inner = self.z[1:-1]
        self.T_z = -0.5*self.L_z

        # Transition from reduced to full wavefunction
        self.Rm12 = spdiags(self.r_inner**(-.5), 0, self.n_r, self.n_r)
        # Transition from full to reduced wavefunction
        self.R12 = spdiags(self.r_inner**(.5), 0, self.n_r, self.n_r)
        
        
        
        # Set up radial Kinetic energy for each |m|
        self.T_m = []
        for m in range(self.n_m+1):
            if m == 0:
                self.T_m.append( -0.5*self.L_r_neumann  )
            else:
                self.T_m.append( -0.5*self.L_r_dirichlet + spdiags(.5*m**2/self.r_inner**2, 0, self.n_r, self.n_r))
                
                
                
        # Wavefunction shape
        self.shape = (2*n_m + 1, n_r, n_z)
        self.n_dof = np.prod(self.shape)
               
    
        # Debug info
        ic()
        ic(self.n_r, self.n_z, self.n_m)
        ic(self.r_max, self.z_max)
        
    def set_potential(self, V_m):
        """ Set the scalar potential.
        
        The function accepts a list of potentials, interpreted as a partial
        wave expansion of the potential. The list is assumed to have `len(V_m) = 2*V_m_max+1` 
        entries, where the `V_m_max+m`-th entry is the potential matrix for `m`.
        
        Args:
            V_m (list of ndarray): list of potential matrices V(r) for each m.
        """
        self.V_m = V_m
        
        self.V_m_max = (len(V_m) - 1)//2
        ic()
        ic(self.V_m_max)
        
        for m in range(-self.V_m_max, self.V_m_max+1):
            assert(self.V_m[m].shape == (self.n_r, self.n_z))
        

    def apply_hamiltonian(self, psi_reduced):
        """Apply the Hamiltonian to a reduced wavefunction.
        
        Args:
            psi_reduced (ndarray): reduced wavefunction of shape (n_r, 2*m_max+1)
            
        Returns:
            Application of the Hamiltonian to psi.
        """
        
        
        assert(psi_reduced.shape == self.shape)
        result = np.zeros(self.shape, dtype=np.complex128)
        
        for m in range(-self.n_m, self.n_m+1):
            m_index = m + self.n_m
            #ic(m, m_index)
            result[m_index,...] = self.T_m[abs(m)] @ psi_reduced[m_index,...] # acts on first dimension, which is radial.
            result[m_index,...] += psi_reduced[m_index,...] @ self.T_z.T # acts on second dimension, which is vertical.
            for m2 in range(-self.V_m_max, self.V_m_max+1):
                m2_index = self.V_m_max + m2
                m3 = m - m2
                if m3 >= -self.n_m and m3 <= self.n_m:
                    m3_index = m3 + self.n_m
                    #ic(m_index, m2_index, m3_index)
                    result[m_index,...] += self.V_m[m2_index] * psi_reduced[m3_index,...]
        
        return result
    

    def get_sparse_matrix(self):
        """Compute sparse matrix representation of H."""
        def apply_hamiltonian(psi_vec):
            return self.apply_hamiltonian(psi_vec.reshape(self.shape)).flatten()

        H = LinearOperator((self.n_dof, self.n_dof), matvec=apply_hamiltonian, dtype=np.complex128)
        
        H_mat = lil_matrix((self.n_dof, self.n_dof), dtype=np.complex128)
        
        ic(self.n_dof)
        for i in range(self.n_dof):
            e = np.zeros(self.n_dof)
            e[i] = 1
            H_mat[:,i] = self.apply_hamiltonian(e.reshape(self.shape)).flatten()
            
        return csr_matrix(H_mat)

            


    # def reorder_dimensions(self, psi, order='mzr'):
    #     """Reorder the dimensions of a wavefunction psi. The default order is 'mzr', which means that the wavefunction
    #     is ordered as psi[m, z, r]. Any order is possible, e.g., 'zmr'.
                
    #     Args:
    #         psi (numpy.ndarray): wavefunction
    #         order (str, optional): order of dimensions. Defaults to 'mzr'.
        
    #     Returns:
    #         numpy.ndarray: reordered wavefunction
    #     """
        
    #     if order == 'mzr':
    #         return psi
    #     elif order == 'zmr':
    #         return np.transpose(psi, (1, 0, 2))
    #     elif order == 'rmz':
    #         return np.transpose(psi, (2, 0, 1))
    #     elif order == 'zrm':
    #         return np.transpose(psi, (1, 2, 0))
    #     elif order == 'rmz':
    #         return np.transpose(psi, (2, 0, 1))
    #     elif order == 'zrm':
    #         return np.transpose(psi, (1, 2, 0))
    #     else:
    #         raise ValueError("Unknown order: {}".format(order))
        
        
if __name__ == "__main__":
    
    solver = cylinder_fdm_3d(r_max = 10, z_max = 10, n_r = 100, n_z = 200, n_m = 1)

    rr, zz = np.meshgrid(solver.r_inner, solver.z_inner, indexing='ij')    
    ic(rr.shape, zz.shape)
    
    V_m = []
    V_m.append(0.5*(rr**2 + zz**2))
    solver.set_potential(V_m)
    
    ic('Computing sparse matrix')
    H_mat_sparse = solver.get_sparse_matrix()
    ic('Computing eigenvalues')
    E, U = eigsh(H_mat_sparse, k=6, sigma = 1.4)
    ic(E)
    
    plt.figure()
    psi_0 = U[:,5].reshape(solver.shape)[0,...]
    psi = solver.G_r_neumann @ solver.Rm12 @ psi_0 
    plt.imshow(np.abs(psi)**2, extent=[-solver.z_max, solver.z_max, 0, solver.r_max], aspect='auto', origin='lower')
    plt.title('Numeric')
    plt.xlabel('z')
    plt.ylabel('r')
    plt.colorbar()
    plt.show()
    
    