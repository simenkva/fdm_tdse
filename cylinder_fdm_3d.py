from icecream import ic
import numpy as np
from polar_fdm_2d import radial_fdm_laplacian
from scipy.sparse import spdiags, csr_matrix, csc_matrix, lil_matrix, kron, identity, block_diag, bmat
from scipy.sparse.linalg import LinearOperator, eigsh, expm_multiply, gmres, expm, splu
import matplotlib.pyplot as plt
from time import time
from timeit import timeit


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
        data[0, :] = -1 / 12
        data[1, :] = 16 / 12
        data[2, :] = -30 / 12
        data[3, :] = 16 / 12
        data[4, :] = -1 / 12
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


class CylinderFDM:
    
    def __init__(self, r_max, z_max, n_r, n_z, n_m):
        """Set up the cylinder coordinate grid. The wavefunction is discretized on a 3d grid, where the first dimension
        is the L_z angular momentum quantum number, the second dimension is the radial direction in the xy-plane, and the 
        third dimension is the vertizal z-axis.
        
        The shape of the wavefunction psi is (n_m, n_r, n_z), where n_m is the number of angular momentum quantum numbers, n_r is the number
        of radial grid points, and n_z is the number of vertical grid points. The component psi[i_m, :, :] is the spatial function of the
        partial wave number m. The index i_m is such that 0 <= i_m < n_m, and the corresponding angular momentum
        quantum numver runs from m=0 (i_m=0) to m=n_m/2-1, jumps to m=-n_m/2 and increases to -1 (i_m = n_m-1). This is the default convention
        inherited from the FFT library.

        The spatial domain is 0 <= r <= r_max, -z_max <= z <= z_max.        
        
        Args:
            r_max (float): right endpoint for radial domain, 0 <= r <= r_max
            z_max (float): right endpoint for z domain, -z_max <= z <= z_max
            n_r (int): number of inner grid points
            n_z (int): number of inner grid points
            n_m (int): maximum angular momentum, i.e., -n_m/2 <= m < n_m/2
        """
        
        # Set up grid parameters
        self.r_max = r_max
        self.z_max = z_max
        self.n_r = n_r
        self.n_z = n_z
        self.n_m = n_m
        #  f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] 
        self.m_i = np.fft.fftfreq(self.n_m, 1/self.n_m)
        ic(self.m_i)
        
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

        # angular grid
        self.theta = np.linspace(0, 2*np.pi, self.n_m, endpoint=False)

        # Transition from reduced to full wavefunction
        self.Rm12 = spdiags(self.r_inner**(-.5), 0, self.n_r, self.n_r)
        # Transition from full to reduced wavefunction
        self.R12 = spdiags(self.r_inner**(.5), 0, self.n_r, self.n_r)
        
        
        
        # Set up radial Kinetic energy for each |m|.
        self.T_m = []
        for i_m in range(self.n_m):
            if i_m == 0:
                self.T_m.append( -0.5*self.L_r_neumann  )
            else:
                m = self.m_i[i_m]
                self.T_m.append( -0.5*self.L_r_dirichlet + spdiags(.5*m**2/self.r_inner**2, 0, self.n_r, self.n_r))
                
                
        # set up default time dependent potential modulator, which
        # is trivial
        self.modulator = lambda t: 1.0
        
        # Wavefunction shape
        self.shape = (n_m, n_r, n_z)
        self.n_dof = np.prod(self.shape)
               
    
        # Debug info
        ic()
        ic(self.n_r, self.n_z, self.n_m)
        ic(self.r_max, self.z_max)
        ic(self.n_dof)
        
    def get_trz_meshgrid(self):
        """Return the meshgrid of the angular, radial and vertical coordinates."""
        return np.meshgrid(self.theta, self.r_inner, self.z_inner, indexing='ij')
    
    def get_rz_meshgrid(self):
        """Return the meshgrid of the radial and vertical coordinates."""
        return np.meshgrid(self.r_inner, self.z_inner, indexing='ij')
        
        
    def reduced_to_full(self, psi_reduced, theta=True):
        """Convert a reduced wavefunction on the inner grid to a full wavefunction on the total grid.
        
        Args:
            psi_reduced (ndarray): reduced wavefunction of shape (n_m, n_r, n_z)
            theta (bool): True if the wavefunction is converted tp real space in angular direction, False otherwise.
        
        """
    
        
        ans = np.zeros((self.n_m, self.n_r+2, self.n_z+2), dtype = complex)
        for i_m in range(self.n_m):
            m = self.m_i[i_m]
            if m == 0:
                ans[i_m, :, :] = self.G_r_neumann @ ((self.Rm12 @ psi_reduced[i_m, :, :]) @ self.G_z.T)
            else:
                ans[i_m, :, :] = self.G_r_dirichlet @ ((self.Rm12 @ psi_reduced[i_m, :, :]) @ self.G_z.T)
                
        if theta:
            ans = np.fft.ifft(ans, axis=0, norm='ortho')
            
        return ans
    
    
    def fourier_analysis_of_potential(self, V):
        """ Perform a Fourier analysis of the potential V. """
        
        # make sure we have the right size of the tensor
        assert(V.shape == (self.n_m, self.n_r, self.n_z))
        
        # Fourier transform in the angular direction
        V_FFT = np.fft.fft(V, axis=0) / self.n_m 
        
        # Determine the maximum value of |m| that contributes to the
        # potential.
        m_max = 0
        for i in range(self.n_m):
            m = self.m_i[i]
            N = np.linalg.norm(V_FFT[i, :, :])
            #ic(m, N)
            #ic(V_FFT[i, :, :])
            if N > 1e-10 and np.abs(m) > m_max:
                    m_max = np.abs(m)


        ic(m_max)
        
        # Create a vector of the potential contributions
        V_m = []
        m_list = []
        for i in range(self.n_m):
            m = self.m_i[i]
            if np.abs(m) <= m_max:
                #ic(m)
                m_list.append(m)
                V_m.append(V_FFT[i, :, :])
        # sort the list in order of increasing m.
        idx = np.argsort(m_list)
        m_list = [m_list[i] for i in idx]
        V_m = [V_m[i] for i in idx]
        
        #for i in range(len(V_m)):
        #    ic(m_list[i], np.linalg.norm(V_m[i]))
        
        return V_m
        
    def set_realspace_potential(self, V, rotation_symmetric=False):
        """ Set the scalar potential.
        
        The function accepts a potential matrix V(r, z) if rotation_symmetric is True, 
        and a potential matrix V(theta, r, z) if rotation_symmetric is False. In the latter case,
        the wavefunction is Fourier transformed in the angular direction before the potential is applied.
        In the former case, the potential is applied directly to the wavefunction's radial and vertical
        directions for each partial wave.
        
        Args:
            V (ndarray): potential matrix V(r, z) if rotation_symmetric is True,
                and a potential matrix V(theta, r, z) if rotation_symmetric is False.
            rotation_symmetric (bool): True if the potential is rotation symmetric, False otherwise.
            
        """
        
        
        if rotation_symmetric:
            self.rotation_symmetric_potential = True
            assert(V.shape == (self.n_r, self.n_z))
            self.V = V
            ic('Warning:  rotation symmetric potential not implemented in time propagation yet.')

        else:
            self.rotation_symmetric_potential = False
            assert(V.shape == (self.n_m, self.n_r, self.n_z))
            self.V = V
            self.V_m = self.fourier_analysis_of_potential(V)
            
    def set_td_potential(self, D, rotation_symmetric=False):
        """ Set the time-dependent scalar potential.
        
        The function accepts a potential matrix D(r, z) if rotation_symmetric is True, 
        and a potential matrix D(theta, r, z) if rotation_symmetric is False. In the latter case,
        the wavefunction is Fourier transformed in the angular direction before the potential is applied.
        In the former case, the potential is applied directly to the wavefunction's radial and vertical
        directions for each partial wave.
        
        Args:
            D (ndarray): potential matrix D(r, z) if rotation_symmetric is True,
                and a potential matrix D(theta, r, z) if rotation_symmetric is False.
            rotation_symmetric (bool): True if the potential is rotation symmetric, False otherwise.
            
        """
        
        
        if rotation_symmetric:
            self.rotation_symmetric_td_potential = True
            assert(D.shape == (self.n_r, self.n_z))
            self.D = D
            ic('Warning:  rotation symmetric potential not implemented in time propagation yet.')
        else:
            self.rotation_symmetric_td_potential = False
            assert(D.shape == (self.n_m, self.n_r, self.n_z))
            self.D = D
        
        
    def set_td_potential_modulator(self, modulator):
        """Set the time-dependent potential's modulator, a callable function of time.
        
        The modulator is a function of time that returns a scalar factor that multiplies the time-dependent potential,

        $$ D(t, theta, r, z) = modulator(t) * D(theta, r, z). $$

        """
        
        self.modulator = modulator
        



    def apply_hamiltonian_multi(self, psi_reduced_mat):
        """ Apply Hamiltonian to a set of wavefunctions. NOT TESTED YET. """
        assert(len(psi_reduced_mat.shape) == len(self.shape) + 1)
        assert(psi_reduced_mat.shape[:-1] == self.shape)
        n_vec = psi_reduced_mat.shape[-1]
        
        result = np.zeros_like(psi_reduced_mat)
        for i in range(n_vec):
            result[...,i] = self.apply_hamiltonian(psi_reduced_mat[...,i])
            
        return result
        
    def apply_hamiltonian(self, psi_reduced):
        """Apply the Hamiltonian to a reduced wavefunction.
        
        Args:
            psi_reduced (ndarray): reduced wavefunction of shape (n_r, 2*m_max+1)
            
        Returns:
            Application of the Hamiltonian to psi.
        """
        
        
        assert(psi_reduced.shape == self.shape)
        result = np.zeros(self.shape, dtype=np.complex128)

        # apply kinetic energy        
        for i_m in range(self.n_m):
            m = self.m_i[i_m]
            result[i_m,...] = self.T_m[i_m] @ psi_reduced[i_m,...] # acts on first dimension, which is radial.
            result[i_m,...] += psi_reduced[i_m,...] @ self.T_z.T # acts on second dimension, which is vertical.

        
        # apply potential energy
        if self.rotation_symmetric_potential:
            # potential is only a function of rho and z
            for i_m in range(self.n_m):
                result[i_m,...] += self.V * psi_reduced[i_m,...]
        else:            
            # potential is also a function of theta
            phi_reduced = np.fft.ifft(psi_reduced, axis=0)
            V_phi_reduced = self.V * phi_reduced
            result += np.fft.fft(V_phi_reduced, axis=0)
        
        return result
    
    def apply_td_potential(self, psi_reduced, t):
        """Apply the time-dependent potential to a (reduced) wavefunction.
        
        NOT TESTED YET
        """
        
        assert(psi_reduced.shape == self.shape)
        result = np.zeros(self.shape, dtype=np.complex128)
        
        # apply time-dendent potential energy
        if self.rotation_symmetric_td_potential:
            # potential is only a function of rho and z
            for i_m in range(self.n_m):
                result[i_m,...] += self.modulator(t) * self.D * psi_reduced[i_m,...]
        else:            
            # potential is also a function of theta
            phi_reduced = np.fft.ifft(psi_reduced, axis=0)
            V_phi_reduced = self.modulator(t) * self.D * phi_reduced
            result += np.fft.fft(V_phi_reduced, axis=0)
        
        return result
    

    def get_laplacian_fast(self, ordering='rz'):
        """ Compute Laplace operator for each m. Returns a list of polar coordinate Laplace operators. 
        
        The default ordering 'rz' means that fastest running index is the r index, and the slowest running index is the z index.
        
        
        Args:
            ordering (str): 'rz' for radial and vertical ordering, 'zr' for vertical and radial ordering.
            
        
        
        NOTE: This is a simple test implementation that uses homogeneous Dirichlet boundary conditions (excefor at the origin).
        
        """
        
        if ordering == 'rz':
            T_z_kron = -2*kron(identity(self.n_r, format='csr'), self.T_z, format='csr')
            blocks = []
            for i_m in range(self.n_m):
                T_m_kron = -2*kron(self.T_m[i_m], identity(self.n_z, format='csr'), format='csr')
                blocks.append(T_m_kron + T_z_kron)
        elif ordering == 'zr':
            T_z_kron = -2*kron(self.T_z, identity(self.n_r, format='csr'), format='csr')
            blocks = []
            for i_m in range(self.n_m):
                T_m_kron = -2*kron(identity(self.n_z, format='csr'), self.T_m[i_m], format='csr')
                blocks.append(T_m_kron + T_z_kron)
        else:
            raise ValueError("Invalid ordering.")
        
        return blocks            

    
        
    def get_sparse_matrix_fast(self, kinetic=True, potential=True, potential_td=False):
        """Compute sparse matrix representation of H in a faster way than the
        brute force approach. I have tested that the brute force way and this very
        fast way gives identical results. 
        
        This matrix is not used in propagation, but can be useful for other purposes.
        """
        
        return_me = csr_matrix((self.n_dof, self.n_dof), dtype=np.complex128)
        
        if potential:
            #
            # Potential energy matrix
            #
            if self.rotation_symmetric_potential:
                # since the potential is rotation symmetric, it acts as a diagonal matrix
                # for each m.
                data = np.tile(self.V.flatten(), self.n_m)
                ic(data.shape)
                diagonals = [0]
                self.H_pot = spdiags(data, diagonals, self.n_dof, self.n_dof)
            else:
                V_m = self.V_m # self.fourier_analysis_of_potential(self.V)
                V_m_max = (len(V_m) - 1) // 2
                ic(V_m_max)
                data = np.zeros((2*V_m_max+1, self.n_r * self.n_z), dtype=complex) # data to hold diagonals
                for m in range(-V_m_max, V_m_max+1):
                    m_ind = m + V_m_max
                    m_ind2 = -m + V_m_max
                    ic(m, m_ind)
                    data[m_ind, :] = V_m[m_ind2].flatten() 
                #diagonals = np.arange(-self.V_m_max, self.V_m_max+1) * self.n_r * self.n_z
                diagonals = np.arange(-V_m_max, V_m_max+1)
                # * self.n_r * self.n_z
                #diagonals = np.hstack([diagonals2, diagonals0, diagonals1]) * self.n_r * self.n_z
                
                # duplicate diagonals to account for periodicity
                data = np.vstack([data[0], data, data[-1]])
                data = np.hstack([data]*self.n_m)
                diagonals = np.hstack([diagonals[0]+self.n_m, diagonals, diagonals[-1] - self.n_m]) * self.n_r * self.n_z
                
                ic(diagonals.shape)
                ic(data.shape, diagonals.shape, diagonals)
            
            self.H_pot = spdiags(data, diagonals, self.n_dof, self.n_dof)
            
            return_me += self.H_pot
            
        if potential_td:
            raise NotImplementedError("Time-dependent potential not implemented yet.")
            
            
        if kinetic:

    
            #
            # Kinetic enery matrix
            #
            
            T_z_kron = kron(identity(self.n_r, format='csr'), self.T_z, format='csr')
            blocks = []
            for i_m in range(self.n_m):
                T_m_kron = kron(self.T_m[i_m], identity(self.n_z, format='csr'), format='csr')
                blocks.append(T_m_kron + T_z_kron)
                
            self.H_kin = block_diag(blocks, format='csr')
            return_me += self.H_kin
        
        if kinetic and potential:
            ic('Setting total Hamiltonian as self.H_tot')
            self.H_tot = self.H_kin + self.H_pot
        
        # Return H_kin + H_tot if td_potential is False
        # Return (H_kin + H_tot,  H_pot_td) if td_potential is True
        if potential_td:
            raise NotImplementedError("Time-dependent potential not implemented yet.")
        else:
            return return_me
               
                    
                
            
    def get_sparse_matrix(self):
        """Compute sparse matrix representation of H. This is a brute force approach. The other method
        get_sparse_matrix_fast is much faster. """
        

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

            
        
    def setup_splitting_scheme(self, dt):
        """Set up the  splitting scheme for the time-dependent Schrödinger equation. """
        
        self.dt = dt
        
        # Solver for T_z Crank-Nicholson step.
        self.T_z_lu = splu(csc_matrix(identity(self.n_z) + 0.5j*dt*self.T_z))
        # Solver for T_rho Crank-Nicholson step.
        self.T_rho_lu = []
        for i_m in range(self.n_m):
            self.T_rho_lu.append(splu(csc_matrix(identity(self.n_r) + 0.5j*dt*self.T_m[i_m])))


        
    def propagate_crank_nicolson(self, psi, t = 0):
        """Propagate the wavefunction using the Crank-Nicolson method.
        
        
        Time propagation over a time step h is approximated as follows:
        The full time-dependent Hamiltonian is
        $$ H(t) = T_z + T_\rho + V + U(t), $$
        where $U(t)$ is a time-dependent potential, and where $T_z$ is the kinetic energy in the z-direction, $T_\rho$ is the kinetic energy in the radial direction
        and which includes the angular momentum terms. We employ a symmetric splitting scheme of local order $O(h^3)$ as follows:
        
        $$ \psi(t+h) = U_V(t + 3h/4) U_z U_\rho U_V(t+h/4)\psi(t), $$
        where 
        $$ U_V(t) e^{-i(h/2)(V + U(t))} $$
        and where 
        $$ U_z = e^{-ih T_z}, \quad U_\rho = e^{-ih T_\rho} $$
        
        I should explain this better at some point.        
        
        """
        
        dt = self.dt
        shape = psi.shape
        assert(shape == self.shape)
        
        # First potential step, expoential integration here.
        if hasattr(self, 'D'):
            U = self.V + self.modulator(t + 0.25*dt) * self.D
        else:
            U = self.V
        psi = np.fft.fft(np.exp(-0.5j*dt*U) * np.fft.ifft(psi, axis=0), axis=0)
        
        # Kinetic step, rho direction
        for i_m in range(self.n_m):
            temp = psi[i_m, ...] - 0.5j*dt*self.T_m[i_m] @ psi[i_m, ...]
            psi[i_m,...] = self.T_rho_lu[i_m].solve(temp)
            
        # Kinetic step, z direction
        for i_m in range(self.n_m):
            temp = psi[i_m, ...] - 0.5j*dt*psi[i_m, ...] @ self.T_z.T 
            psi[i_m,...] = self.T_z_lu.solve(temp.T).T
        
        # Second potential step.
        if hasattr(self, 'D'):
            U = self.V + self.modulator(t + 0.75*dt) * self.D
        else:
            U = self.V
        psi = np.fft.fft(np.exp(-0.5j*dt*U) * np.fft.ifft(psi, axis=0), axis=0)
        
                   
        return psi
    
        
        
        
def sample(n, fast = True):
    n_m = 4
    solver = CylinderFDM(r_max = 10, z_max = 10, n_r = n, n_z = n , n_m = n_m)

    rr, zz = solver.get_rz_meshgrid()    
    ic(rr.shape, zz.shape)

    V = 0.5*(rr**2 + zz**2)

    # V_m = []
    # V_m.append(0.5*(rr**2 + zz**2))
    solver.set_realspace_potential(V, rotation_symmetric=True)
    
    ic('Computing sparse matrix')
    
    if fast:
        start = time()
        H_mat_sparse = solver.get_sparse_matrix_fast()
        time_taken_sparse = time() - start
        ic(time_taken_sparse)
    else:
        
        start = time()
        H_mat_sparse = solver.get_sparse_matrix()
        time_taken_sparse_slow = time() - start
        ic(time_taken_sparse_slow)

        
    ic('Computing eigenvalues')
    E, U = eigsh(H_mat_sparse, k=10, sigma = 1.2)
    i = np.argsort(E)
    E = E[i]
    U = U[:,i]
    
    ic(E)
    
    E_error = np.abs(E - np.array([1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4.5]))

    ic(E_error)
    
    
    # P_init = np.random.rand(solver.n_dof, 1)
    # P = solver.imag_time_prop(P_init, dt = 0.01, n_steps = 1000)
    
    
    
    # plot a few eigenstates
    if n == 100:
        for k in [0, 1, 2, 3, 4]:
            psi_0 = U[:,k].reshape(solver.shape)[0,...]
            psi = solver.G_r_neumann @ solver.Rm12 @ psi_0 
            plt.figure()
            plt.imshow(np.abs(psi)**2, extent=[-solver.z_max, solver.z_max, 0, solver.r_max], aspect='auto', origin='lower')
            plt.title(f'psi_{k}, E_{k} = {E[k]}')
            plt.xlabel('z')
            plt.ylabel('r')
            plt.colorbar()
        
    return E_error
    
    # plt.figure()
    # psi_0 = U[:,0].reshape(solver.shape)[0,...]
    # psi = solver.G_r_neumann @ solver.Rm12 @ psi_0 
    # plt.imshow(np.abs(psi)**2, extent=[-solver.z_max, solver.z_max, 0, solver.r_max], aspect='auto', origin='lower')
    # plt.title('Numeric')
    # plt.xlabel('z')
    # plt.ylabel('r')
    # plt.colorbar()
    # plt.show()
 
    
    
if __name__ == "__main__":
    

    n_list = np.array([50, 70, 80, 100])
    E_error = np.zeros((len(n_list), 10))
    for i in range(len(n_list)):
        E_error[i] = sample(n_list[i])
        ic(E_error[i])
    
    plt.figure()
    for i in range(10):
        alpha, beta = np.polyfit(np.log(1.0/n_list), np.log(E_error[:,i]), 1)
        ic(alpha,beta)
        plt.loglog(1.0/n_list, E_error[:,i], '*-', label=f'E_{i}, alpha = {alpha}')
    plt.legend()
    plt.xlabel('n = n_r = n_z')
    plt.ylabel('Energy error for HO eigenstates, 3d, m=0')
    plt.show()
    
    
    # solver = cylinder_fdm_3d(r_max = 10, z_max = 10, n_r = 100, n_z = 100, n_m = 0)

    # rr, zz = np.meshgrid(solver.r_inner, solver.z_inner, indexing='ij')    
    # ic(rr.shape, zz.shape)
    
    # V_m = []
    # V_m.append(0.5*(rr**2 + zz**2))
    # solver.set_potential(V_m)
    
    # ic('Computing sparse matrix')
    # H_mat_sparse = solver.get_sparse_matrix()
    # ic('Computing eigenvalues')
    # E, U = eigsh(H_mat_sparse, k=10, sigma = 1.5)
    # ic(E)
    
    # plt.figure()
    # psi_0 = U[:,0].reshape(solver.shape)[0,...]
    # psi = solver.G_r_neumann @ solver.Rm12 @ psi_0 
    # plt.imshow(np.abs(psi)**2, extent=[-solver.z_max, solver.z_max, 0, solver.r_max], aspect='auto', origin='lower')
    # plt.title('Numeric')
    # plt.xlabel('z')
    # plt.ylabel('r')
    # plt.colorbar()
    # plt.show()
    
    