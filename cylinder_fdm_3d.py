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
        """Set up the cylinder coordinate grid. The wavefunction is decomposed into n_m partial waves, assumed to be an even number, and
        each partial wave is discretized on a grid with n_r x n_z *inner* grid points. The domain is 0 <= r <= r_max, 
        -z_max <= z <= z_max. The wavefunction is thus a tensor psi of shape (n_m, n_r, n_z), where psi[i_m, :, :] is the spatial function of
        partial wave number m. The index i_m is such that 0 <= i_m < n_m, and the corresponding angular momentum
        quantum numver runs from m=0 (i_m=0) to m=n_m/2-1, jumps to m=-n_m/2 and increases to -1 (i_m = n_m-1). This is the default convention
        inherited from the FFT library.
        
        
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
            ic(m, N)
            ic(V_FFT[i, :, :])
            if N > 1e-10 and np.abs(m) > m_max:
                    m_max = np.abs(m)


        ic(m_max)
        
        # Create a vector of the potential contributions
        V_m = []
        m_list = []
        for i in range(self.n_m):
            m = self.m_i[i]
            if np.abs(m) <= m_max:
                ic(m)
                m_list.append(m)
                V_m.append(V_FFT[i, :, :])
        # sort the list in order of increasing m.
        idx = np.argsort(m_list)
        m_list = [m_list[i] for i in idx]
        V_m = [V_m[i] for i in idx]
        
        for i in range(len(V_m)):
            ic(m_list[i], np.linalg.norm(V_m[i]))
        
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
        
        
    # def set_potential(self, V_m):
    #     """ Set the scalar potential.
        
    #     The function accepts a list of potentials, interpreted as a partial
    #     wave expansion of the potential. The list is assumed to have `len(V_m) = 2*V_m_max+1` 
    #     entries, where the `V_m_max+m`-th entry is the potential matrix for `m`.
        
    #     Args:
    #         V_m (list of ndarray): list of potential matrices V(r) for each m.
    #     """
    #     self.V_m = V_m
        
    #     self.V_m_max = (len(V_m) - 1)//2
    #     ic()
    #     ic(self.V_m_max)
        
    #     for m in range(-self.V_m_max, self.V_m_max+1):
    #         assert(self.V_m[m].shape == (self.n_r, self.n_z))
        
    # def set_td_potential(self, D_m):
    #     """ Set the time-dependent scalar potential.
        
    #     The function accepts a list of potentials, interpreted as a partial
    #     wave expansion of the potential. The list is assumed to have `len(V_m) = 2*V_m_max+1` 
    #     entries, where the `V_m_max+m`-th entry is the potential matrix for `m`.
        
    #     Args:
    #         V_m (list of ndarray): list of potential matrices V(r) for each m.
    #     """
    #     self.D_m = D_m
        
    #     self.D_m_max = (len(D_m) - 1)//2
    #     ic()
    #     ic(self.D_m_max)
        
    #     for m in range(-self.D_m_max, self.D_m_max+1):
    #         assert(self.D_m[m].shape == (self.n_r, self.n_z))
        



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
    

    def get_sparse_matrix_fast(self, kinetic=True, potential=True, potential_td=False):
        """Compute sparse matrix representation of H in a faster way than the
        brute force approach. I have tested that the brute force way and this very
        fast way gives identical results. """
        
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

            
    # def imag_time_prop_ode(self, P):
    #     """ TESTING """
    #     # P is assumed to have shape (n_dof, n_psi) and to have orthonormal columns
        
    #     n_psi = P.shape[1]
    #     assert(P.shape[0] == self.n_dof)
    #     result = np.zeros((self.n_dof, n_psi), dtype=np.complex128)
    #     for i in range(n_psi):
    #         result[:,i] = self.apply_hamiltonian(P[:,i].reshape(self.shape)).flatten()
        
    #     H = P.conjugate().T @ result
    #     result = result - P @ H
            
    #     return result, np.linalg.eigh(.5*(H + H.T.conjugate()))[0]
        
        
    # def imag_time_prop(self, psi_list, dt, n_steps):
    #     """ TESTING """
        
    #     # psi_list is assumed to have shape (n_dof, n_psi)
        
    #     # orthogonalize
    #     P, R = np.linalg.qr(psi_list)

    #     for i in range(n_steps):    
    #         dP, Evals =  self.imag_time_prop_ode(P)
    #         ic(np.linalg.norm(dP), np.sum(Evals))
    #         P = P + dt * dP
    #         P, R = np.linalg.qr(P)
        
            

    #     return P


       


        
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
        """Propagate the wavefunction using the Crank-Nicolson method."""
        
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
        
        
        
def sample(n, fast = True):
    n_m = 4
    solver = CylinderFDM(r_max = 10, z_max = 10, n_r = n, n_z = n , n_m = n_m)

    rr, zz = np.meshgrid(solver.r_inner, solver.z_inner, indexing='ij')    
    ic(rr.shape, zz.shape)

    x0 = 0.0
    y0 = 0.0
    alpha = x0 - 1j*y0
    z0 = 0.0
    V_m = []
    V_m.append(-alpha*0.5*rr)
    V_m.append(0.5*(zz-z0)**2 + 0.5*rr**2 + 0.5*np.abs(alpha)**2)
    V_m.append(-alpha.conjugate()*0.5*rr)

    # V_m = []
    # V_m.append(0.5*(rr**2 + zz**2))
    solver.set_potential(V_m)
    
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
    
    E_error = np.abs(E - np.array([1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]))

    ic(E_error)
    
    
    # P_init = np.random.rand(solver.n_dof, 1)
    # P = solver.imag_time_prop(P_init, dt = 0.01, n_steps = 1000)
    
    
    
    # plot a few eigenstates
    if n == 100:
        for k in [0, 1, 2, 3, 4]:
            psi_0 = U[:,k].reshape(solver.shape)[n_m,...]
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
    
    