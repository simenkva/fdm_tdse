import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from scipy.sparse import csr_matrix, identity, spdiags, lil_matrix
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh
from time import time


def radial_fdm_laplacian(r_max, n, left_bc="neumann_simple"):
    """Generate laplacian in 2d polar coordinates, radial part only.

    This function generates the radial part of the laplacian in 2d polar coordinates.
    The matrix L generated is symmetric, and acts on reduced wavefunctions $u(\rho)$ defined as
    $$ u(\rho) = \rho^{1/2} \psi(\rho), $$
    where $\psi(\rho)$ is the full wavefunction.

    The discretization is as follows:
    Discretize [0, r_max] with n+2 points, and apply
    von Neumann or Dicichlet conditions at r=0 and Dirichlet at r=r_max,
    resulting in in total n inner grid points as degrees of freedom.
    resulting in an nxn matrix for the inner grid points.

    The central difference scheme used is of second order, and can be interpreted as a finite
    element discretization with piecewise linear basis functions, using the trapezoidal rule
    for element-wise integration. (This will lead to a diagonal mass matrix, and diagonal potential
    matrices. The mass matrix becomes $diag(r_i)$.


    Args:
        r_max (float): right endpoint
        n (int): number of inner grid points
        left_bc (str, optional): 'neumann' for fancy neumann BC, 'neumann_simple' for simple Neumann, else dirichlet.

    Returns:
        L (csr_matrix): Laplacian.
        r (ndrarray): radial grid with endpoints
        G (csr_matrix): transition matrix from inner to global grid
    """

    # Set up grid
    r = np.linspace(0, r_max, n + 2)
    h = r[1] - r[0]

    # Initialize matrix
    # L = np.zeros((n+2,n+2))
    # set up a sparse matrix that supports assignment
    L = lil_matrix((n + 2, n + 2))

    # Assign matrix elements from analytic formula.
    for i in range(n + 2):
        if i < n + 1:
            L[i, i + 1] = r[i] + h / 2
        if i > 0:
            L[i, i - 1] = r[i] - h / 2
        if i > 0 and i < n + 1:
            L[i, i] = -2 * r[i]
        elif i == 0:
            L[i, i] = -(r[i] + h / 2)
        else:
            L[i, i] = -(r[i] + h / 2)

    # Set up transition matrix
    # G = np.zeros((n+2,n))
    G = lil_matrix((n + 2, n))
    G[1:-1, :] = identity(n)
    if left_bc == "neumann_simple":
        G[0, 0] = 1.0

    elif left_bc == "neumann":
        G[0, 0] = 1.0
        G[0, 0] = 4 / 3
        G[0, 1] = -1 / 3
    else:
        pass

    Rm12 = spdiags(r[1:-1] ** (-0.5), 0, n, n)
    # L1 =  np.diag(r[1:-1]**(-.5)) @ (G.T @ (L @ G) / h**2) @ np.diag(r[1:-1]**(-.5))
    L1 = Rm12 @ (G.T @ (L @ G) / h**2) @ Rm12
    return csr_matrix(L1), r, csr_matrix(G)


class polar_fdm_2d:
    """Class for 2d polar FDM SE solver.

    This class implements a partial wave expansion for the total wavefunction.

    Attributes:
        r_max (float): right endpoint
        n_r (int): number of inner grid points
        m_max (int): maximum angular momentum
        L_neumann (ndarray): radial laplacian with Neumann BC
        L_0 (ndarray): radial laplacian with Dirichlet BC
        r0 (ndarray): radial grid with endpoints
        r_inner (ndarray): inner radial grid points
        G_neumann (ndarray): transition matrix from inner to global grid with Neumann BC
        G_0 (ndarray): transition matrix from inner to global grid with Dirichlet BC
        T_m (list of ndarray): Kinetic energy for each |m|
        V_m (list of ndarray): list of potential matrices V(r) for each m.
        V_m_max (int): number of potential matrices
    """

    def __init__(self, r_max, n_r, m_max):
        """Initialize the polar FDM grid.

        Args:
            r_max (float): right endpoint
            n_r (int): number of inner grid points
            m_max (int): maximum angular momentum
        """
        self.r_max = r_max
        self.n_r = n_r
        self.m_max = m_max

        # Set up radial laplacian for Neumann and Dirichlet BC
        self.L_neumann, self.r0, self.G_neumann = radial_fdm_laplacian(
            r_max, n_r, left_bc="neumann"
        )
        self.L_neumann = csr_matrix(self.L_neumann)
        self.L_0, _, self.G_0 = radial_fdm_laplacian(r_max, n_r, left_bc="dirichlet")
        self.L_0 = csr_matrix(self.L_0)
        self.r_inner = self.r0[1:-1]

        # Transition from reduced to full wavefunction
        self.Rm12 = spdiags(self.r_inner ** (-0.5), 0, n_r, n_r)
        # Transition from full to reduced wavefunction
        self.R12 = spdiags(self.r_inner ** (0.5), 0, n_r, n_r)

        # Set up Kinetic energy for each |m|
        self.T_m = []
        for m in range(m_max + 1):
            if m == 0:
                self.T_m.append(-0.5 * self.L_neumann)
            else:
                self.T_m.append(
                    -0.5 * self.L_0 + np.diag(0.5 * m**2 / self.r_inner**2)
                )

        # Set wavefunction shape
        self.shape = (n_r, 2 * m_max + 1)
        # Set number of degrees of freedom
        self.n_dof = np.prod(self.shape)

    def reduced_to_full(self, psi_reduced):
        """Convert reduced wavefunction to full wavefunction.

        Args:
            psi_reduced (ndarray): reduced wavefunction of shape (n_r, 2*m_max+1)

        Returns:
            psi_full (ndarray): full wavefunction of shape (n_r+2, 2*m_max+1)
        """
        psi_full = self.Rm12 @ psi_reduced
        return psi_full

    def full_to_reduced(self, psi_full):
        """Convert full wavefunction to reduced wavefunction.

        Args:
            psi_full (ndarray): full wavefunction of shape (n_r, 2*m_max+1)

        Returns:
            psi_reduced (ndarray): reduce wavefunction of shape (n_r+2, 2*m_max+1)
        """
        psi_reduced = self.R12 @ psi_full
        return psi_reduced

    def set_potential(self, V_m):
        """Set the scalar potential.

        The function accepts a list of potentials, interpreted as a partial
        wave expansion of the potential. The list is assumed to have `len(V_m) = 2*V_m_max+1`
        entries, where the `V_m_max+m`-th entry is the potential matrix for `m`.

        Args:
            V_m (list of ndarray): list of potential matrices V(r) for each m.
        """
        self.V_m = V_m
        self.V_m_max = (len(V_m) - 1) // 2
        ic(self.V_m_max)

    # def apply_weight(self, psi):
    #     """RHS of the SE with weight function, generalized EVP"""

    #     result = np.zeros(self.shape, dtype=np.complex128)
    #     for m in range(-self.m_max, self.m_max+1):
    #         result[:,m+self.m_max] = self.r_inner * psi[:,m+self.m_max]

    #     return result

    def apply_hamiltonian(self, psi_reduced):
        """Apply the Hamiltonian to a reduced wavefunction.

        Args:
            psi_reduced (ndarray): reduced wavefunction of shape (n_r, 2*m_max+1)

        Returns:
            Application of the Hamiltonian to psi.
        """

        assert psi_reduced.shape == self.shape
        result = np.zeros(self.shape, dtype=np.complex128)

        for m in range(-self.m_max, self.m_max + 1):
            m_index = m + self.m_max
            # ic(m, m_index)
            result[:, m_index] = self.T_m[abs(m)] @ psi_reduced[:, m_index]
            for m2 in range(-self.V_m_max, self.V_m_max + 1):
                m2_index = self.V_m_max + m2
                m3 = m - m2
                if m3 >= -self.m_max and m3 <= self.m_max:
                    m3_index = m3 + self.m_max
                    # ic(m_index, m2_index, m3_index)
                    result[:, m_index] += self.V_m[m2_index] * psi_reduced[:, m3_index]

        return result

    # def get_dense_matrix(self):
    #     """Compute dense matrix representation of H."""
    #     def apply_hamiltonian(psi_vec):
    #         return self.apply_hamiltonian(psi_vec.reshape(self.shape)).flatten()

    #     H = LinearOperator((self.n_dof, self.n_dof), matvec=apply_hamiltonian, dtype=np.complex128)

    #     H_mat = np.zeros((self.n_dof, self.n_dof), dtype=np.complex128)

    #     ic(self.n_dof)
    #     for i in range(self.n_dof):
    #         e = np.zeros(self.n_dof)
    #         e[i] = 1
    #         H_mat[:,i] = self.apply_hamiltonian(e.reshape(self.shape)).flatten()

    #     return H_mat

    def get_sparse_matrix(self):
        """Compute sparse matrix representation of H."""

        def apply_hamiltonian(psi_vec):
            return self.apply_hamiltonian(psi_vec.reshape(self.shape)).flatten()

        H = LinearOperator(
            (self.n_dof, self.n_dof), matvec=apply_hamiltonian, dtype=np.complex128
        )

        H_mat = lil_matrix((self.n_dof, self.n_dof), dtype=np.complex128)

        ic(self.n_dof)
        for i in range(self.n_dof):
            e = np.zeros(self.n_dof)
            e[i] = 1
            H_mat[:, i] = self.apply_hamiltonian(e.reshape(self.shape)).flatten()

        return csr_matrix(H_mat)


if __name__ == "__main__":
    # L, r, G = radial_fdm_laplacian(1, 5, left_bc = 'neumann')
    # print(L.toarray())
    # print(G.toarray())
    # print(r)
    # # plt.spy(L)
    # # plt.show()
    # # plt.spy(G)
    # # plt.show()

    # Solve a shifted Harmonic oscillator.
    solver = polar_fdm_2d(r_max=30, n_r=1000, m_max=2)

    x0 = 0.2
    y0 = 0.1
    alpha = x0 - 1j * y0
    V_m = []
    V_m.append(-alpha * 0.5 * solver.r_inner)
    V_m.append(0.5 * solver.r_inner**2 + 0.5 * np.abs(alpha) ** 2)
    V_m.append(-alpha.conjugate() * 0.5 * solver.r_inner)
    solver.set_potential(V_m)

    # Compute dense matrix version of Hamiltonian.

    H_mat_sparse = solver.get_sparse_matrix()
    #    H_mat = H_mat_sparse.todense()

    A = LinearOperator(
        (solver.n_dof, solver.n_dof), matvec=H_mat_sparse.dot, dtype=np.complex128
    )

    print("Go!")
    start = time()
    # Compute eigenvalues and eigenvectors.
    E, U = eigsh(H_mat_sparse, k=6, sigma=1.0)
    print(time() - start)
    print(E[:10])

    # Plot sparsity pattern. Zoom in for details.
    # plt.figure()
    # plt.spy(H_mat)
    # plt.show()

    print("shape = ", H_mat_sparse.shape)
    print("nnz = ", H_mat_sparse.getnnz())
