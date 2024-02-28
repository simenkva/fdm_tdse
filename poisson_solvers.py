import numpy as np
import scipy as sp


class PoissonSpherical:
    def __init__(self, r, l_max, order=2):
        """
        Initialize the PoissonSolver object.

        Parameters:
        - r (array-like): The radial grid points.
        - l_max (int): The maximum angular momentum quantum number.
        - order (int, optional): The order of the finite difference scheme. Defaults to 2.
        """
        self.r = r
        self.nr = len(self.r) - order
        self.dr = self.r[1] - self.r[0]
        self.r_interior = self.r[order // 2 : -order // 2]

        self.l_max = l_max

        L_diag = -2 * np.ones(self.nr) / self.dr**2
        L_subdiag = np.ones(self.nr) / self.dr**2
        L_subdiag[-1] = 0

        self.A = []
        for l in range(l_max + 1):
            centrifugal = l * (l + 1) / self.r_interior**2
            self.A.append(np.array([L_diag - centrifugal, L_subdiag]))
        self.A = np.array(self.A)

    def solve(self, l, f, w_rmax=0):
        """
        Solve the Poisson equation
            (d^2/dr^2)*w(r) - l(l+1)/r^2*w(r) = f(r),
        for a given angular momentum quantum number l subject to the (Dirichlet) boundary conditions
            w(0) = 0,
            w(r_max) = w_rmax.

        Parameters:
        l (int): The angular momentum quantum number.
        f (ndarray): The right-hand side of the Poisson equation.
        w_rmax (float, optional): The value of w at r_max. Defaults to 0.

        Returns:
        ndarray: The solution w(r) of the Poisson equation.
        """
        # solveh_banded requires the coefficient matrix A to be positive definite,
        # therefore we multiply the equation by -1.
        b = -f
        b[-1] += w_rmax / self.dr**2
        return sp.linalg.solveh_banded(-self.A[l], b, lower=True)


class PoissonCylindrical:
    def __init__(self, Laplacian, rho, z, m=0):
        ########################################
        self.Laplacian = Laplacian
        self.rho = rho
        self.z = z
        self.drho = rho[1] - rho[0]
        self.dz = z[1] - z[0]
        self.rho_inner = rho[1:-1]
        self.z_inner = z[1:-1]
        self.n_rho = len(self.rho_inner)
        self.n_z = len(self.z_inner)
        self.m = m
        #########################################
        g = np.zeros((self.n_rho + 2, self.n_z + 2))
        for i in range(1, self.n_rho + 2):
            g[i, 0] = np.sqrt(2 * np.pi * rho[i] / (rho[i] ** 2 + z[0] ** 2))
            g[i, self.n_z + 1] = np.sqrt(
                2 * np.pi * rho[i] / (rho[i] ** 2 + z[self.n_z + 1] ** 2)
            )

        for j in range(self.n_z + 2):
            g[self.n_rho + 1, j] = np.sqrt(
                2
                * np.pi
                * rho[self.n_rho + 1]
                / (rho[self.n_rho + 1] ** 2 + z[j] ** 2)
            )
        inhomogenous_dirichlet_boundary = np.zeros(
            (self.n_rho + 2, self.n_z + 2)
        )

        for j in range(1, self.n_z + 1):
            inhomogenous_dirichlet_boundary[self.n_rho, j] = (
                -g[self.n_rho + 1, j]
                * (rho[self.n_rho + 1] + self.drho / 2)
                / (
                    np.sqrt(rho[self.n_rho] * rho[self.n_rho + 1])
                    * self.drho**2
                )
            )
            if j == self.n_z:
                inhomogenous_dirichlet_boundary[self.n_rho, self.n_z] -= (
                    g[self.n_rho, j + 1] / self.dz**2
                )
            if j == 1:
                inhomogenous_dirichlet_boundary[self.n_rho, 1] -= (
                    g[self.n_rho, j - 1] / self.dz**2
                )

        for i in range(1, self.n_rho):
            inhomogenous_dirichlet_boundary[i, 1] = -g[i, 0] / self.dz**2
            inhomogenous_dirichlet_boundary[i, self.n_z] = (
                -g[i, self.n_z + 1] / self.dz**2
            )

        self.inhomogenous_dirichlet_boundary_inner = (
            inhomogenous_dirichlet_boundary[1:-1, 1:-1]
        )

    def solve(self, f, inhomgenous_dirichlet=False):
        if inhomgenous_dirichlet:
            f_t = f + self.inhomogenous_dirichlet_boundary_inner
            return self.Laplacian.Linv_v(f_t)
        else:
            return self.Laplacian.Linv_v(f)


class PoissonCylindricalOld:
    def __init__(self, rho, z, m=0):

        self.rho = rho
        self.z = z
        self.drho = rho[1] - rho[0]
        self.dz = z[1] - z[0]
        self.rho_inner = rho[1:-1]
        self.z_inner = z[1:-1]
        self.n_rho = len(self.rho_inner)
        self.n_z = len(self.z_inner)
        self.m = m
        ########################################################################################

        L_rho = np.zeros((self.n_rho, self.n_rho))
        for i in range(self.n_rho):
            L_rho[i, i] = -2 / self.drho**2
            if i < self.n_rho - 1:
                L_rho[i, i + 1] = (self.rho_inner[i] + self.drho / 2) / (
                    self.drho**2
                    * np.sqrt(self.rho_inner[i] * self.rho_inner[i + 1])
                )
            if i > 0:
                L_rho[i, i - 1] = (self.rho_inner[i] - self.drho / 2) / (
                    self.drho**2
                    * np.sqrt(self.rho_inner[i - 1] * self.rho_inner[i])
                )

        if self.m == 0:
            # Add Neumann boundary condition
            L_rho[0, 0] += (self.rho_inner[0] - self.drho / 2) / (
                self.drho**2 * self.rho_inner[0]
            )

        L_rho -= np.diag(self.m**2 / self.rho_inner**2)

        self.lambda_rho, self.U_rho = np.linalg.eigh(L_rho)

        L_z = np.zeros((self.n_z, self.n_z))
        for i in range(self.n_z):
            L_z[i, i] = -2 / self.dz**2
            if i < self.n_z - 1:
                L_z[i, i + 1] = 1 / self.dz**2
            if i > 0:
                L_z[i, i - 1] = 1 / self.dz**2

        self.lamdbda_z, self.U_z = np.linalg.eigh(L_z)

        self.Dinv = 1 / (self.lambda_rho[:, np.newaxis] + self.lamdbda_z)
        ########################################################################################

    def solve(self, f):
        """
        Solve the Poisson equation

        Parameters:
        f (ndarray): The right-hand side of the Poisson equation.

        Returns:
        ndarray: The solution w(rho, z) of the Poisson equation.
        """
        tmp = np.dot(self.U_rho.T, np.dot(f, self.U_z))
        tmp2 = np.multiply(self.Dinv, tmp)
        w_sol = np.dot(self.U_rho, np.dot(tmp2, self.U_z.T))
        return w_sol
