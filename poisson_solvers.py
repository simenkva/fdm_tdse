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
