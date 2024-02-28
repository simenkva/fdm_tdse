import numpy as np
from scipy.sparse import (
    spdiags,
    csr_matrix,
    csc_matrix,
    lil_matrix,
    kron,
    identity,
    block_diag,
    bmat,
)


class CylinderLaplacian:
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

        L_rho = lil_matrix((self.n_rho, self.n_rho))
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

        L_z = lil_matrix((self.n_z, self.n_z))
        for i in range(self.n_z):
            L_z[i, i] = -2 / self.dz**2
            if i < self.n_z - 1:
                L_z[i, i + 1] = 1 / self.dz**2
            if i > 0:
                L_z[i, i - 1] = 1 / self.dz**2

        self.L = kron(L_rho, identity(self.n_z)) + kron(
            identity(self.n_rho), L_z
        )
        ########################################################################################
        self.lambda_rho, self.U_rho = np.linalg.eigh(L_rho.todense())
        self.lamdbda_z, self.U_z = np.linalg.eigh(L_z.todense())
        self.Dinv = 1 / (self.lambda_rho[:, np.newaxis] + self.lamdbda_z)

    def L_v(self, v):
        return self.L.dot(v)

    def Linv_v(self, v):
        tmp = np.dot(self.U_rho.T, np.dot(v, self.U_z))
        tmp2 = np.multiply(self.Dinv, tmp)
        Linv_v = np.dot(self.U_rho, np.dot(tmp2, self.U_z.T))
        return Linv_v
