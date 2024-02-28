import numpy as np


class CylinderFockMatrix:
    def __init__(self, Laplacian, V):
        self.Laplacian = Laplacian
        self.V = V

    def F_Psi(self, Psi, W_direct, W_exchange):
        Fpsi = np.zeros(Psi.shape)
        for i in range(Psi.shape[0]):
            Fpsi[i] = -0.5 * self.Laplacian.L_v(Psi[i]) + self.V * Psi[i]
            for j in range(Psi.shape[0]):
                Fpsi[i] += 2 * W_direct[j] / np.sqrt(2 * np.pi) * Psi[i]
                if i == j:
                    Fpsi[i] -= W_direct[j] / np.sqrt(2 * np.pi) * Psi[j]
                else:
                    Fpsi[i] -= W_exchange / np.sqrt(2 * np.pi) * Psi[j]

        return Fpsi
