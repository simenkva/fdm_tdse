from scipy.sparse import spdiags, kron, identity
import numpy as np
from icecream import ic

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


def fdm_laplacian_1d_nonuniform(x):
    
    n = len(x)
    L = np.zeros((len(x), len(x)))
    
    for i in range(1, n-1):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        L[i ,i-1] = 2 / (h1 * (h1 + h2))
        L[i, i] = -2 / (h1 * h2)
        L[i, i+1] = 2 / (h2 * (h1 + h2))

    return L[1:-1, 1:-1], x[1:-1]
        
    

def fdm_laplacian_1d(x_min, x_max, n, order=2):
    """Construct a 1D discretization of the Laplacian operator with Dirichlet
    boundary conditions, using 2, 4 or 6 order FDM.
    
    Parameters
    ----------
    x_min : float
        The left boundary value.
    x_max : float
        The right boundary value.
    n : int
        The number of grid points.
    order : int, optional
        The order of the FDM. The default is 2.

    Returns
    -------
    scipy.sparse.csr_matrix
        The 1D Laplacian operator.
    numpy.ndarray
        The grid points.
    """

    # Grid spacing
    dx = (x_max - x_min) / (n - 1)

    # Grid points
    x = np.linspace(x_min, x_max, n)

    # Construct the 1D Laplacian operator
    data, offsets = laplace_stencil_1d(order=order)
    data = np.repeat(data, n, axis=1)
    # Construct CSR matrix    
    L = spdiags(data, offsets, n, n) / dx**2

    return L, x

def fdm_laplacian_3d(x_min, x_max, n, order=2):
    """
    Construct a 3D discretization of the Laplacian operator with Dirichlet
    boundary conditions, using 2, 4 or 6 order FDM.
    """
        
    L0, x = fdm_laplacian_1d(x_min, x_max, n, order=order)
    I = identity(n)
    
    L = kron(kron(L0, I), I) + kron(kron(I, L0), I) + kron(kron(I, I), L0)
    x, y, z = np.meshgrid(x, x, x, indexing='ij')
    
    return L, x.flatten(), y.flatten(), z.flatten()

def fdm_laplacian_conv_kernel_3d(x_min, x_max, n, order=2):
    """
    Construct a 3D discretization of the Laplacian operator with Dirichlet
    boundary conditions, using 2, 4 or 6 order FDM.
    """
        
    data, offsets = laplace_stencil_1d(order=order)
    m = len(data)
    dx = (x_max - x_min) / (n - 1)
    
    L = np.zeros((m, m, m))
    j0 = max(offsets)
    for i in range(m):
        j = offsets[i] + j0
        L[j, j0, j0] += data[j]/dx**2
        L[j0, j, j0] += data[j]/dx**2
        L[j0, j0, j] += data[j]/dx**2
        
    return L
