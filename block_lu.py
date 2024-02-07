from scipy.sparse import spdiags, kron, identity
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from time import time
from icecream import ic

def extract_tridiag_blocks(X, block_size, n_blocks):
    """
    Extract tridiagonal blocks from a block tridiagonal matrix X.
    
    Args:
    - X: scipy.sparse matrix, block tridiagonal matrix
    - block_size: int, size of each block
    - n_blocks: int, number of blocks
    
    Returns:
    - D: list of scipy.sparse matrices, diagonal blocks
    - A: list of scipy.sparse matrices, subdiagonal blocks
    - C: list of scipy.sparse matrices, superdiagonal blocks
    """

    n = X.shape[0]
    assert(n == block_size*n_blocks)

    n_blocks = n_blocks
    D = []
    C = []
    A = []
    for i in range(n_blocks):
        D.append( X[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size]  )
        if i < n_blocks - 1:
            C.append( X[i*block_size:(i+1)*block_size, (i+1)*block_size:(i+2)*block_size]  )
        if i > 0:
            A.append( X[i*block_size:(i+1)*block_size, (i-1)*block_size:i*block_size]  )

    return D, A, C

def block_lu(D, A, C):
    """
    Compute the block-LU decomposition of a block tridiagonal matrix. The output
    can be fed to block_lu_solve to solve a block-tridiagonal system of equations.
    
    Args:
    - D: list of scipy.sparse matrices, diagonal blocks
    - A: list of scipy.sparse matrices, subdiagonal blocks
    - C: list of scipy.sparse matrices, superdiagonal blocks
    
    Returns:
    - L: list of, lower triangular blocks
    - lu_U: list of tuples, LU factorization of the upper triangular blocks U
    - C: list of matrices, superdiagonal blocks (from input)
    
    """    
    n_blocks = len(D)
    
    U = []
    L = []
    
    U.append(D[0].todense())
    for k in range(1, n_blocks):
        L.append(A[k-1].todense() @ np.linalg.inv(U[k-1]))
        U.append(D[k].todense() - L[k-1] @ C[k-1].todense())
        
    lu_U = [lu_factor(U[i]) for i in range(n_blocks)]
    
    return L, lu_U, C


def block_lu_solve(L, lu_U, C, b):
    """
    Backward and forward substitution to solve a block-tridiagonal system of equations.
    
    Args:
    - L: list of, lower triangular blocks
    - lu_U: list of tuples, LU factorization of the upper triangular blocks U
    - C: list of matrices, superdiagonal blocks
    - b: numpy array, right-hand side
    
    Returns:
    - x: numpy array, solution
    """

    n1 = L[0].shape[0]
    n_blocks = len(L) + 1
    y = np.zeros_like(b)
    y[:n1] = b[:n1]
    for i in range(1, n_blocks):
        y[i*n1:(i+1)*n1] = b[i*n1:(i+1)*n1] - L[i-1] @ y[(i-1)*n1:i*n1]

    x = np.zeros_like(b)
    x[(n_blocks-1)*n1:] = lu_solve(lu_U[-1], y[(n_blocks-1)*n1:])
    
    for m in range(n_blocks-2, -1, -1):
        #ic(m)
        x[m*n1:(m+1)*n1] = lu_solve(lu_U[m], y[m*n1:(m+1)*n1] - C[m] @ x[(m+1)*n1:(m+2)*n1])

    return x

if __name__ == "__main__":
    # set up 2d finite difference operator
    n = 200
    e = np.ones(n)
    Lap1d = spdiags(np.vstack([-e, 2*e, -e]), [-1, 0, 1], n, n)
    Lap = kron(Lap1d, identity(n)) + kron(identity(n), Lap1d)
    rhs = np.random.rand(n*n)
    
    D, A, C = extract_tridiag_blocks(Lap, n, n)
    lu_Lap = block_lu(D, A, C)
    start = time()
    x = block_lu_solve(*lu_Lap, rhs)
    ic(time() - start)

    ic(Lap.shape)
    ic(x.shape)
    ic(rhs.shape)
    residual = Lap @ x - rhs
    ic(np.linalg.norm(residual, ord=np.inf))
    ic(np.linalg.norm(residual, ord=2))
    ic(np.linalg.norm(residual, ord=1))