import numpy as np 
import matplotlib.pyplot as plt

# get gauss legendre quadrature rule for interval [a, b] with m points:
def get_gauss_legendre(a, b, m):
    """Get Gauss-Legendre quadrature rule on [a, b] with m points"""
    x, w = np.polynomial.legendre.leggauss(m)
    x = 0.5*(b-a)*x + 0.5*(b+a)
    w *= 0.5*(b-a)
    return x, w



r_max = 10
n = 10


def compute_fem_matrices(r_max, n, V_func = lambda r: 0.0):
    # compute nodes
    r_nodes = np.linspace(0, r_max, n+2)
    h = r_nodes[1] - r_nodes[0]
    s = 20 # sufficient for exact quad of linear elements and radial volume element.     

    # set up quadrature points
    r_quad0, w_quad0 = get_gauss_legendre(0, h, s)
    print(r_quad0)
    r_quad = np.hstack(
        [r_quad0 + r_nodes[k] for k in range(n+1)]
    )
    w_quad = np.hstack(
        [w_quad0 for k in range(n+1)]
    )
    
    
    def basis_function0(t, dt):
        if t >= -dt and t < 0.0:
            return (t + dt) / dt
        elif t >= 0.0 and t < dt:
            return 1.0 - t/dt
        else:
            return 0.0

    
    def basis_function_deriv0(t, dt):
        if t >= -dt and t < 0.0:
            return 1.0 / dt
        elif t >= 0.0 and t < dt:
            return -1.0/dt
        else:
            return 0.0
        
    basis_function_vec = np.vectorize(lambda t: basis_function0(t, h))
    basis_function_deriv_vec = np.vectorize(lambda t: basis_function_deriv0(t, h))
    
    def basis_function(r, nodes, k):
        return basis_function_vec(r - nodes[k])

    def basis_function_deriv(r, nodes, k):
        return basis_function_deriv_vec(r - nodes[k])
    
    S = np.zeros((n+2, n+2))
    T = np.zeros((n+2, n+2))
    V = np.zeros((n+2, n+2))
    for k in range(n+2):
        u = basis_function(r_quad, r_nodes, k)
        du = basis_function_deriv(r_quad, r_nodes, k)
        for l in range(n+2):
            if np.abs(l-k) <= 1:
                v = basis_function(r_quad, r_nodes, l)
                dv = basis_function_deriv(r_quad, r_nodes, l)
                S[k,l] = np.sum(u*v*r_quad*w_quad)
                T[k,l] = .5 * np.sum(du*dv*r_quad*w_quad)
                V[k,l] = np.sum(u*v*r_quad*w_quad*V_func(r_quad))
            

    return S, T, V, r_nodes

def compute_bc_matrices(r_max, n):
    
    # there are n interior grid points
    G = np.zeros((n+2, n))
    G[0, 0] = 1.0
    G[1:n+1, :] = np.eye(n)
    G[n+1:] = 0.0
    
    return G
    
    
def main():
    m = 0.0
    V_func = lambda r: 0.5*r**2 + .5*m**2*r**(-2)
    n = 10
    r_max = 10
    S0, T0, V0, r = compute_fem_matrices(r_max, n, V_func)
    S0 = np.diag(np.sum(S0, axis=1)) # lump mass matrix
    G = compute_bc_matrices(r_max, n)
    S = G.T @ S0 @ G
    T = G.T @ T0 @ G
    V = G.T @ V0 @ G
    
    print(S)
    print(T)
    
    
    from scipy.linalg import eigh 
    evals, evecs = eigh(T + V, S)
    print(evals)
    
if __name__ == "__main__":
    main()