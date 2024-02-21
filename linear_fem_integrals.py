import numpy as np 
import matplotlib.pyplot as plt
from icecream import ic
from scipy.linalg import eigh 
    
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

def compute_bc_matrices(r_max, n, left_bc = "neumann", gamma = 1.0, h = 0.0):
    
    # there are n interior grid points
    G = np.zeros((n+2, n))
    G[1:n+1, :] = np.eye(n)
    #G[n+1:] = 0.0
    
    #G = lil_matrix((n + 2, n))
    if left_bc == "neumann_simple":
        G[0, 0] = 1.0

    elif left_bc == "neumann":
        G[0, 0] = 4 / 3
        G[0, 1] = -1 / 3

    elif left_bc == "robin":
        G[0, 0] = 2/(h*gamma + 3/2)
        G[0, 1] = -1 / (2*h*gamma + 3)

    else:
        G = np.zeros((n+2, n+1))
        G[:n+1, :n+1] = np.eye(n+1)
        pass    
    
    return G
    

def calculate(n = 10, left_bc = "neumann", gamma = -1.0):
#    V_func = lambda r: 0.5*r*r
    V_func = lambda r: -1/r

    r_max = 10
    S0, T0, V0, r = compute_fem_matrices(r_max, n, V_func)
    h = r[1] - r[0]
    S0 = np.diag(np.sum(S0, axis=1)) / h # lump mass matrix
    T0 = T0 / h
    V0 = np.diag(np.sum(V0, axis=1)) / h # lump potential matrix
#    V0 = V0 / h
    if left_bc == "neumann":
        G = compute_bc_matrices(r_max, n, left_bc = "neumann")
        ic(G.shape, S0.shape, T0.shape, V0.shape)
        S = G.T @ S0 @ G
        T = G.T @ T0 @ G
        V = G.T @ V0 @ G
        rr = r[1:-1]
    elif left_bc == "neumann_simple":
        G = compute_bc_matrices(r_max, n, left_bc = "neumann")
        ic(G.shape, S0.shape, T0.shape, V0.shape)
        S = G.T @ S0 @ G
        T = G.T @ T0 @ G
        V = G.T @ V0 @ G
        rr = r[1:-1]
    elif left_bc == "robin":
        G = compute_bc_matrices(r_max, n, left_bc = "robin", gamma = gamma, h = h)
        ic(G.shape, S0.shape, T0.shape, V0.shape)
        S = G.T @ S0 @ G
        T = G.T @ T0 @ G
        V = G.T @ V0 @ G
        rr = r[1:-1]
    elif left_bc == "none":
        G = compute_bc_matrices(r_max, n, left_bc = "none")
        ic(G.shape, S0.shape, T0.shape, V0.shape)
        S = G.T @ S0 @ G
        T = G.T @ T0 @ G
        V = G.T @ V0 @ G
        rr = r[:-1]
    else:
        raise ValueError("Invalid left_bc")
        
    
    # ic(r)
    # ic(V_func(r))
    # ic(S)
    # ic(T)
    # ic(np.diag(V)/np.diag(S))

    
    
    evals, evecs = eigh(T + V, S)
    print(evals)
    
    return evals, G @ evecs, r

def main():
    
    n = 100
    evals0, evecs0, r0 = calculate(n = n, left_bc = "none")
    evals1, evecs1, r1 = calculate(n = n, left_bc = "neumann_simple")
    evals2, evecs2, r2 = calculate(n = n, left_bc = "neumann")
    evals3, evecs3, r3 = calculate(n = n, left_bc = "robin", gamma=-2)
    
    
    ic(evals0[0], evals1[0], evals2[0], evals3[0])
    ic(len(r0), len(r1), len(r2), len(r3))
    
    plt.figure()
    plt.plot(r0, evecs0[:, 0], label="none", marker="o")
    plt.plot(r1, evecs1[:, 0], label="neumann_simple", marker="s")
    plt.plot(r2, evecs2[:, 0], label="neumann", marker="^")
    plt.plot(r3, evecs3[:, 0], label="robin", marker="d")
    plt.legend()
    plt.show()
    
    # # analytical solution
    # E_analytical = -1/(2*(.5-np.arange(1, n+1))**2)
    # ic(E_analytical[:4])

        
if __name__ == "__main__":
    main()