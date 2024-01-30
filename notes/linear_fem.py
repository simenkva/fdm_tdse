import numpy as np 
import matplotlib.pyplot as plt

# get gauss legendre quadrature rule for interval [a, b] with m points:
def get_gauss_legendre(a, b, m):
    """Get Gauss-Legendre quadrature rule on [a, b] with m points"""
    x, w = np.polynomial.legendre.leggauss(m)
    x = 0.5*(b-a)*x + 0.5*(b+a)
    w *= 0.5*(b-a)
    return x, w



class RadialLinearFEM:
    def __init__(self, r_max, n_grid, s):
        """ Set up linear FEM on [0, r_max] with n_grid _internal_ grid points. Use s quadrature points per element."""
        
        self.r_max = r_max
        self.n_grid = n_grid
        self.r_nodes = np.linspace(0, r_max, n_grid+2)
        self.h = r_max / (n_grid - 1)
        self.s = 20
        
        self.G_neumann = self.compute_bc_matrices(bc_type_left = "neumann")
        self.G_dirichlet = self.compute_bc_matrices(bc_type_left = "dirichlet")
        
        # set up quadrature points
        self.compute_quadrature(s)
        # compute support indices of basis functions
        self.compute_support_indices()
        
        
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
            
        self.basis_function_vec = np.vectorize(lambda t: basis_function0(t, self.h))
        self.basis_function_deriv_vec = np.vectorize(lambda t: basis_function_deriv0(t, self.h))

        
    def compute_quadrature(self, s):
        # set up quadrature points
        self.s = s
        r_quad0, w_quad0 = get_gauss_legendre(0, self.h, s)
        self.r_quad = np.hstack(
            [r_quad0 + self.r_nodes[k] for k in range(n+1)]
        )
        self.w_quad = np.hstack(
            [w_quad0 for k in range(n+1)]
        )
        
    
    def compute_support_indices(self):
        self.support_indices = []
        for k in range(self.n_grid+2):
            u = self.basis_function_vec(self.r_quad - self.r_nodes[k])
            self.support_indices.append( np.where(np.abs(u) > 0.0)[0] )
            print(self.support_indices[-1])            
        
        
    def compute_bc_matrices(self, bc_type_left = "neumann"):
        """ Compute boundary condition matrix G."""
        
        n_grid = self.n_grid
        # there are n interior grid points
        G = np.zeros((n_grid+2, n_grid))
        G[0, 0] = 1.0 if bc_type_left == "neumann" else 0.0
        G[1:n_grid+1, :] = np.eye(n_grid)
        G[n_grid+1:] = 0.0
        
        return G
    

    def compute_fem_matrices(self, V_func = lambda r: 0.0):
        # compute nodes
        r_nodes = self.r_nodes
        n = self.n_grid
        
            
        r_quad = self.r_quad
        w_quad = self.w_quad
        
        S = np.zeros((n+2, n+2))
        T = np.zeros((n+2, n+2))
        V = np.zeros((n+2, n+2))
        for k in range(n+2):
            for l in range(n+2):
                if np.abs(l-k) <= 1:
                    # compute intersection of supports
                    supp_u = self.support_indices[k]
                    supp_v = self.support_indices[l]
                    supp = np.intersect1d(supp_u, supp_v)
                    
                    rr = r_quad[supp]
                    ww = w_quad[supp]

                    u = self.basis_function_vec(rr - self.r_nodes[k])
                    v = self.basis_function_vec(rr - self.r_nodes[l])
                    u = self.basis_function_vec(self.r_quad - self.r_nodes[k])

                    # u = self.basis_function(rr, r_nodes, k)
                    du = self.basis_function_deriv(rr, r_nodes, k)
                    v = self.basis_function(rr, r_nodes, l)
                    dv = self.basis_function_deriv(rr, r_nodes, l)
                    S[k,l] = np.sum(u*v*rr*ww)
                    T[k,l] = .5 * np.sum(du*dv*rr*ww)
                    V[k,l] = np.sum(u*v*rr*ww*V_func(rr))
                

        return S, T, V, r_nodes


    
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