import numpy as np
from scipy.special import erf



class ErfgauPotential:
    """Erfgau potential
    
    $$ V(r) = erf(mu*r)/r + c*exp(-alpha^2*r^2) $$
    
    Args:
    - mu: float, parameter in the potential


    
    """
    def __init__(self, mu=1.0):

        c = 0.923+1.568*mu
        alpha = 0.2411+1.405*mu

        self.mu = mu
        self.c = c
        self.alpha = alpha
        
        self.__call__ = self.potential

    def long_range(self, r):
        
        if r == 0:
            V0 = self.mu
        else:
            V0 = erf(self.mu*r)/r
            
        return V0


        
    def potential(self, x, y, z):

        r2 = (x*x+y*y+z*z)

        V0 = np.vectorize(self.long_range)(r2**.5)
            
        V = V0 + self.c*np.exp(-self.alpha**2*r2)
        return -V

    def potential_radial(self, r):

        r2 = r*r

        V0 = np.vectorize(self.long_range)(r)
            
        V = V0 + self.c*np.exp(-self.alpha**2*r2)
        return -V