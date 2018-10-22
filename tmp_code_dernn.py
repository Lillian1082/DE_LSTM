import numpy as np
import sympy
from numpy.core.multiarray import ndarray
from scipy.sparse import csgraph
from scipy.sparse.csgraph import laplacian

# MAX_VALUE = 10
# MIN_VALUE = 1
# K = 10
delta = np.ones((10,2))
print("delta:", delta)

a = sympy.hessian(delta)

print("a:", a)
