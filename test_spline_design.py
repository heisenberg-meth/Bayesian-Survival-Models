import numpy as np
from scipy.interpolate import BSpline

def generate_knots(times, df=8, degree=3):
    n_interior = df - degree - 1
    if n_interior < 0:
        raise ValueError("df must be >= degree + 1")
    
    t_min, t_max = np.min(times), np.max(times)
    # Quantiles for interior knots
    quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
    interior_knots = np.quantile(times, quantiles)
    
    # Augmented knot vector
    t = np.concatenate([
        [t_min] * (degree + 1),
        interior_knots,
        [t_max] * (degree + 1)
    ])
    return t

times = np.array([0.1, 1.0, 2.0, 5.0, 10.0])
t = generate_knots(times, df=5, degree=3)
mat = BSpline.design_matrix(times, t, 3).toarray()
print(mat.shape)
print(mat)
