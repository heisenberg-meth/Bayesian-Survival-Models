import numpy as np
from src.models.bayesian.splines import SplineBasis

def test_splines():
    times = np.array([0.1, 1.0, 2.0, 5.0, 10.0])
    spline = SplineBasis(df=8, degree=3)
    spline.fit(times)
    
    # Check knots
    assert len(spline.knots) == 8 + 3 + 1
    
    # Check transform
    B = spline.transform(times)
    assert B.shape == (5, 8)
    assert not np.any(np.isnan(B))
    
    # Check bounds
    out_of_bounds = np.array([-1.0, 15.0])
    B_out = spline.transform(out_of_bounds)
    assert B_out.shape == (2, 8)
    assert not np.any(np.isnan(B_out))
    
    # Check integration grid
    B_grid, weights, mask = spline.get_integration_matrix(times, n_grid=100)
    assert B_grid.shape == (100, 8)
    assert weights.shape == (100,)
    assert mask.shape == (5, 100)
    
    print("ALL PASS")

test_splines()
