import numpy as np
from scipy.interpolate import BSpline


class SplineBasis:
    """
    Constructs a cubic B-spline basis for survival baseline hazards.
    """

    def __init__(self, df: int = 8, degree: int = 3):
        if df < degree + 1:
            raise ValueError(f"df ({df}) must be >= degree + 1 ({degree + 1})")
        self.df = df
        self.degree = degree
        self.knots = None
        self.t_min = None
        self.t_max = None

    def fit(self, times: np.ndarray) -> "SplineBasis":
        """
        Calculates knot positions based on the distribution of training times.
        """
        valid_times = times[times > 0]
        if len(valid_times) == 0:
            valid_times = times

        self.t_min = 0.0
        self.t_max = np.max(valid_times)

        n_interior = self.df - self.degree - 1
        if n_interior > 0:
            quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior_knots = np.quantile(valid_times, quantiles)
        else:
            interior_knots = np.array([])

        self.knots = np.concatenate(
            [
                [self.t_min] * (self.degree + 1),
                interior_knots,
                [self.t_max] * (self.degree + 1),
            ]
        )

        # Ensure knots are strictly non-decreasing
        self.knots = np.sort(self.knots)
        return self

    def transform(self, times: np.ndarray) -> np.ndarray:
        """
        Evaluates the B-spline basis matrix B(t) at the given times.
        Clips times to [t_min, t_max] to gracefully handle predictions out of bounds.
        """
        if self.knots is None:
            raise ValueError("SplineBasis is not fitted yet.")

        # Clip times to avoid evaluating to 0 outside the boundary knots
        clipped_times = np.clip(times, self.t_min, self.t_max)

        # BSpline.design_matrix returns a sparse array, convert to dense
        mat = BSpline.design_matrix(clipped_times, self.knots, self.degree).toarray()

        return mat

    def get_integration_matrix(
        self, times: np.ndarray, n_grid: int = 200
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a fine grid and a mapping matrix to perform numerical integration
        H_0(t_i) = \\int_0^{t_i} \\lambda_0(u) du using the trapezoidal rule.

        Args:
            times: Array of times t_i to integrate up to (shape N).
            n_grid: Number of points for the fine numerical grid.

        Returns:
            B_grid: Spline basis evaluated at the fine grid points (n_grid, df).
            weights: Quadrature weights for the grid (n_grid,).
            mask: A (N, n_grid) boolean mask indicating which grid points fall before each t_i.
        """
        if self.knots is None:
            raise ValueError("SplineBasis is not fitted yet.")

        max_time = np.max(times)
        if max_time <= 0:
            max_time = self.t_max

        grid_times = np.linspace(0.0, max_time, n_grid)
        dt = grid_times[1] - grid_times[0]

        # Trapezoidal weights
        weights = np.full(n_grid, dt)
        weights[0] = dt / 2.0
        weights[-1] = dt / 2.0

        B_grid = self.transform(grid_times)

        # Create mask: M[i, j] = 1 if grid_times[j] <= times[i]
        mask = grid_times[None, :] <= times[:, None]

        return B_grid, weights, mask
