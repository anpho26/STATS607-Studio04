import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

def test_bootstrap_integration():
    """
    Integration: bootstrap_sample → R_squared → bootstrap_ci.
    This will initially fail until both R_squared and bootstrap_ci are implemented.
    """
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(size=n)
    X = np.c_[np.ones(n), x]          # intercept + x
    beta = np.array([1.0, 2.0])
    y = X @ beta + rng.normal(scale=0.4, size=n)

    # 1) point estimate via R^2
    point = R_squared(X, y)

    # 2) bootstrap distribution of R^2 using bootstrap_sample
    reps = bootstrap_sample(X, y, R_squared, n_bootstrap=200)

    # 3) percentile CI from the bootstrap reps
    low, high = bootstrap_ci(reps, alpha=0.05)

    # basic end-to-end assertions
    assert reps.shape == (200,)
    assert 0.0 <= low < high <= 1.0
    assert low <= point <= high

# TODO: Add your unit tests here

# -------------------------
# Tests: bootstrap_sample
# -------------------------

def test_bootstrap_sample_basic_shape():
    """bootstrap_sample returns a 1D array of length n_bootstrap."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.normal(size=40)

    def stat_mean(_X, _y):
        return float(np.mean(_y))

    B = 100
    reps = bootstrap_sample(X, y, stat_mean, n_bootstrap=B)
    assert isinstance(reps, np.ndarray)
    assert reps.shape == (B,)

def test_bootstrap_sample_constant_y():
    """With constant y and stat=mean, every bootstrap replicate equals that constant."""
    X = np.arange(12).reshape(6, 2).astype(float)
    y = np.full(6, 5.0)
    reps = bootstrap_sample(X, y, lambda _X, _y: float(np.mean(_y)), n_bootstrap=25)
    assert np.allclose(reps, 5.0)

def test_bootstrap_sample_input_validation():
    """bootstrap_sample raises ValueError on invalid inputs and sizes."""
    X = np.zeros((5, 2))
    y = np.zeros(5)

    with pytest.raises(ValueError):
        bootstrap_sample(X, y, compute_stat=None)

    with pytest.raises(ValueError):
        bootstrap_sample(X, y, compute_stat=lambda X, y: 0.0, n_bootstrap=0)

    with pytest.raises(ValueError):
        bootstrap_sample(X[:0], y[:0], compute_stat=lambda X, y: 0.0, n_bootstrap=5)

    with pytest.raises(ValueError):
        bootstrap_sample(X, y[:4], compute_stat=lambda X, y: 0.0, n_bootstrap=5)