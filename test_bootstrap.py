import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

# -------------------------
# Integration test
# -------------------------

def test_bootstrap_integration():



# -------------------------
# Unit tests: bootstrap_sample
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


# -------------------------
# Unit tests: bootstrap_ci
# -------------------------

def test_bootstrap_ci_percentile_and_validation():
    """
    Test bootstrap_ci:
    - Ensures returned interval matches numpy quantiles on a known distribution.
    - Validates error handling for invalid inputs.
    """
    samp = np.arange(100, dtype=float)

    # Compare bootstrap_ci to numpy's quantile for 80% CI
    low, high = bootstrap_ci(samp, ci=0.80)
    ql, qh = np.quantile(samp, [0.10, 0.90])
    assert low == pytest.approx(ql)
    assert high == pytest.approx(qh)

    # Input validation tests
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))  # empty array

    with pytest.raises(ValueError):
        bootstrap_ci(np.array([[1, 2], [3, 4]]))  # wrong shape

    with pytest.raises(ValueError):
        bootstrap_ci(samp, ci=0.0)

    with pytest.raises(ValueError):
        bootstrap_ci(samp, ci=1.0)

    with pytest.raises(ValueError):
        bootstrap_ci(samp, ci=-0.1)


def test_bootstrap_ci_contains_true_mean():
    """
    Checks whether the CI for the mean contains the true mean.
    Generates bootstrap reps first, then applies bootstrap_ci.
    """
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=2.0, size=200)

    # Generate bootstrap reps of the mean
    reps = bootstrap_sample(data.reshape(-1, 1), data,
                            lambda _X, _y: float(np.mean(_y)),
                            n_bootstrap=500)

    # CI from bootstrap reps
    low, high = bootstrap_ci(reps, ci=0.95)

    assert low < 5.0 < high


# -------------------------
# Unit tests: R_squared
# -------------------------

def test_R_squared_perfect_fit():
    """Perfect linear relationship should yield R^2 = 1.0."""
    x = np.linspace(0, 10, 50)
    X = np.c_[np.ones_like(x), x]
    y = 2 + 3 * x
    r2 = R_squared(X, y)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_R_squared_mean_only_model():
    """Intercept-only model should have R^2 ≈ 0 with random noise."""
    rng = np.random.default_rng(42)
    y = rng.normal(size=100)
    X = np.ones((100, 1))
    r2 = R_squared(X, y)
    assert r2 == pytest.approx(0.0, abs=1e-12)

def test_R_squared_input_validation():
    """Invalid shapes should raise ValueError."""
    X = np.zeros((5, 2))
    y = np.zeros(5)

    with pytest.raises(ValueError):
        R_squared(X.reshape(10), y)  # X not 2D

    with pytest.raises(ValueError):
        R_squared(X, y.reshape(5, 1))  # y not 1D

    with pytest.raises(ValueError):
        R_squared(X[:4], y)  # mismatched lengths
