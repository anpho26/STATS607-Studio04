import numpy as np

"""
Strong linear model in regression
    Y = X beta + eps, where eps ~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution.
    If you include an intercept beta_0:
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic by resampling (X, y) pairs.

    Parameters
    ----------
    X : array-like of shape (n, p+1)
        Design matrix including intercept column.
    y : array-like of shape (n,)
        Response vector.
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y).
        It must have the signature `compute_stat(X, y) -> float`.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples to generate.

    Returns
    -------
    numpy.ndarray of shape (n_bootstrap,)
        Bootstrap distribution of the statistic.

    Raises
    ------
    ValueError
        If X and y have incompatible shapes, or if n_bootstrap <= 0.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Basic validation
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")

    n = X.shape[0]
    stats = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        X_resampled = X[idx]
        y_resampled = y[idx]
        stats[i] = compute_stat(X_resampled, y_resampled)

    return stats


def bootstrap_ci(bootstrap_stats, ci=0.95):
    """
    Calculate confidence interval from bootstrap statistics.

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics.
    ci : float, default 0.95
        Confidence level in (0, 1).

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI.
    
    Raises
    ------
    ValueError
        If samples is empty or not 1D, or if ci is not in (0, 1).
    """
    samp = np.asarray(bootstrap_stats, dtype=float)

    if samp.ndim != 1:
        raise ValueError("samples must be a 1D array.")
    if samp.size == 0:
        raise ValueError("samples cannot be empty.")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in the open interval (0, 1).")

    alpha = 1.0 - ci
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    low, high = np.quantile(samp, [lower_q, upper_q])

    return float(low), float(high)


def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix (with intercept column if needed).
    y : array-like, shape (n,)
        Response vector.

    Returns
    -------
    float
        R-squared value (can be negative if the model is worse than mean-only).
    
    Raises
    ------
    ValueError
        If inputs have incompatible shapes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    n = X.shape[0]
    if n == 0:
        raise ValueError("X and y must be non-empty.")
    if y.shape[0] != n:
        raise ValueError("X and y must have the same number of rows.")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    sse = float(np.dot(resid, resid))
    y_bar = float(np.mean(y))
    sst = float(np.sum((y - y_bar) ** 2))

    if sst == 0.0:
        return 1.0 if sse == 0.0 else 0.0

    return float(1.0 - sse / sst)
