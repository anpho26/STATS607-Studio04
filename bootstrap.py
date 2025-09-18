
"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""
import numpy as np

def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    

    ....
    """
    pass

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    Raises
    ------
    ValueError
        If samples is empty or not 1D, or if ci is not in (0, 1).
    """
    samp = np.asarray(bootstrap_stats, dtype=float)

    # Validate inputs
    if samp.ndim != 1:
        raise ValueError("samples must be a 1D array.")
    if samp.size == 0:
        raise ValueError("samples cannot be empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in the open interval (0, 1).")

    # Percentile bounds
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
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
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

    # OLS via least squares (robust to singular X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    sse = float(np.dot(resid, resid))
    y_bar = float(np.mean(y))
    sst = float(np.sum((y - y_bar) ** 2))

    # If y is constant: define R^2 = 1 if perfect fit, else 0
    if sst == 0.0:
        return 1.0 if sse == 0.0 else 0.0

    r2 = 1.0 - sse / sst
    # Numerical guard to keep within [0, 1]
    return float(min(1.0, max(0.0, r2)))