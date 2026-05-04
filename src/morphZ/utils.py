import numpy as np
from scipy.signal import correlate
from statsmodels.tsa.stattools import acf

def compute_rho_f2_0_via_statsmodels(f2_values, nlags=None):
    """
    Estimate integrated autocorrelation time using ``statsmodels.acf``.
    
    The integrated autocorrelation time is defined as:
        ``tau = 1 + 2 * sum_{lag>=1} acf(lag)``

    where the sum stops at the first negative value.
    
    Args:
        f2_values: One-dimensional values computed from posterior samples.
        nlags: Maximum number of lags to compute. If None, uses
            ``len(f2_values) - 1``.
    
    Returns:
        float: Estimated integrated autocorrelation time.
    """
    if nlags is None:
        nlags = len(f2_values) - 1
    # Compute the autocorrelation function using fft for speed.
    acf_values = acf(f2_values, nlags=nlags, fft=True)
    # Start with lag 0 which is 1; then sum positive autocorrelations until the first negative value.
    tau = 1.0
    for lag in range(1, len(acf_values)):
        if acf_values[lag] > 0:
            tau += 2 * acf_values[lag]
        else:
            break
    return tau

def compute_rho_f2_0_via_correlate(f2_values):
    """
    Estimate integrated autocorrelation time using ``scipy.signal.correlate``.
    
    The integrated autocorrelation time is defined as:
        ``tau = 1 + 2 * sum_{lag>=1} acf(lag)``

    where the sum stops at the first negative value.
    
    Args:
        f2_values: One-dimensional values computed from posterior samples.
    
    Returns:
        float: Estimated integrated autocorrelation time.
    """
    x = f2_values - np.mean(f2_values)
    n = x.size
    # Full correlation has length 2*n - 1
    corr = correlate(x, x, mode='full')
    # The 'center' index for lag=0:
    mid = n - 1
    # Keep only nonnegative lags: corr[mid:] => lags 0,1,...,n-1
    corr = corr[mid:]
    # Normalize so corr[0] = 1
    corr /= corr[0]

    # Sum the positive portion of corr
    # (some strategies sum all lags until correlation first becomes negative).
    sum_pos = 0.0
    for val in corr[1:]:  # skip lag0 because it's 1
        if val > 0:
            sum_pos += val
        else:
            break

    # Integrated autocorr time approx: tau = 1 + 2 * sum_{k>0} corr(k)
    tau = 1.0 + 2.0 * sum_pos
    return tau

def log_plus(x,y):
    """
    Compute ``log(exp(x) + exp(y))`` in a numerically stable way.

    Args:
        x: First log-space scalar.
        y: Second log-space scalar.

    Returns:
        float: Log-space sum of the two inputs.
    """
    if x > y:
      summ = x + np.log(1+np.exp(y-x))
    else:
        summ = y + np.log(1+np.exp(x-y))
    return summ

def log_sum(vec): 
    """
    Compute ``log(sum(exp(vec)))`` for a vector of log-space values.

    Args:
        vec: Sequence of log-space scalar values.

    Returns:
        float: Stable log-space sum.
    """
    r = -np.inf
    for i in range(len(vec)):
       r =log_plus(r, vec[i])
    return r
def error_bound_from_oscillation(x):
    """
    Estimate a fixed-point error bound from oscillatory iterates.

    The first 20 percent of values are discarded as transient behavior, then
    the bound is the remaining range ``max(x) - min(x)``.

    Args:
        x: Sequence of fixed-point iterates.

    Returns:
        float: Oscillation range after discarding the initial transient.

    Raises:
        ValueError: If fewer than two post-transient iterates remain.
    """
    x = np.array(x, dtype=float)
    x = x[int(0.2*len(x)):] 
    if len(x) < 2:
        raise ValueError("Need at least two iterates to compute oscillation bounds.")
    lower = min(x)
    upper = max(x)
    return (upper - lower)
