"""Verification metrics for model-observation pairs.

All functions take observations and model values as matching one-dimensional
arrays, drop pairs where either side is missing, and return plain floats. The
sign convention for bias is **model minus observation**: a positive bias means
the model is too warm, too windy, too wet.

.. warning::

   :func:`kling_gupta_efficiency` divides by the mean of the observations and
   is therefore only meaningful for strictly positive quantities such as wind
   speed, precipitation or irradiance. Applied to temperature in degrees
   Celsius it is meaningless -- the observed mean passes through zero, and the
   metric explodes. Use it for wind, not for temperature, or convert to kelvin
   first and accept that the ratio term then says very little.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "METRIC_NAMES",
    "bias",
    "correlation",
    "kling_gupta_efficiency",
    "mean_absolute_error",
    "root_mean_square_error",
    "standard_deviation_ratio",
    "verification_metrics",
]

#: Metrics returned by :func:`verification_metrics`, in reporting order.
METRIC_NAMES: tuple[str, ...] = (
    "n_pairs",
    "bias",
    "rmse",
    "mae",
    "correlation",
    "obs_mean",
    "model_mean",
    "obs_std",
    "model_std",
    "std_ratio",
)


def _paired(observed, modelled) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite, aligned pairs of two arrays."""
    obs = np.asarray(observed, dtype=float).ravel()
    mod = np.asarray(modelled, dtype=float).ravel()

    if obs.shape != mod.shape:
        raise ValueError(
            f"Observations and model values must have the same length; "
            f"got {obs.size} and {mod.size}"
        )

    valid = np.isfinite(obs) & np.isfinite(mod)
    return obs[valid], mod[valid]


def bias(observed, modelled) -> float:
    """Mean error, model minus observation.

    Returns
    -------
    float
        Mean signed error in the input units, or ``nan`` if no valid pairs
        remain.
    """
    obs, mod = _paired(observed, modelled)
    return float(np.mean(mod - obs)) if obs.size else float("nan")


def root_mean_square_error(observed, modelled) -> float:
    """Root mean square error.

    Returns
    -------
    float
        RMSE in the input units, or ``nan`` if no valid pairs remain.
    """
    obs, mod = _paired(observed, modelled)
    return float(np.sqrt(np.mean((mod - obs) ** 2))) if obs.size else float("nan")


def mean_absolute_error(observed, modelled) -> float:
    """Mean absolute error.

    Returns
    -------
    float
        MAE in the input units, or ``nan`` if no valid pairs remain.
    """
    obs, mod = _paired(observed, modelled)
    return float(np.mean(np.abs(mod - obs))) if obs.size else float("nan")


def correlation(observed, modelled) -> float:
    """Pearson correlation coefficient.

    Returns
    -------
    float
        Correlation in [-1, 1], or ``nan`` if fewer than two valid pairs remain
        or either series is constant.
    """
    obs, mod = _paired(observed, modelled)
    if obs.size < 2 or np.std(obs) == 0 or np.std(mod) == 0:
        return float("nan")
    return float(np.corrcoef(obs, mod)[0, 1])


def standard_deviation_ratio(observed, modelled) -> float:
    """Ratio of model to observed standard deviation.

    Values below 1 indicate a model that under-represents variability -- the
    characteristic signature of a coarse product in complex terrain.

    Returns
    -------
    float
        Ratio, or ``nan`` if the observed spread is zero.
    """
    obs, mod = _paired(observed, modelled)
    if obs.size < 2:
        return float("nan")
    obs_std = float(np.std(obs))
    return float(np.std(mod) / obs_std) if obs_std > 0 else float("nan")


def kling_gupta_efficiency(observed, modelled) -> float:
    """Kling-Gupta efficiency, decomposing correlation, spread and mean.

    ``KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)`` with
    ``alpha`` the ratio of standard deviations and ``beta`` the ratio of means.
    A perfect model scores 1.

    See the module warning: the ``beta`` term makes this unsuitable for
    temperature in degrees Celsius.

    Returns
    -------
    float
        KGE, or ``nan`` if it cannot be computed.
    """
    obs, mod = _paired(observed, modelled)
    if obs.size < 2:
        return float("nan")

    obs_mean = float(np.mean(obs))
    obs_std = float(np.std(obs))
    if obs_mean == 0 or obs_std == 0:
        return float("nan")

    r = correlation(obs, mod)
    if not np.isfinite(r):
        return float("nan")

    alpha = float(np.std(mod)) / obs_std
    beta = float(np.mean(mod)) / obs_mean
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def verification_metrics(observed, modelled) -> dict[str, float]:
    """Compute the standard set of metrics for one model-observation pair.

    Parameters
    ----------
    observed, modelled
        Matching series. Pairs with a missing value on either side are dropped.

    Returns
    -------
    dict
        Metrics keyed by :data:`METRIC_NAMES`. ``n_pairs`` records how many
        valid pairs contributed, which is essential context in complex terrain
        where station records are patchy.
    """
    obs, mod = _paired(observed, modelled)

    if obs.size == 0:
        return {name: (0.0 if name == "n_pairs" else float("nan")) for name in METRIC_NAMES}

    return {
        "n_pairs": float(obs.size),
        "bias": bias(obs, mod),
        "rmse": root_mean_square_error(obs, mod),
        "mae": mean_absolute_error(obs, mod),
        "correlation": correlation(obs, mod),
        "obs_mean": float(np.mean(obs)),
        "model_mean": float(np.mean(mod)),
        "obs_std": float(np.std(obs)),
        "model_std": float(np.std(mod)),
        "std_ratio": standard_deviation_ratio(obs, mod),
    }
