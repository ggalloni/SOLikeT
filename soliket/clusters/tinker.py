"""
.. module:: tinker

Parameters and functions used internally by the cluster likelihood for the Tinker profile.

"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

# Tinker stuff

tinker_data = np.transpose(
    [
        [float(x) for x in line.split()]
        for line in """200 0.186 1.47 2.57 1.19
                               300 0.200 1.52 2.25 1.27
                               400 0.212 1.56 2.05 1.34
                               600 0.218 1.61 1.87 1.45
                               800 0.248 1.87 1.59 1.58
                              1200 0.255 2.13 1.51 1.80
                              1600 0.260 2.30 1.46 1.97
                              2400 0.260 2.53 1.44 2.24
                              3200 0.260 2.66 1.41 2.44""".split("\n")
    ]
)

tinker_splines: list[iuSpline] | None = None


def tinker_params_spline(
    delta: float, z: float | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global tinker_splines
    if tinker_splines is None:
        tinker_splines = []
        D, data = np.log(tinker_data[0]), tinker_data[1:]
        for y in data:
            # Extend to large Delta
            p = np.polyfit(D[-2:], y[-2:], 1)
            x = np.hstack((D, D[-1] + 3.0))
            y = np.hstack((y, np.polyval(p, x[-1])))
            tinker_splines.append(iuSpline(x, y, k=2))
    A0, a0, b0, c0 = (ts(np.log(delta)) for ts in tinker_splines)
    if z is None:
        return A0, a0, b0, c0

    z = np.asarray(z)
    A = A0 * (1 + z) ** -0.14
    a = a0 * (1 + z) ** -0.06
    alpha = 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))
    b = b0 * (1 + z) ** -alpha
    c = np.zeros(np.shape(z)) + c0
    return A, a, b, c


def tinker_params_analytic(
    delta: float | np.ndarray, z: float | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = None
    if np.asarray(delta).ndim == 0:  # scalar delta.
        A0, a0, b0, c0 = (p[0] for p in tinker_params(np.array([delta]), z=None))
        if z is not None:
            if delta < 75.0:
                alpha = 1.0
            else:
                alpha = 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))

    else:
        log_delta = np.log10(delta)
        A0 = 0.1 * log_delta - 0.05
        a0 = 1.43 + (log_delta - 2.3) ** 1.5
        b0 = 1.0 + (log_delta - 1.6) ** (-1.5)
        c0 = log_delta - 2.35
        A0[delta > 1600] = 0.26
        a0[log_delta < 2.3] = 1.43
        b0[log_delta < 1.6] = 1.0
        c0[c0 < 0] = 0.0
        c0 = 1.2 + c0**1.6
    if z is None:
        return A0, a0, b0, c0
    A = A0 * (1 + z) ** -0.14
    a = a0 * (1 + z) ** -0.06
    if alpha is None:
        alpha = 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))
        alpha[delta < 75.0] = 1.0
    b = b0 * (1 + z) ** -alpha
    c = np.zeros(np.shape(z)) + c0
    return A, a, b, c


tinker_params = tinker_params_spline


def tinker_f(
    sigma: float | np.ndarray, params: tuple[float, float, float, float]
) -> float | np.ndarray:
    A, a, b, c = params
    return A * ((sigma / b) ** -a + 1) * np.exp(-c / sigma**2)


# Sigma-evaluation, and top-hat functions.


def radius_from_mass(
    M: float | np.ndarray, rho: float | np.ndarray
) -> float | np.ndarray:
    """
    Convert mass M to radius R assuming density rho.
    """
    return (3.0 * M / (4.0 * np.pi * rho)) ** (1 / 3.0)


def top_hatf(kR: float | np.ndarray) -> float | np.ndarray:
    """
    Returns the Fourier transform of the spherical top-hat function
    evaluated at a given k*R.
    Notes:
    -------
    * This is called many times and costs a lot of runtime.
    * For small values, use Taylor series.
    """
    out = np.nan_to_num(3 * (np.sin(kR) - kR * np.cos(kR))) / (kR**3)
    return out


def sigma_sq_integral(
    R_grid: np.ndarray, power_spt: np.ndarray, k_val: np.ndarray
) -> np.ndarray:
    """
    Determines the sigma^2 parameter over the m-z grid by integrating over k.
    Notes:
    -------
    * Fastest python solution I have found for this. There is probably a
      smarter way using numpy arrays.
    """
    to_integ = np.array(
        [
            top_hatf(R_grid * k) ** 2
            * np.tile(
                power_spt[:, i],
                (R_grid.shape[0], 1),
            )
            * k**2
            for k, i in zip(k_val, np.arange(len(k_val)))
        ]
    )

    return simpson(to_integ / (2 * np.pi**2), x=k_val, axis=0)


def dn_dlogM(
    M: np.ndarray,
    z: np.ndarray,
    rho: np.ndarray,
    delta: float | np.ndarray,
    k: np.ndarray,
    P: np.ndarray,
    comoving: bool = False,
) -> np.ndarray:
    """
    M      is  (nM)  or  (nM, nz)
    z      is  (nz)
    rho    is  (nz)
    delta  is  (nz)  or  scalar
    k      is  (nk)
    P      is  (nz,nk)

    Somewhat awkwardly, k and P are comoving.  rho really isn't.
    return is  (nM,nz)
    """

    if M.ndim == 1:
        M = M[:, None]
    # Radius associated to mass, co-moving
    R = radius_from_mass(M, rho)
    if not comoving:  # if you do this make sure rho still has shape of z.
        R = R * np.transpose(1 + z)
    # Fluctuations on those scales (P and k are comoving)
    sigma = sigma_sq_integral(R, P, k) ** 0.5
    # d log(sigma^-1)
    # gradient is broken.
    if R.shape[-1] == 1:
        dlogs = -np.gradient(np.log(sigma[..., 0]))[:, None]
    else:
        dlogs = -np.gradient(np.log(sigma))[0]
    # Evaluate Tinker mass function.
    tp = tinker_params(delta, z)
    tf = tinker_f(sigma, tp)
    # dM; compute as M * dlogM since it is likely log-spaced.
    if M.shape[-1] == 1:
        dM = np.gradient(np.log(M[:, 0]))[:, None] * M
    else:
        dM = np.gradient(np.log(M))[0] * M
    # Return dn / dlogM
    return tf * rho * dlogs / dM
