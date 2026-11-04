"""Transit model fit (Mandel & Agol 2002 shape via the batman package) for one signal.

Fits, on the detrended light curve near the dips: period, mid-time t0, size ratio Rp/Rs,
orbit size a/Rs and impact parameter b (0 = crosses the star's centre, near 1 = grazes the edge).
Limb darkening is fixed to a generic quadratic law (u1=0.4, u2=0.25); 30-min exposures are
supersampled. Errors come from the fit's covariance matrix (no MCMC), so treat them as rough.

What it gives the user:
 - a sharper period and t0 (better next-dip predictions)
 - planet size with an error bar, if the star's radius is known
 - shape: a grazing / V-shaped dip (b + Rp/Rs > 1) is a common eclipsing-binary sign
 - the star's density implied by the dip vs the catalogue density: a large mismatch suggests
   the dip is not a planet on this star (or the period is wrong)
"""
import numpy as np

U = [0.4, 0.25]
G_CGS = 6.674e-8
R_SUN_CM, M_SUN_G = 6.957e10, 1.989e33


def _model(params, t, period, exp_time):
    import batman
    t0, per, rp, ars, b = params
    pm = batman.TransitParams()
    pm.t0, pm.per, pm.rp, pm.a = t0, per, rp, ars
    pm.inc = float(np.degrees(np.arccos(np.clip(b / ars, 0, 1))))
    pm.ecc, pm.w, pm.u, pm.limb_dark = 0.0, 90.0, U, 'quadratic'
    m = batman.TransitModel(pm, t, supersample_factor=7, exp_time=exp_time)
    return m.light_curve(pm)


def fit_transit(t, f, period, t0, duration, depth_ppm, star_rad=np.nan, star_logg=np.nan, mission='kepler'):
    from scipy.optimize import least_squares
    dt = (t - t0 + 0.5 * period) % period - 0.5 * period
    near = np.abs(dt) < 3 * duration
    if near.sum() < 30:
        return None
    tt, ff = t[near], f[near]
    exp_time = 0.0208                                   # 30-min points (Kepler long cadence; TESS binned)
    rp0 = float(np.sqrt(max(depth_ppm, 10) * 1e-6))
    # a/Rs from duration for a central transit: T ~ P/pi / (a/Rs)
    ars0 = float(np.clip(period / (np.pi * duration), 1.5, 500))
    x0 = [t0, period, rp0, ars0, 0.3]
    lo = [t0 - 0.5 * duration, period * (1 - 1e-3), 1e-3, 1.2, 0.0]
    hi = [t0 + 0.5 * duration, period * (1 + 1e-3), 0.9, 1000.0, 1.5]
    x0 = list(np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9))
    res_fn = lambda p: _model(p, tt, period, exp_time) - ff
    try:
        r = least_squares(res_fn, x0, bounds=(lo, hi), x_scale='jac', max_nfev=400)
    except Exception:
        return None
    resid = r.fun; dof = max(resid.size - 5, 1)
    s2 = float(np.sum(resid ** 2) / dof)
    try:
        cov = np.linalg.pinv(r.jac.T @ r.jac) * s2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        err = np.full(5, np.nan)
    t0f, perf, rp, ars, b = r.x
    out = dict(fit_ok=bool(r.success), period=float(perf), period_err=float(err[1]), t0=float(t0f), t0_err=float(err[0]),
               rp_rs=float(rp), rp_rs_err=float(err[2]), a_rs=float(ars), a_rs_err=float(err[3]), b=float(b), b_err=float(err[4]),
               grazing=bool(b + rp > 1.0), chi2_red=float(s2 / max(np.var(ff[np.abs((tt - t0f + 0.5 * perf) % perf - 0.5 * perf) > duration]), 1e-12)))
    if np.isfinite(star_rad):
        out['planet_radius_re'] = float(rp * star_rad * 109.1)
        out['planet_radius_err'] = float(err[2] * star_rad * 109.1)
    # stellar density from the fit (circular orbit): rho = 3 pi (a/Rs)^3 / (G P^2)
    p_s = perf * 86400.0
    out['rho_star_fit'] = float(3 * np.pi * ars ** 3 / (G_CGS * p_s ** 2))
    if np.isfinite(star_rad) and np.isfinite(star_logg):
        m = 10 ** star_logg * (star_rad * R_SUN_CM) ** 2 / G_CGS
        out['rho_star_cat'] = float(m / (4 / 3 * np.pi * (star_rad * R_SUN_CM) ** 3))
    return out
