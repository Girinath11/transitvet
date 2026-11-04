"""Follow-up planner: can a small telescope at your site see the next dips of a candidate?

For each predicted dip in the next N nights it checks
  1. timing: the whole dip, plus the timing uncertainty and 30 min of baseline on each side, falls in
     astronomical night (Sun below -12 degrees) with the star above 30 degrees altitude;
  2. detectability: the expected signal-to-noise of the dip for your telescope, from a simple noise budget
     (photon noise, atmospheric scintillation by Young's formula, and a 500 ppm floor for the ground
     systematics a small telescope usually has). SNR >= 7 'clear', 4-7 'marginal', below 4 'too faint'.
These are planning estimates, not guarantees: clouds, the moon, comparison stars and your camera matter.
"""
import numpy as np

V_ZERO_PHOTONS = 8.8e5      # photons / s / cm^2 for a V = 0 star across the V band (~1000 photons/s/cm^2/A x 880 A)
SYSTEM_EFFICIENCY = 0.3     # optics + filter + detector, typical small-telescope CCD/CMOS
FLOOR_PPM = 500.0           # ground-based systematics floor on a dip average
MIN_ALT, SUN_ALT = 30.0, -12.0


def expected_snr(depth_ppm, duration_h, vmag, aperture_cm, airmass=1.3, site_height_m=400.0):
    """Expected SNR of one dip: depth / noise of (in-dip mean minus out-of-dip mean, equal time)."""
    if not (np.isfinite(depth_ppm) and np.isfinite(vmag) and depth_ppm > 0):
        return np.nan
    t = duration_h * 3600.0
    area = np.pi * (aperture_cm / 2) ** 2
    n_phot = V_ZERO_PHOTONS * 10 ** (-0.4 * vmag) * area * SYSTEM_EFFICIENCY * t
    s_phot = 1 / np.sqrt(n_phot)
    s_scint = 0.09 * aperture_cm ** (-2 / 3) * airmass ** 1.75 * np.exp(-site_height_m / 8000.0) / np.sqrt(2 * t)
    s = np.sqrt(s_phot ** 2 + s_scint ** 2 + (FLOOR_PPM * 1e-6) ** 2) * np.sqrt(2)   # in-dip vs baseline
    return float(depth_ppm * 1e-6 / s)


def rating(snr):
    if not np.isfinite(snr):
        return 'unknown'
    return 'clear' if snr >= 7 else 'marginal' if snr >= 4 else 'too faint'


def plan(ra_deg, dec_deg, period, t0_bjd, duration_d, timing_err_h_per_orbit, depth_ppm, vmag,
         lat, lon, height_m=400.0, aperture_cm=20.0, nights=14, tz_hours=5.5, now=None, max_events=8):
    """List the dips in the next `nights` nights that are observable from the site.
    t0_bjd: a dip mid-time in BJD (TDB). Returns a list of dicts (local times use tz_hours)."""
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun
    from astropy.utils import iers
    iers.conf.auto_download = False            # works offline; precision is ample for planning
    now = now or Time.now()
    site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height_m * u.m)
    star = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    k0 = int(np.ceil((now.jd - t0_bjd) / period))
    k1 = int(np.floor((now.jd + nights - t0_bjd) / period))
    out = []
    for k in range(k0, k1 + 1):
        mid = t0_bjd + k * period
        err_d = abs(k) * timing_err_h_per_orbit / 24.0
        half = duration_d / 2 + err_d + 30 / 1440.0
        if err_d > duration_d * 2:              # timing too uncertain to plan
            out.append(dict(mid_utc=Time(mid, format='jd', scale='tdb').utc.iso[:16], observable=False,
                            reason=f'timing uncertain by ±{err_d * 24:.1f} h'))
            continue
        ts = Time(np.linspace(mid - half, mid + half, 25), format='jd', scale='tdb')
        frame = AltAz(obstime=ts, location=site)
        alt = star.transform_to(frame).alt.deg
        sun = get_sun(ts).transform_to(frame).alt.deg
        ok = bool((alt >= MIN_ALT).all() and (sun <= SUN_ALT).all())
        am = float(1 / np.sin(np.radians(max(alt.min(), 10))))
        snr = expected_snr(depth_ppm, duration_d * 24, vmag, aperture_cm, am, height_m)
        reason = '' if ok else ('star too low' if (alt < MIN_ALT).any() and (sun <= SUN_ALT).all()
                                else 'daytime or twilight' if (sun > SUN_ALT).any() and (alt >= MIN_ALT).all()
                                else 'star too low and daytime')
        loc = Time(mid, format='jd', scale='tdb').utc + tz_hours * u.hour
        out.append(dict(mid_utc=Time(mid, format='jd', scale='tdb').utc.iso[:16], mid_local=loc.iso[:16],
                        start_local=(loc - half * u.day).iso[11:16], end_local=(loc + half * u.day).iso[11:16],
                        min_altitude=round(float(alt.min()), 0), observable=ok, snr=round(snr, 1) if np.isfinite(snr) else None,
                        rating=rating(snr), reason=reason, timing_err_min=round(err_d * 1440, 0)))
        if sum(e.get('observable', False) for e in out) >= max_events:
            break
    return out


def never_visible(dec_deg, lat):
    """True if the star never gets above MIN_ALT at this latitude."""
    return 90 - abs(lat - dec_deg) < MIN_ALT
