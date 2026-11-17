"""Hear a signal: the folded light curve played as a tone. Brightness sets the pitch, so each dip sounds like a short
drop in pitch, once per orbit. Uses the model's own whole-orbit view (median 0, deepest bin -1)."""
import io
import wave

import numpy as np

RATE = 22050


def tone(view, orbits=3, seconds_per_orbit=1.6, base=523.25, octaves=0.9):
    v = np.nan_to_num(np.asarray(view, float), nan=0.0)
    v = np.clip(v, -1.5, 0.5)
    n = int(RATE * seconds_per_orbit)
    one = np.interp(np.linspace(0, len(v) - 1, n), np.arange(len(v)), v)
    # put the dip in the middle of each orbit, as in the plots
    curve = np.tile(one, orbits)
    freq = base * 2 ** (octaves * curve)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    y = 0.6 * np.sin(phase) + 0.15 * np.sin(2 * phase)
    fade = np.minimum(1, np.minimum(np.arange(y.size), np.arange(y.size)[::-1]) / (0.03 * RATE))
    y *= fade
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((y * 32767 * 0.8).astype('<i2').tobytes())
    return buf.getvalue()
