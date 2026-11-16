/* TransitVet live view: one animated scene per analysis step, drawn from the star's real data (window.D). */
(function () {
'use strict';
const D = window.D;
const C = { bg: '#070a14', panel: '#0b1020', line: 'rgba(214,222,240,.12)', grid: 'rgba(214,222,240,.07)', ink: '#ece9e1', mute: '#8f93a0',
  ice: '#5ee7ff', star: '#a78bfa', pink: '#ff8bd1', good: '#66d19e', bad: '#f07b6c', warn: '#ffc861' };
const SEG = [C.ice, C.star, C.pink, '#7cf2b8', '#ffd28a'];
const RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = s => document.querySelector(s);
const cv = $('#cv'), ctx = cv.getContext('2d');
let W = 0, H = D.h || 420;

/* ---------- helpers ---------- */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, u) => a + (b - a) * u;
const ease = u => { u = clamp(u, 0, 1); return u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; };
const seg = (t, a, b) => clamp((t - a) / (b - a), 0, 1);
const fmt = (v, d = 0) => v === null || v === undefined || !isFinite(v) ? '—' : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
function pct(arr, q) { const a = arr.filter(isFinite).slice().sort((x, y) => x - y); if (!a.length) return 0; return a[Math.floor(clamp(q, 0, 1) * (a.length - 1))]; }
function median(a) { return pct(a, .5); }
function mad(a) { const m = median(a); return 1.4826 * median(a.map(v => Math.abs(v - m))); }
function sc(lo, hi, a, b) { const k = (b - a) / ((hi - lo) || 1); return v => a + (v - lo) * k; }
function logsc(lo, hi, a, b) { const l0 = Math.log(lo), l1 = Math.log(hi); return v => a + (Math.log(v) - l0) / (l1 - l0) * (b - a); }
function txt(s, x, y, o = {}) {
  ctx.font = `${o.w || 500} ${o.s || 12}px ${o.f || '"Manrope", system-ui, sans-serif'}`;
  ctx.fillStyle = o.c || C.mute; ctx.textAlign = o.a || 'left'; ctx.textBaseline = o.b || 'alphabetic';
  ctx.globalAlpha = o.al === undefined ? 1 : o.al; ctx.fillText(s, x, y); ctx.globalAlpha = 1;
}
function fitText(s, w, size, weight) { ctx.font = `${weight || 500} ${size}px "Manrope", system-ui, sans-serif`; if (ctx.measureText(s).width <= w) return s; while (s.length > 4 && ctx.measureText(s + '…').width > w) s = s.slice(0, -1); return s + '…'; }
const mono = (s, x, y, o = {}) => txt(s, x, y, Object.assign({ f: '"JetBrains Mono", ui-monospace, monospace', s: 11.5 }, o));
function panel(x, y, w, h, label) {
  ctx.fillStyle = C.panel; ctx.strokeStyle = C.line; ctx.lineWidth = 1;
  rr(x, y, w, h, 8); ctx.fill(); ctx.stroke();
  if (label) txt(label, x + 10, y + 16, { s: 11.5, c: C.mute, w: 600 });
}
function rr(x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
function dots(xs, ys, fx, fy, col, r = 1.3, al = .7, n) {
  ctx.fillStyle = col; ctx.globalAlpha = al; n = n === undefined ? xs.length : n;
  for (let i = 0; i < n; i++) { const X = fx(xs[i]), Y = fy(ys[i]); if (isFinite(X) && isFinite(Y)) ctx.fillRect(X - r / 2, Y - r / 2, r, r); }
  ctx.globalAlpha = 1;
}
function path(pts, col, lw = 1.5, al = 1) {
  ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.globalAlpha = al; ctx.beginPath(); let on = false;
  for (const [x, y] of pts) { if (!isFinite(x) || !isFinite(y)) { on = false; continue; } if (on) ctx.lineTo(x, y); else { ctx.moveTo(x, y); on = true; } }
  ctx.stroke(); ctx.globalAlpha = 1;
}
function hline(y, x0, x1, col = C.line, dash) { ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
function vline(x, y0, y1, col = C.line, dash, lw = 1) { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash || []); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function clip(x, y, w, h, fn) { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); fn(); ctx.restore(); }
function binMean(xs, ys, lo, hi, nb) {
  const s = new Float64Array(nb), n = new Float64Array(nb);
  for (let i = 0; i < xs.length; i++) { const b = Math.floor((xs[i] - lo) / (hi - lo) * nb); if (b >= 0 && b < nb && isFinite(ys[i])) { s[b] += ys[i]; n[b]++; } }
  return Array.from(s, (v, i) => n[i] ? v / n[i] : NaN);
}
function heat(v, vmax) {     // signed: toward planet = cyan, toward false alarm = coral
  const u = clamp(Math.abs(v) / (vmax || 1), 0, 1);
  return v >= 0 ? `rgba(94,231,255,${.08 + .92 * u})` : `rgba(240,123,108,${.08 + .92 * u})`;
}
function gauge(x, y, w, v, col, label, sub) {
  ctx.fillStyle = 'rgba(214,222,240,.10)'; rr(x, y, w, 8, 4); ctx.fill();
  ctx.fillStyle = col; rr(x, y, Math.max(2, w * clamp(v, 0, 1)), 8, 4); ctx.fill();
  vline(x + w / 2, y - 3, y + 11, 'rgba(214,222,240,.45)');
  if (label) txt(label, x, y - 6, { s: 11.5, c: C.ink, w: 600 });
  if (sub) mono(sub, x + w, y - 6, { a: 'right', c: col });
}
function arrow(x0, y0, x1, y1, col, al = 1) {
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.globalAlpha = al; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  const a = Math.atan2(y1 - y0, x1 - x0); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 7 * Math.cos(a - .4), y1 - 7 * Math.sin(a - .4)); ctx.lineTo(x1 - 7 * Math.cos(a + .4), y1 - 7 * Math.sin(a + .4)); ctx.fill(); ctx.globalAlpha = 1;
}
const NAMES = { gb: 'Gradient boosting', mamba: 'Mamba-2', tf: 'Transformer', combined: 'Combined' };
const MCOL = { gb: C.warn, mamba: C.ice, tf: C.star };
const FEAT = { depth_ppm: 'dip depth (ppm)', snr: 'signal-to-noise', odd_even_sigma: 'odd/even difference (σ)', secondary_ppm: 'dip at half orbit (ppm)',
  secondary_sigma: 'dip at half orbit (σ)', shape_ratio: 'dip shape (U vs V)', n_transits: 'dips seen', tce_period: 'period (days)', tce_duration: 'duration (days)',
  local_cover_biweight: 'close-up data coverage', cen_sigma: 'light-position shift (σ)', cen_offset_arcsec: 'source offset (″)', rp_est_re: 'size (Earth radii)',
  contratio: 'light from neighbours' };

/* ---------- scene: download and join ---------- */
function sceneLoad() {
  const S = D.segs, all = [];
  S.forEach(s => s.y.forEach(v => all.push(v)));
  const t0 = Math.min(...S.map(s => s.t[0])), t1 = Math.max(...S.map(s => s.t[s.t.length - 1]));
  const loR = pct(all, .005), hiR = pct(all, .995);
  const norm = []; S.forEach(s => s.y.forEach(v => norm.push(v - s.off)));
  const loN = pct(norm, .005), hiN = pct(norm, .995);
  return {
    dur: 5.2,
    say: [[0, `NASA's MAST archive sends this star's brightness measurements, one piece per ${D.mission === 'kepler' ? 'Kepler quarter' : 'TESS sector'}.`],
          [2.8, 'Every piece sits at its own brightness level (the star lands on different camera pixels). Each piece is divided by its own median so they line up.'],
          [4.4, `Joined into one light curve: ${fmt(D.n_points)} measurements over ${fmt(t1 - t0)} days.`]],
    nums: [[0, 'mission', D.mission.toUpperCase()], [.4, 'pieces', D.segs.length], [1.2, 'measurements', fmt(D.n_points)],
           [1.8, 'time span', fmt(t1 - t0) + ' d'], [2.2, 'one point every', fmt(D.cadence_min, 1) + ' min'], [2.6, 'gaps over 1 day', D.gaps]],
    draw(t) {
      const x = 14, y = 26, w = W - 28, h = H - 44;
      panel(x, y - 12, w, h + 18);
      const u = ease(seg(t, 0, 2.6)), s = ease(seg(t, 2.9, 4.3));
      const fx = sc(t0, t1, x + 46, x + w - 12), fy = sc(lerp(loR, loN, s) * 1.15, lerp(hiR, hiN, s) * 1.15, y + h - 18, y + 14);
      const tc = lerp(t0, t1, u);
      [0, .5, 1].forEach(q => { const v = lerp(lerp(loR, loN, s), lerp(hiR, hiN, s), q); hline(fy(v), x + 46, x + w - 12, C.grid); mono(fmt(v) + '', x + 42, fy(v) + 4, { a: 'right', s: 10 }); });
      mono('ppm', x + 8, y + 8, { s: 10 });
      let cur = -1;
      S.forEach((g, k) => {
        let n = 0; while (n < g.t.length && g.t[n] <= tc) n++;
        if (n > 0 && g.t[0] <= tc) cur = k;
        dots(g.t, g.y.map(v => v - s * g.off), fx, fy, SEG[k % SEG.length], 1.4, .65, n);
      });
      for (let d = Math.ceil(t0 / 100) * 100; d <= t1; d += t1 - t0 > 600 ? 200 : 100) mono(fmt(d), fx(d), y + h + 2, { a: 'center', s: 10 });
      if (u < 1) { vline(fx(tc), y + 4, y + h - 14, C.ice, [3, 3]); if (cur >= 0) txt(`piece ${cur + 1} of ${S.length}`, fx(tc) + 6, y + 20, { c: C.ice, s: 12, w: 600 }); }
      if (s > 0 && s < 1) S.forEach((g, k) => { if (Math.abs(g.off) > 1) arrow(fx(g.t[0]) + 8, fy(g.y[0] - 0) - 2, fx(g.t[0]) + 8, fy(g.y[0] - g.off) - 2, SEG[k % SEG.length], .7 * (1 - s)); });
      txt('time (days)', x + w - 12, y + h + 2, { a: 'right', s: 10.5 });
    }
  };
}

/* ---------- scene: flatten ---------- */
function sceneFlatten() {
  const t = D.t, r = D.r, tr = D.trend, fl = D.f, n = t.length;
  const t0 = t[0], t1 = t[n - 1];
  const loR = pct(r, .003), hiR = pct(r, .997), loF = pct(fl, .003), hiF = pct(fl, .997);
  // zoom window: around the deepest flattened point
  let k = 0; for (let i = 1; i < n; i++) if (fl[i] < fl[k]) k = i;
  const zc = t[k], zw = 3.5;
  const zi = []; for (let i = 0; i < n; i++) if (Math.abs(t[i] - zc) < zw) zi.push(i);
  const zr = zi.map(i => r[i]), zf = zi.map(i => fl[i]), zt = zi.map(i => tr[i]);
  const zlo = Math.min(pct(zr, 0), pct(zf, 0)), zhi = Math.max(pct(zr, 1), pct(zf, 1), pct(zt, 1));
  return {
    dur: 7,
    say: [[0, "The star's own slow ups and downs (spots, rotation, instrument drift) would hide small dips. A sliding 1-day window follows them."],
          [1.6, 'Inside the window, a robust average (biweight) ignores short dips, so the trend line passes under a transit instead of following it.'],
          [3.6, "Dividing by the trend leaves a flat line: what's left is noise plus any short dips."],
          [5.2, `Upward spikes (cosmic rays, glints) are removed: ${fmt(D.removed)} points. Dips are never removed here.`]],
    nums: [[0, 'method', 'biweight'], [.3, 'window', '1 day'], [1, 'scatter before', fmt(D.sd_before) + ' ppm'], [4.6, 'scatter after', fmt(D.sd_after) + ' ppm'],
           [5.4, 'points removed', fmt(D.removed)], [5.8, 'points kept', fmt(D.n_kept)]],
    draw(T) {
      const x = 14, w = W - 28, h1 = Math.round((H - 40) * .52), y1 = 14, y2 = y1 + h1 + 14, h2 = H - y2 - 22;
      const u = ease(seg(T, 0, 3.4)), m = ease(seg(T, 3.6, 5.0)), sp = seg(T, 5.2, 6.4);
      // top: whole light curve
      panel(x, y1, w, h1, 'whole light curve');
      const fx = sc(t0, t1, x + 10, x + w - 10), fy = sc(lerp(loR, loF, m) * 1.2, lerp(hiR, hiF, m) * 1.2, y1 + h1 - 8, y1 + 24);
      const ys = new Array(n); for (let i = 0; i < n; i++) ys[i] = lerp(r[i], fl[i], m);
      dots(t, ys, fx, fy, C.ice, 1.3, .55);
      const tc = lerp(t0, t1, u), tp = [];
      for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 1500))) { if (t[i] > tc) break; tp.push([fx(t[i]), fy(lerp(tr[i], 0, m))]); }
      path(tp, C.warn, 1.6, .95);
      if (u < 1) vline(fx(tc), y1 + 20, y1 + h1 - 6, C.warn, [3, 3]);
      if (D.rem_t && sp > 0) dots(D.rem_t, D.rem_r, fx, fy, C.bad, 2.6, (1 - sp) * (Math.sin(T * 18) * .3 + .7));
      // bottom: zoom
      panel(x, y2, w, h2, `close-up: days ${fmt(zc - zw, 1)} to ${fmt(zc + zw, 1)}`);
      const gx = sc(zc - zw, zc + zw, x + 10, x + w - 10), gy = sc(zlo - (zhi - zlo) * .15, zhi + (zhi - zlo) * .15, y2 + h2 - 8, y2 + 24);
      const wc = lerp(zc - zw + .5, zc + zw - .5, ease(seg(T, 0, 3.4)));
      if (m < 1) { ctx.fillStyle = 'rgba(255,200,97,.10)'; ctx.fillRect(gx(wc - .5), y2 + 22, gx(wc + .5) - gx(wc - .5), h2 - 28); txt('1-day window', gx(wc), y2 + 34, { a: 'center', c: C.warn, s: 11 }); }
      dots(zi.map(i => t[i]), zi.map((i, j) => lerp(zr[j], zf[j], m)), gx, gy, C.ice, 2.4, .8);
      const zp = []; zi.forEach((i, j) => { if (t[i] <= wc || m > 0) zp.push([gx(t[i]), gy(lerp(zt[j], 0, m))]); });
      path(zp, C.warn, 2, .95);
      txt('trend (yellow)', x + w - 12, y2 + 16, { a: 'right', c: C.warn, s: 11 });
    }
  };
}

/* ---------- scene: search (scan while the server works, then the real result) ---------- */
function foldAt(P, t, f, nb, t0) {
  const ph = new Float64Array(t.length);
  for (let i = 0; i < t.length; i++) { let p = ((t[i] - t0) / P) % 1; if (p < 0) p += 1; ph[i] = p >= .5 ? p - 1 : p; }
  return ph;
}
function sceneSearch() {
  const t = D.t, f = D.f, n = t.length, lo = pct(f, .002), hi = pct(f, .998), sd = mad(f) || 1;
  const T0 = t[0], T1 = t[n - 1], found = D.mode === 'found', sig = D.sig || {};
  const Pp = found ? (sig.doubled ? sig.period / 2 : sig.period) : null;     // where BLS peaked
  const prev = new Float32Array(400).fill(NaN);
  let pg = null;
  if (found && D.pg) pg = { p: D.pg.p, w: D.pg.w, max: Math.max(...D.pg.w.filter(isFinite)) };
  const dur = found ? (sig.doubled ? 9.6 : 8.2) : 9;
  function quickScore(P) {      // browser preview only: fold, 120 phase bins, depth of the lowest bin over its noise
    const nb = 120, s = new Float64Array(nb), c = new Float64Array(nb);
    for (let i = 0; i < n; i++) { let p = (t[i] / P) % 1; const b = Math.floor(p * nb); s[b] += f[i]; c[b]++; }
    let best = 0, bb = 0; for (let b = 0; b < nb; b++) if (c[b] > 2) { const v = -s[b] / c[b] / (sd / Math.sqrt(c[b])); if (v > best) { best = v; bb = b; } }
    return [best, (bb + .5) / nb * P];
  }
  const say = found ? [[0, `Search ${D.k + 1} finished on the server: ${fmt(D.n_trials)} trial orbits, each scored by how well a box-shaped dip fits. Tall spike = a repeating dip.`],
                       [2.2, `The tallest spike: an orbit of ${fmt(Pp, 4)} days. Folding the data at that orbit makes all the dips land on top of each other.`],
                       [3.9, sig.weak ? `Signal-to-noise ${fmt(sig.snr, 1)} is below 7.1, so this is too weak to trust and the search stops here.` :
                             `A box ${fmt(sig.depth, 0)} ppm deep and ${fmt(sig.dur_h, 1)} h long fits it, with signal-to-noise ${fmt(sig.snr, 1)}.`]]
                    : [[0, `Search ${D.k + 1}: the server is trying ${fmt(D.n_trials)} orbits from ${fmt(D.pmin, 1)} to ${fmt(D.pmax, 0)} days on all ${fmt(D.n_all)} points. Here your browser previews the idea on ${fmt(n)} of them.`]];
  if (found && sig.doubled) say.push([5.6, `Check: at ${fmt(Pp, 2)} d every second dip is empty (${fmt(sig.half_sigma, 0)} σ). So the real orbit is twice as long: ${fmt(sig.period, 2)} days.`]);
  if (found && !sig.weak) say.push([sig.doubled ? 7.6 : 6.0, 'These dips are set aside, and the search runs again on what is left, looking for another planet.']);
  return {
    dur, loop: !found,
    say,
    nums: found ? [[0, 'search', `${D.k + 1} of up to 3`], [.2, 'trial orbits', fmt(D.n_trials)], [2.2, 'tallest spike', fmt(Pp, 4) + ' d'], ...(sig.doubled ? [[5.6, 'true orbit (doubled)', fmt(sig.period, 4) + ' d']] : []),
                   [3.9, 'depth', fmt(sig.depth) + ' ppm'], [4.1, 'duration', fmt(sig.dur_h, 1) + ' h'], [4.3, 'signal-to-noise', fmt(sig.snr, 1)], [4.5, 'dips with data', fmt(sig.n_dips)]]
                 : [[0, 'search', `${D.k + 1} of up to 3`], [0, 'trial orbits', fmt(D.n_trials)], [0, 'orbit range', `${fmt(D.pmin, 1)}–${fmt(D.pmax, 0)} d`], [0, 'points', fmt(D.n_all)]],
    draw(T) {
      const x = 14, w = W - 28, gap = 12, hTop = Math.round((H - 30) * .58), wL = Math.round(w * .56), y1 = 12, y2 = y1 + hTop + gap, h2 = H - y2 - 20;
      const px = x + 10, pw = wL - 20, fxp = logsc(D.pmin, D.pmax, px, px + pw);
      let P, ph0 = 0, sharp = 0;
      panel(x, y1, wL, hTop, found ? 'periodogram (server, all points)' : 'trying orbits (preview in your browser)');
      if (!found) {
        const uu = (T / 9) % 1; P = Math.exp(lerp(Math.log(D.pmin), Math.log(D.pmax), uu));
        const [q, c] = quickScore(P); prev[Math.floor(uu * 399)] = q; ph0 = c;
        const mx = Math.max(8, ...Array.from(prev).filter(isFinite));
        const fy = sc(0, mx * 1.1, y1 + hTop - 18, y1 + 26), pts = [];
        prev.forEach((v, i) => pts.push([px + i / 399 * pw, isFinite(v) ? fy(v) : NaN]));
        path(pts, C.star, 1.2, .85);
        vline(px + uu * pw, y1 + 22, y1 + hTop - 16, C.ice, [3, 3]);
        sharp = clamp((q - 5) / 10, 0, 1);
        txt(`${fmt(P, 2)} d`, px + uu * pw + 5, y1 + 36, { c: C.ice, s: 12, w: 600 });
        $('#say').textContent = say[0][1] + `  Now trying ${fmt(P, 2)} d: ` + (q > 9 && q >= .8 * mx ? 'dips line up: a strong candidate orbit!' : 'dips smear out, weak match.');
      } else {
        const fy = sc(0, pg.max * 1.08, y1 + hTop - 18, y1 + 26), ud = ease(seg(T, 0, 2.0)), pts = [];
        for (let i = 0; i < pg.p.length; i++) { if (i / pg.p.length > ud) break; pts.push([fxp(pg.p[i]), fy(pg.w[i])]); }
        path(pts, C.ice, 1, .9);
        (D.others || []).forEach(o => { if (o >= D.pmin && o <= D.pmax) vline(fxp(o), y1 + 22, y1 + hTop - 16, 'rgba(167,139,250,.35)', [2, 3]); });
        const lp = T < 2.0 ? Math.exp(lerp(Math.log(D.pmin), Math.log(D.pmax), ud)) : Math.exp(lerp(Math.log(D.pmax), Math.log(Pp), ease(seg(T, 2.2, 3.7))));
        P = lp; sharp = ease(seg(T, 2.2, 3.7));
        vline(fxp(lp), y1 + 22, y1 + hTop - 16, C.warn, [3, 3]);
        if (T > 2.2) txt(`${fmt(lp, lp < 10 ? 3 : 2)} d`, fxp(lp) + (fxp(lp) > px + pw - 70 ? -5 : 5), y1 + 36, { c: C.warn, s: 12, w: 600, a: fxp(lp) > px + pw - 70 ? 'right' : 'left' });
        ph0 = sig.t0;
        if (sig.doubled && T > 5.6) P = sig.period;
      }
      [1, 10, 100].forEach(v => { if (v >= D.pmin && v <= D.pmax) mono(v + ' d', fxp(v), y1 + hTop - 4, { a: 'center', s: 10 }); });
      // fold panel
      const fxw = w - wL - gap, fx0 = x + wL + gap;
      panel(fx0, y1, fxw, hTop, found && sig.doubled && T > 5.6 ? `folded at ${fmt(sig.period / 2, 2)} d, coloured by orbit number` : `folded at ${fmt(P, P < 10 ? 3 : 2)} d`);
      const tref = found ? sig.t0 : ph0;
      const oddEven = found && sig.doubled && T > 5.6, Pu = oddEven ? sig.period / 2 : P;
      const ph = foldAt(Pu, t, f, 0, tref);
      const zoomed = found && T > 3.9 && sig.dur_d ? Math.min(.5, 4 * sig.dur_d / Pu) : .5;
      const zx = sc(-zoomed, zoomed, fx0 + 10, fx0 + fxw - 10), zy = sc(lo * 1.25, hi * 1.25, y1 + hTop - 14, y1 + 26);
      clip(fx0 + 2, y1 + 20, fxw - 4, hTop - 22, () => {
        let col = C.ice;
        if (oddEven) {   // colour by orbit number: odd vs even
          const half = sig.period / 2, xs = [[], []], ys = [[], []];
          for (let i = 0; i < n; i++) { const q = (t[i] - sig.t0) / half, e = Math.round(q), par = ((e % 2) + 2) % 2; xs[par].push(q - e); ys[par].push(f[i]); }
          dots(xs[0], ys[0], zx, zy, C.ice, 1.8, .7); dots(xs[1], ys[1], zx, zy, C.pink, 1.8, .7);
          [0, 1].forEach(par => { const bm = binMean(xs[par], ys[par], -zoomed, zoomed, 60); path(bm.map((v, i) => [zx(-zoomed + (i + .5) / 60 * 2 * zoomed), zy(v)]), par ? C.pink : C.ice, 2, .95); });
        } else {
          dots(Array.from(ph), f, zx, zy, col, 1.6, .25 + .45 * sharp);
          const bm = binMean(Array.from(ph), f, -zoomed, zoomed, 90);
          path(bm.map((v, i) => [zx(-zoomed + (i + .5) / 90 * 2 * zoomed), zy(v)]), C.warn, 1.6, .9);
        }
        if (found && T > 3.9 && !sig.weak && !oddEven) {
          const a = ease(seg(T, 3.9, 4.6)), hd = sig.dur_d / Pu / 2, dep = -sig.depth * a;
          path([[zx(-zoomed), zy(0)], [zx(-hd), zy(0)], [zx(-hd), zy(dep)], [zx(hd), zy(dep)], [zx(hd), zy(0)], [zx(zoomed), zy(0)]], C.good, 2, .95);
        }
      });
      if (oddEven) { txt('odd orbits', fx0 + 12, y1 + hTop - 22, { c: C.ice, s: 11 }); txt('even orbits', fx0 + 12, y1 + hTop - 8, { c: C.pink, s: 11 }); }
      // time strip with dip marks
      panel(x, y2, w, h2, 'flattened light curve' + (found ? ' · dips of this orbit marked' : ''));
      const tx = sc(T0, T1, x + 10, x + w - 10), ty = sc(lo * 1.3, hi * 1.3, y2 + h2 - 6, y2 + 22);
      const fade = found && !sig.weak ? 1 - .75 * seg(T, sig.doubled ? 7.8 : 6.2, sig.doubled ? 9.0 : 7.4) : 1;
      dots(t, f, tx, ty, C.ice, 1.2, .45);
      if (found && T > 4.2) {
        const Pm = sig.period, hw = sig.dur_d * 1.5;
        let e0 = Math.ceil((T0 - sig.t0) / Pm), e1 = Math.floor((T1 - sig.t0) / Pm);
        for (let e = e0; e <= e1; e++) { const tc = sig.t0 + e * Pm; vline(tx(tc), y2 + 20, y2 + h2 - 4, `rgba(102,209,158,${.55 * fade})`, [2, 2]); }
        const xs = [], ys = []; for (let i = 0; i < n; i++) { let p = ((t[i] - sig.t0) / Pm) % 1; if (p < 0) p++; const d = Math.min(p, 1 - p) * Pm; if (d < hw) { xs.push(t[i]); ys.push(f[i]); } }
        dots(xs, ys, tx, ty, C.good, 2.4, .95 * fade);
      }
    }
  };
}

/* ---------- scene: fold (stacking each dip) ---------- */
function sceneFold() {
  const S = D.sigs, per = 5.4;
  const say = [];
  S.forEach((s, i) => {
    say.push([i * per, `Signal ${i + 1} (${fmt(s.period, 3)} d): each dip is cut out and laid on top of the others. ${s.tr.length} dips shown.`]);
    say.push([i * per + 2.6, `Stacked: random noise cancels while the real dip adds up, so the shape gets sharper with every orbit.`]);
    say.push([i * per + 3.6, `What the models read: the whole orbit squeezed into 2001 bins and a close-up of the dip in 201 bins, cut into 63 + 26 = 89 patches.`]);
  });
  return {
    dur: per * S.length, tabs: S.map((s, i) => [i * per, `Signal ${i + 1}`]),
    say,
    nums: S.flatMap((s, i) => [[i * per, `signal ${i + 1} orbit`, fmt(s.period, 3) + ' d'], [i * per + .2, `signal ${i + 1} dips`, fmt(s.n_dips)]]),
    draw(T) {
      const i = Math.min(S.length - 1, Math.floor(T / per)), s = S[i], u = T - i * per;
      const x = 14, w = W - 28, hTop = Math.round((H - 34) * .62), y1 = 12, y2 = y1 + hTop + 12, h2 = H - y2 - 18;
      const cols = Math.min(4, Math.ceil(Math.sqrt(s.tr.length))), rows = Math.ceil(s.tr.length / cols) || 1;
      const gw = Math.round(w * .42), cw = (gw - 8) / cols, ch = (hTop - 26) / rows;
      const all = s.tr.flatMap(e => e.y), lo = pct(all, .01) * 1.3, hi = pct(all, .99) * 1.6;
      panel(x, y1, gw, hTop, 'each dip on its own');
      const bx = x + gw + 12, bw = w - gw - 12;
      panel(bx, y1, bw, hTop, 'stacked');
      const BX = sc(-2.5, 2.5, bx + 14, bx + bw - 14), BY = sc(lo, hi, y1 + hTop - 14, y1 + 26);
      hline(BY(0), bx + 10, bx + bw - 10, C.grid);
      const nT = s.tr.length, tPer = 2.5 / Math.max(1, nT);
      const sx = [], sy = [];
      s.tr.forEach((e, j) => {
        const c = j % cols, r = Math.floor(j / cols);
        const cx = x + 4 + c * cw, cy = y1 + 22 + r * ch;
        const fx = sc(-2.5, 2.5, cx + 4, cx + cw - 4), fy = sc(lo, hi, cy + ch - 4, cy + 4);
        const k = seg(u, j * tPer, j * tPer + .55), a = ease(k);
        ctx.strokeStyle = k > 0 && k < 1 ? C.warn : C.line; rr(cx + 1, cy + 1, cw - 2, ch - 2, 4); ctx.stroke();
        mono(`orbit ${e.n}`, cx + 5, cy + 12, { s: 9.5, al: .8 });
        if (a < 1) dots(e.x.map((v, q) => lerp(fx(v), BX(v), a)), e.y.map((v, q) => lerp(fy(v), BY(v), a)), z => z, z => z, SEG[j % 3], 1.6, .8);
        if (k >= 1) { dots(e.x, e.y, fx, fy, SEG[j % 3], 1.4, .25); e.x.forEach((v, q) => { sx.push(v); sy.push(e.y[q]); }); }
      });
      dots(sx, sy, BX, BY, C.ice, 1.6, .45);
      if (sx.length) { const bm = binMean(sx, sy, -2.5, 2.5, 40); path(bm.map((v, q) => [BX(-2.5 + (q + .5) / 40 * 5), BY(v)]), C.warn, 2, .95); }
      const inn = sy.filter((v, q) => Math.abs(sx[q]) < .35), out = sy.filter((v, q) => Math.abs(sx[q]) > 1);
      if (inn.length > 2) {
        const dep = median(out) - inn.reduce((a, b) => a + b, 0) / inn.length, err = mad(out) / Math.sqrt(inn.length);
        mono(`${s.tr.filter((e, j) => u >= j * tPer + .55).length} stacked · depth ${fmt(dep)} ± ${fmt(err)} ppm`, bx + bw - 12, y1 + 16, { a: 'right', c: C.ice });
      }
      mono('time from the middle of the dip (in dip lengths)', BX(0), y1 + hTop - 2, { a: 'center', s: 9.5 });
      // model views
      const vA = ease(seg(u, 3.6, 4.3));
      panel(x, y2, w, h2, 'what the models read (63 + 26 patches)');
      if (vA > 0) {
        ctx.globalAlpha = vA;
        const gW = Math.round((w - 30) * .66), lX = x + 10 + gW + 10, lW = w - gW - 30;
        [[s.g, x + 10, gW, s.ng, 'whole orbit, 2001 bins'], [s.l, lX, lW, s.nl, 'close-up, 201 bins']].forEach(([v, X0, Wd, np, lab]) => {
          const vx = sc(0, v.length - 1, X0, X0 + Wd), vy = sc(Math.min(...v) * 1.1, Math.max(.3, Math.max(...v)) * 1.1, y2 + h2 - 6, y2 + 22);
          for (let p = 0; p <= np; p++) vline(X0 + p * Wd / np, y2 + 22, y2 + h2 - 6, 'rgba(167,139,250,.18)');
          path(v.map((q, j) => [vx(j), vy(q)]), C.ice, 1.3);
          txt(lab, X0 + Wd, y2 + 16, { a: 'right', s: 10.5 });
        });
        ctx.globalAlpha = 1;
      }
    }
  };
}

/* ---------- scene: checks ---------- */
function sceneCheck() {
  const S = D.sigs.map(s => Object.assign({}, s, { rows: s.rows.filter(r => !/^Trust check/.test(r[0])) })), per = 5.6;
  const say = [];
  S.forEach((s, i) => { say.push([i * per, `Signal ${i + 1}: each test asks one question a real planet should pass. Green passes, amber points to a false alarm, grey could not be checked.`]);
    say.push([i * per + 3.0, 'These numbers are also inputs to the gradient-boosting model in the next step; the tests here are shown so you can judge them yourself.']); });
  return {
    dur: per * S.length, tabs: S.map((s, i) => [i * per, `Signal ${i + 1}`]), say,
    nums: S.flatMap((s, i) => [[i * per, `signal ${i + 1} passes`, `${s.rows.filter(r => r[2] === 'pass').length} of ${s.rows.length}`]]),
    draw(T) {
      const i = Math.min(S.length - 1, Math.floor(T / per)), s = S[i], u = T - i * per;
      const x = 14, w = W - 28, lw = Math.round(w * .45), y = 12, h = H - 30;
      panel(x, y, lw, h, `signal ${i + 1} · ${fmt(s.period, 3)} d`);
      const rh = Math.min(30, (h - 30) / s.rows.length);
      s.rows.forEach((r, j) => {
        const a = ease(seg(u, .2 + j * .26, .45 + j * .26)); if (a <= 0) return;
        const yy = y + 34 + j * rh, col = r[2] === 'pass' ? C.good : r[2] === 'warn' ? C.warn : C.mute;
        ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x + 18, yy - 4, 6, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
        txt(r[2] === 'pass' ? '✓' : r[2] === 'warn' ? '!' : '–', x + 18, yy - 0.5, { a: 'center', c: '#07101c', s: 10, w: 800, al: a });
        txt(fitText(r[0], lw - 44, 12, 600), x + 32, yy - 3, { c: C.ink, s: 12, w: 600, al: a });
        txt(fitText(r[1], lw - 44, 10.5), x + 32, yy + 10, { s: 10.5, al: a });
      });
      // visual tiles
      const tx = x + lw + 12, tw = w - lw - 12, th = (h - 10) / 2, tw2 = (tw - 10) / 2;
      const tiles = [[tx, y], [tx + tw2 + 10, y], [tx, y + th + 10], [tx + tw2 + 10, y + th + 10]];
      // 1 odd/even
      (function () {
        const [X, Y] = tiles[0], a = ease(seg(u, .6, 1.4)); panel(X, Y, tw2, th, 'odd vs even dips');
        if (!s.tr.length) return; ctx.globalAlpha = a;
        const all = s.tr.flatMap(e => e.y), lo = pct(all, .02) * 1.3, hi = pct(all, .98) * 1.5;
        const fx = sc(-2.5, 2.5, X + 10, X + tw2 - 10), fy = sc(lo, hi, Y + th - 22, Y + 24);
        [0, 1].forEach(par => { const xs = [], ys = []; s.tr.forEach(e => { if (((e.n % 2) + 2) % 2 === par) e.x.forEach((v, q) => { xs.push(v); ys.push(e.y[q]); }); });
          if (xs.length) { const bm = binMean(xs, ys, -2.5, 2.5, 26); path(bm.map((v, q) => [fx(-2.5 + (q + .5) / 26 * 5), fy(v)]), par ? C.pink : C.ice, 2); } });
        ctx.globalAlpha = 1;
        txt('odd', X + 10, Y + th - 8, { c: C.ice, s: 10.5 }); txt('even', X + 40, Y + th - 8, { c: C.pink, s: 10.5 });
        mono(isFinite(s.oe) ? `differ ${fmt(s.oe, 1)} σ` : 'not measured', X + tw2 - 10, Y + th - 8, { a: 'right', c: s.oe > 3 ? C.warn : C.good });
      })();
      // 2 half-orbit
      (function () {
        const [X, Y] = tiles[1], a = ease(seg(u, 1.2, 2.0)); panel(X, Y, tw2, th, 'half an orbit later');
        if (!s.sec || !s.sec.x.length) { txt('no data there', X + 10, Y + th / 2, { s: 11 }); return; }
        ctx.globalAlpha = a;
        const all = s.tr.flatMap(e => e.y), lo = pct(all, .02) * 1.3, hi = pct(all, .98) * 1.5;
        const fx = sc(-2.5, 2.5, X + 10, X + tw2 - 10), fy = sc(lo, hi, Y + th - 22, Y + 24);
        dots(s.sec.x, s.sec.y, fx, fy, C.star, 1.6, .5);
        const bm = binMean(s.sec.x, s.sec.y, -2.5, 2.5, 20); path(bm.map((v, q) => [fx(-2.5 + (q + .5) / 20 * 5), fy(v)]), C.warn, 2);
        ctx.globalAlpha = 1;
        mono(isFinite(s.ss) ? `${fmt(s.ss, 1)} σ` : '—', X + tw2 - 10, Y + th - 8, { a: 'right', c: s.ss > 5 ? C.warn : C.good });
      })();
      // 3 size
      (function () {
        const [X, Y] = tiles[2], a = ease(seg(u, 1.8, 3.0)); panel(X, Y, tw2, th, 'size');
        const fx = logsc(.5, 40, X + 14, X + tw2 - 14), yy = Y + th * .55;
        ctx.fillStyle = 'rgba(240,123,108,.16)'; ctx.fillRect(fx(22.4), yy - 12, X + tw2 - 14 - fx(22.4), 24);
        hline(yy, X + 14, X + tw2 - 14, C.mute);
        [[1, 'Earth'], [3.9, 'Neptune'], [11.2, 'Jupiter'], [22.4, 'too big']].forEach(([v, l], j) => { vline(fx(v), yy - 6, yy + 6, C.mute); txt(l, fx(v), j % 2 ? yy - 12 : yy + 20, { a: 'center', s: 10, c: v === 22.4 ? C.bad : C.mute }); });
        if (isFinite(s.rp)) { const v = Math.exp(lerp(0, Math.log(clamp(s.rp, .5, 40)), a)); ctx.fillStyle = s.rp > 22.4 ? C.bad : C.ice; ctx.beginPath(); ctx.arc(fx(v), yy, 6, 0, 7); ctx.fill();
          mono(`${fmt(s.rp, 1)} Earth radii`, X + tw2 - 10, Y + th - 8, { a: 'right', c: s.rp > 22.4 ? C.bad : C.ice }); }
        else txt('star size unknown', X + 12, Y + th - 8, { s: 11 });
      })();
      // 4 light position
      (function () {
        const [X, Y] = tiles[3], a = ease(seg(u, 2.6, 3.6)); panel(X, Y, tw2, th, s.px ? 'dip source (camera pixels)' : 'light position during the dip');
        const cx = X + tw2 / 2, cy = Y + th / 2 + 6, R = Math.min(tw2, th) * .32, pix = D.mission === 'kepler' ? 4 : 21;
        const off = s.px ? s.px.off : (D.mission === 'tess' ? NaN : s.co), sig = s.px ? s.px.sig : s.cs;
        ctx.strokeStyle = C.line; [1, 2, 3].forEach(k => { ctx.beginPath(); ctx.arc(cx, cy, R * k / 3, 0, 7); ctx.stroke(); });
        txt(`rings ${pix}″ apart`, X + 10, Y + th - 8, { s: 10 });
        ctx.fillStyle = C.warn; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 7); ctx.fill(); txt('target', cx + 8, cy - 6, { s: 10, c: C.warn });
        if (isFinite(off)) { const d = Math.min(R * 1.1, off / (3 * pix) * R) * a; ctx.fillStyle = sig > 5 && off > 10 ? C.bad : C.ice; ctx.beginPath(); ctx.arc(cx + d * .8, cy - d * .6, 5, 0, 7); ctx.fill();
          mono(`${fmt(off, 0)}″, ${fmt(sig, 1)} σ`, X + tw2 - 10, Y + th - 8, { a: 'right', c: C.ice }); }
        else mono(D.mission === 'tess' && !s.px ? 'not reliable on TESS' : 'not measured', X + tw2 - 10, Y + th - 8, { a: 'right' });
      })();
    }
  };
}

/* ---------- scene: models and the verdict ---------- */
function sceneModels() {
  const S = D.sigs, per = 15.5;
  const say = [];
  S.forEach((s, i) => {
    const o = i * per;
    say.push([o, `Signal ${i + 1}: three different models judge it. Each copy was trained on thousands of Kepler/TESS signals that experts had already labelled.`]);
    say.push([o + 1.2, 'Gradient boosting reads the check numbers. Bars: how much each number moves this score compared with a typical value (right = toward planet).']);
    say.push([o + 3.8, 'Mamba-2 (a state-space model) reads the 89 patches in order, keeping a running memory, forwards and then backwards.']);
    say.push([o + 6.6, 'The Transformer compares every patch with every other patch at once (attention).']);
    say.push([o + 8.6, 'Colours on the patches: how much the score falls when that patch is hidden. Cyan = evidence for a planet, coral = against.']);
    say.push([o + 10.2, 'The combined score adds each model\'s log-odds times a weight learned on validation stars, then turns the sum back into a chance.']);
    say.push([o + 12.6, 'The 95% check: cut-offs chosen so that at most 5% of real planets are ever dropped and at most 5% of false alarms kept.']);
  });
  return {
    dur: per * S.length, tabs: S.map((s, i) => [i * per, `Signal ${i + 1}`]), say,
    nums: S.flatMap((s, i) => [[i * per + 3.4, `${i + 1} · gradient boosting`, fmt(s.cal.gb, 2)], [i * per + 6.4, `${i + 1} · Mamba-2`, fmt(s.cal.mamba, 2)],
                               [i * per + 8.4, `${i + 1} · Transformer`, fmt(s.cal.tf, 2)], [i * per + 11.6, `${i + 1} · combined`, fmt(s.combined, 2)],
                               [i * per + 12.2, `${i + 1} · verdict`, s.verdict], [i * per + 14.2, `${i + 1} · 95% check`, s.call || '—']]),
    draw(T) {
      const i = Math.min(S.length - 1, Math.floor(T / per)), s = S[i], u = T - i * per;
      const x = 14, w = W - 28;
      const hA = Math.round(H * .16), yA = 10, yB = yA + hA + 10, hB = Math.round(H * .42), yC = yB + hB + 10, hC = H - yC - 14;
      const imp = s.imp || {}, vmax = Math.max(1e-6, ...Object.values(imp).flatMap(v => v.map(Math.abs)));
      // A: inputs with patch heat
      panel(x, yA, w, hA, '');
      const gW = Math.round((w - 30) * .66), lX = x + 10 + gW + 10, lW = w - gW - 30;
      const heatOn = ease(seg(u, 8.6, 9.6));
      [[s.g, x + 10, gW, s.ng, 0], [s.l, lX, lW, s.nl, s.ng]].forEach(([v, X0, Wd, np, off]) => {
        const vx = sc(0, v.length - 1, X0, X0 + Wd), vy = sc(Math.min(...v) * 1.1, Math.max(.3, Math.max(...v)) * 1.1, yA + hA - 22, yA + 8);
        path(v.map((q, j) => [vx(j), vy(q)]), C.ice, 1.2);
        ['mamba', 'tf'].forEach((k, r) => { if (!imp[k]) return; for (let p = 0; p < np; p++) {
          ctx.fillStyle = heatOn > 0 ? heat(imp[k][off + p], vmax) : 'rgba(214,222,240,.06)'; ctx.globalAlpha = heatOn > 0 ? heatOn : 1;
          ctx.fillRect(X0 + p * Wd / np + .5, yA + hA - 18 + r * 7, Wd / np - 1, 6); ctx.globalAlpha = 1; } });
      });
      txt('patch strips: Mamba-2 (top), Transformer (bottom)', x + w - 10, yA + 14, { a: 'right', s: 9.5 });
      // B: three lanes
      const lw3 = (w - 20) / 3;
      const lanes = [['gb', x, 'Gradient boosting · check numbers', 1.2, 3.6], ['mamba', x + lw3 + 10, 'Mamba-2 · reads patches in order', 3.8, 6.4], ['tf', x + 2 * (lw3 + 10), 'Transformer · all patches at once', 6.6, 8.4]];
      lanes.forEach(([k, X, lab, a0, a1]) => {
        const on = u >= a0, a = seg(u, a0, a1);
        ctx.globalAlpha = on ? 1 : .35; panel(X, yB, lw3, hB, lab); ctx.globalAlpha = 1;
        if (!on) return;
        const inX = X + 10, inW = lw3 - 20, top = yB + 26, bot = yB + hB - 34;
        if (k === 'gb') {
          const rows = (s.gbin || []).slice(0, 6), mx = Math.max(.3, ...rows.map(r => Math.abs(r[2]))), rh = (bot - top) / 6;
          rows.forEach((r, j) => { const b = ease(seg(u, a0 + j * .3, a0 + j * .3 + .5)); if (b <= 0) return; const yy = top + j * rh + 10;
            txt(FEAT[r[0]] || r[0], inX, yy, { s: 10.5, c: C.ink, al: b });
            mono(r[1] === null ? 'n/a' : fmt(r[1], Math.abs(r[1]) < 10 ? 2 : 0), inX + inW, yy, { a: 'right', s: 10, al: b });
            const cx = inX + inW / 2, bw = (inW / 2 - 4) * Math.abs(r[2]) / mx * b;
            ctx.fillStyle = r[2] >= 0 ? C.ice : C.bad; ctx.fillRect(r[2] >= 0 ? cx : cx - bw, yy + 4, bw, 5);
            vline(cx, yy + 2, yy + 11, 'rgba(214,222,240,.3)'); });
        } else {
          const n = s.ng + s.nl, cols = 15, rowsN = Math.ceil(n / cols), cw = inW / cols, chh = Math.min(24, (bot - top - 18 - (k === 'tf' ? 22 : 0)) / rowsN);
          const gy0 = top + (k === 'tf' ? 22 : 4);
          const cellXY = p => [inX + (p % cols) * cw, gy0 + Math.floor(p / cols) * chh];
          if (k === 'tf') { const cX = inX + inW / 2, cY = top + 6; ctx.fillStyle = C.star; rr(cX - 14, cY - 6, 28, 12, 3); ctx.fill(); txt('CLS', cX, cY + 3.5, { a: 'center', s: 8.5, c: '#0b0820', w: 800 });
            const b = ease(seg(u, a0 + .2, a1 - .4));
            for (let p = 0; p < n; p++) { const [cx, cy] = cellXY(p), v = imp.tf ? Math.abs(imp.tf[p]) / vmax : .3; ctx.strokeStyle = `rgba(167,139,250,${clamp(.05 + .9 * v, 0, 1) * b})`; ctx.lineWidth = .6 + 2 * v;
              ctx.beginPath(); ctx.moveTo(cX, cY + 6); ctx.lineTo(cx + cw / 2, cy + chh / 2); ctx.stroke(); } }
          const fw = ease(seg(u, a0, a0 + (a1 - a0) * .5)), bw = ease(seg(u, a0 + (a1 - a0) * .5, a1));
          if (!(k === 'mamba' && a > 0 && a < 1)) txt('patches 1–63 whole orbit · 64–89 close-up', inX, bot - 4, { s: 9.5 });
          for (let p = 0; p < n; p++) {
            const [cx, cy] = cellXY(p); let col = 'rgba(214,222,240,.10)';
            if (k === 'mamba') { const hf = p / n <= fw ? 1 : 0, hb = (n - 1 - p) / n <= bw ? 1 : 0; if (hf) col = 'rgba(94,231,255,.35)'; if (hb) col = 'rgba(167,139,250,.55)'; }
            else col = 'rgba(167,139,250,.25)';
            if (heatOn > 0 && imp[k]) col = heat(imp[k][p], vmax);
            ctx.fillStyle = col; ctx.fillRect(cx + 1, cy + 1, cw - 2, chh - 2);
          }
          if (k === 'mamba' && a > 0 && a < 1) { const p = fw < 1 ? Math.floor(fw * (n - 1)) : n - 1 - Math.floor(bw * (n - 1)); const [cx, cy] = cellXY(p);
            ctx.strokeStyle = C.warn; ctx.lineWidth = 2; ctx.strokeRect(cx, cy, cw, chh); txt(fw < 1 ? 'reading → (memory carried along)' : '← reading back', inX, bot - 4, { s: 10, c: C.warn }); }
        }
        // output bar: raw then calibrated
        const raw = s.raw[k], cal = s.cal[k], b = ease(seg(u, a1 - .6, a1));
        if (b > 0) { gauge(inX, yB + hB - 16, inW, lerp(0, cal, b), MCOL[k]); mono(`raw ${fmt(raw, 2)} → calibrated ${fmt(cal, 2)}`, inX, yB + hB - 24, { c: MCOL[k], s: 10.5, al: b }); }
      });
      // C: combine + verdict + 95% check
      const cA = seg(u, 10.2, 11.6);
      ctx.globalAlpha = u >= 10.2 ? 1 : .35; panel(x, yC, w, hC, 'combined score and verdict'); ctx.globalAlpha = 1;
      if (u >= 10.2 && s.comb) {
        const cb = s.comb, terms = [['start', cb.intercept, C.mute]].concat(cb.kinds.map((k, j) => [NAMES[k], cb.weights[j] * cb.logodds[j], MCOL[k]]));
        const tot = terms.reduce((a, r) => a + r[1], 0); let cum = 0, mn = 0, mx = 0;
        terms.forEach(r => { cum += r[1]; mn = Math.min(mn, cum); mx = Math.max(mx, cum); });
        const half = Math.round(w * .5), wx = sc(Math.min(-1, mn) - .3, Math.max(1, mx) + .3, x + 150, x + half - 10), ry = yC + 30, rh = Math.min(16, (hC - 50) / terms.length);
        vline(wx(0), ry - 6, ry + terms.length * rh, 'rgba(214,222,240,.35)');
        cum = 0;
        terms.forEach((r, j) => { const b = ease(seg(u, 10.2 + j * .35, 10.6 + j * .35)); if (b <= 0) return; const a = cum, z = cum + r[1] * b;
          ctx.fillStyle = r[2]; ctx.globalAlpha = .85; ctx.fillRect(Math.min(wx(a), wx(z)), ry + j * rh, Math.abs(wx(z) - wx(a)), rh - 4); ctx.globalAlpha = 1;
          mono(`${r[0]} ${r[1] >= 0 ? '+' : '−'}${fmt(Math.abs(r[1]), 2)}`, x + 142, ry + j * rh + rh - 6, { a: 'right', s: 10, c: C.ink, al: b }); cum += r[1]; });
        mono(`log-odds sum ${fmt(tot, 2)}  →  ${fmt(1 / (1 + Math.exp(-tot)), 2)}`, x + 14, yC + hC - 8, { s: 10.5, c: C.ink, al: cA });
        // right half: score dial and verdict, then the 95% check
        const R0 = x + half + 10, RW = w - half - 24, vb = ease(seg(u, 11.4, 12.4));
        const gx = sc(0, 1, R0, R0 + RW), gy = yC + 40;
        ctx.fillStyle = 'rgba(255,200,97,.12)'; ctx.fillRect(gx(.35), gy - 8, gx(.65) - gx(.35), 16);
        gauge(R0, gy - 4, RW, s.combined * vb, s.verdict === 'candidate' ? C.good : s.verdict === 'false alarm' ? C.bad : C.warn, 'combined score', fmt(s.combined, 2));
        txt('0.35–0.65: uncertain zone', gx(.5), gy + 18, { a: 'center', s: 9.5, al: .8 });
        if (u >= 12.0) { const col = s.verdict === 'candidate' ? C.good : s.verdict === 'false alarm' ? C.bad : C.warn;
          txt(s.verdict.toUpperCase(), R0, gy + 40, { s: 17, w: 800, c: col, f: '"Sora", "Manrope", sans-serif', al: ease(seg(u, 12, 12.5)) });
          if (s.why) txt(s.why.length > 60 ? s.why.slice(0, 58) + '…' : s.why, R0, gy + 54, { s: 10.5, al: ease(seg(u, 12, 12.5)) }); }
        if (u >= 12.6 && s.conf) {
          const yy = gy + 80, q0 = s.conf.q0, q1 = s.conf.q1, p = s.cp, cc = ease(seg(u, 12.6, 13.6));
          if (yy + 22 < yC + hC) {
            ctx.fillStyle = 'rgba(240,123,108,.18)'; ctx.fillRect(gx(0), yy, gx(q0) - gx(0), 10);
            ctx.fillStyle = 'rgba(102,209,158,.18)'; ctx.fillRect(gx(1 - q1), yy + 12, gx(1) - gx(1 - q1), 10);
            txt('false alarm still possible', gx(0) + 2, yy - 3, { s: 9.5, c: C.bad }); txt('planet still possible', gx(1) - 2, yy + 33, { s: 9.5, c: C.good, a: 'right' });
            if (isFinite(p)) { vline(gx(p * cc), yy - 2, yy + 24, C.ink, null, 2); mono(`p ${fmt(p, 2)}`, gx(p * cc) + 4, yy + 21, { s: 9.5, c: C.ink }); }
            if (u >= 13.6) mono(`95% check: ${s.call}${s.override ? ' (score and check disagree)' : ''}`, R0, yy + 48, { c: s.call === 'keep' ? C.good : s.call === 'drop' ? C.bad : C.warn, s: 11.5 });
          }
        }
      }
    }
  };
}

/* ---------- engine ---------- */
const SCENES = { load: sceneLoad, flatten: sceneFlatten, search: sceneSearch, fold: sceneFold, check: sceneCheck, models: sceneModels };
const scene = SCENES[D.kind]();
let start = performance.now(), sayIdx = -1, nShown = 0, tabIdx = -1;
const nums = $('#nums'), say = $('#say'), pb = $('#pb'), tabs = $('#tabs');
if (scene.tabs && scene.tabs.length > 1) {
  tabs.innerHTML = scene.tabs.map((t, i) => `<button data-i="${i}">${t[1]}</button>`).join('');
  tabs.querySelectorAll('button').forEach(b => b.onclick = () => { start = performance.now() - scene.tabs[+b.dataset.i][0] * 1000; nShown = 0; nums.innerHTML = ''; sayIdx = -1; });
}
$('#rp').onclick = () => { start = performance.now(); nShown = 0; nums.innerHTML = ''; sayIdx = -1; };
function resize() { const r = cv.parentElement.getBoundingClientRect(); W = Math.max(300, Math.floor(r.width)); const d = window.devicePixelRatio || 1;
  cv.width = W * d; cv.height = H * d; cv.style.width = W + 'px'; cv.style.height = H + 'px'; ctx.setTransform(d, 0, 0, d, 0, 0); }
resize(); addEventListener('resize', resize);
function frame(now) {
  let T = Math.max(0, (now - start) / 1000);
  if (RM && !scene.loop) T = scene.dur;
  if (!scene.loop && T > scene.dur) T = scene.dur;
  ctx.clearRect(0, 0, W, H);
  try { scene.draw(T); } catch (e) { console.error(e); }
  if (!scene.loop) {
    let k = -1; scene.say.forEach((s, i) => { if (T >= s[0]) k = i; });
    if (k !== sayIdx) { sayIdx = k; say.classList.remove('in'); void say.offsetWidth; say.textContent = k >= 0 ? scene.say[k][1] : ''; say.classList.add('in'); }
    pb.style.width = (100 * T / scene.dur).toFixed(1) + '%';
  } else pb.style.width = (100 * ((T / 9) % 1)).toFixed(1) + '%';
  const due = scene.nums.filter(n => T >= n[0]);
  if (due.length !== nShown) { nShown = due.length; nums.innerHTML = due.map(n => `<div><span>${n[1]}</span><b>${n[2]}</b></div>`).join(''); }
  if (scene.tabs && tabs.children.length) { let k = 0; scene.tabs.forEach((t, i) => { if (T >= t[0]) k = i; }); if (k !== tabIdx) { tabIdx = k; [...tabs.children].forEach((b, i) => b.classList.toggle('on', i === k)); } }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
})();
