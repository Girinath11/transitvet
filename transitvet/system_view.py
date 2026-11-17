"""Top-down view of the signals found around one star. Orbit sizes follow Kepler's third law (distance grows as
period^(2/3)), so they are to scale relative to each other; planet dots are enlarged so they can be seen."""
import html
import json

import numpy as np

COLOR = {'candidate': '#5ee7ff', 'uncertain': '#ffc861', 'false alarm': '#f07b6c'}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent;font:13px Manrope,system-ui,sans-serif;color:#d9d6ce}
.w{border:1px solid rgba(214,222,240,.14);border-radius:12px;background:#070a14;padding:10px}
canvas{width:100%;height:@@H@@px;display:block}.k{display:flex;flex-wrap:wrap;gap:14px;margin-top:6px;font-size:12px;color:#8f93a0}
.k i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:-1px}
</style></head><body><div class="w"><canvas id="c"></canvas><div class="k">@@KEY@@</div></div><script>
const S=@@DATA@@, cv=document.getElementById('c'), x=cv.getContext('2d'), RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function size(){const r=devicePixelRatio||1;cv.width=cv.clientWidth*r;cv.height=cv.clientHeight*r;x.setTransform(r,0,0,r,0,0)}size();addEventListener('resize',size);
const t0=performance.now();
function draw(now){const W=cv.clientWidth,H=cv.clientHeight,cx=W/2,cy=H/2,R=Math.min(W,H)/2-34,T=(now-t0)/1000;
if(R<20){requestAnimationFrame(draw);return}
x.clearRect(0,0,W,H);const g=x.createRadialGradient(cx,cy,0,cx,cy,16);g.addColorStop(0,'#fff8e6');g.addColorStop(1,'#ffb84d');x.fillStyle=g;x.beginPath();x.arc(cx,cy,10,0,7);x.fill();
S.forEach((s,i)=>{const r=10+s.a*(R-10);x.strokeStyle=s.c+'55';x.lineWidth=1;x.setLineDash(s.v==='false alarm'?[3,4]:[]);x.beginPath();x.arc(cx,cy,r,0,7);x.stroke();x.setLineDash([]);
const ang=(RM?0:T*2*Math.PI/(4*s.rel))+i*1.3,px=cx+r*Math.cos(ang),py=cy+r*Math.sin(ang);x.fillStyle=s.c;x.beginPath();x.arc(px,py,s.d,0,7);x.fill();
x.fillStyle='#ece9e1';x.font='12px Manrope,sans-serif';x.fillText(s.l,px+s.d+4,py-s.d-2)});
if(!RM)requestAnimationFrame(draw)}requestAnimationFrame(draw);
</script></body></html>"""


def page(res, height=360):
    sig = res.get('signals') or []
    if not sig:
        return None
    P = np.array([s['period'] for s in sig], float)
    a = (P / P.max()) ** (2 / 3)
    out = []
    for s, ai, p in zip(sig, a, P):
        rp = s.get('rp_est_re')
        d = 3 + 2.2 * np.log2(1 + (rp if rp and np.isfinite(rp) else 2))
        name = (s.get('known') or {}).get('name') or ''
        lab = f"{p:.1f} d" + (f" · {rp:.1f} Earths" if rp and np.isfinite(rp) else '') + (f" · {name}" if name else '')
        out.append(dict(a=float(ai), rel=float(p / P.min()), d=float(min(d, 14)), c=COLOR.get(s['verdict'], '#aaa'), v=s['verdict'], l=lab))
    key = ''.join(f'<span><i style="background:{c}"></i>{html.escape(v)}</span>' for v, c in COLOR.items())
    key += '<span>Orbit distances to scale with each other (Kepler\'s third law); dots enlarged; the fastest orbit takes 4 s here.</span>'
    return PAGE.replace('@@H@@', str(height)).replace('@@DATA@@', json.dumps(out).replace('</', '<\\/')).replace('@@KEY@@', key)
