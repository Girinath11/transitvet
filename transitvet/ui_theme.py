"""Look of the Streamlit dashboard. Matches the TransitVet website: serif headlines, plain sans text,
hairline rules, and computer-generated deep-space backgrounds (images in transitvet/assets)."""
import base64
import os

_A = os.path.join(os.path.dirname(__file__), 'assets')


def _img(name):
    try:
        with open(os.path.join(_A, name), 'rb') as fh:
            return 'data:image/jpeg;base64,' + base64.b64encode(fh.read()).decode()
    except OSError:
        return ''


_DEEP, _BAND = _img('deepfield.jpg'), _img('hero_band.jpg')

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --void:#03050b; --panel:rgba(10,14,25,.80); --line:rgba(214,222,240,.14); --line2:rgba(214,222,240,.28);
        --ink:#ece9e1; --soft:#c9c6bd; --mute:#8f93a0; --star:#a78bfa; --ice:#5ee7ff; }
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, textarea, button { font-family: "Manrope", "Segoe UI", system-ui, sans-serif !important; }
.stApp { background: linear-gradient(rgba(3,5,11,.62), rgba(3,5,11,.62)), url(__DEEP__) center / cover fixed, #03050b; }
[data-testid="stHeader"] { background: rgba(3,5,11,.7); }
[data-testid="stSidebar"] { background: rgba(6,9,17,.97); border-right: 1px solid var(--line); }
h1, h2, h3, [data-testid="stHeading"] * { font-family: "Sora", "Segoe UI", sans-serif !important; font-weight: 600 !important; letter-spacing: -.03em; }
h2 { font-size: 1.8rem !important; } h3 { font-size: 1.35rem !important; }
[data-testid="stSidebar"] h2 { font-size: 1.25rem !important; color: var(--star); }
code, pre, .stCode * { font-family: "JetBrains Mono", Consolas, monospace !important; }
[data-baseweb="tab-list"] { gap: 22px; border-bottom: 1px solid var(--line); }
[data-baseweb="tab-list"] button p { font-size: 15.5px !important; }
[data-baseweb="tab-highlight"] { background-color: var(--star) !important; }
[data-testid="stMetric"] { border-top: 1px solid var(--line2); padding: 10px 0 0; }
[data-testid="stMetricLabel"] p { font-size: 13px !important; color: var(--mute) !important; }
[data-testid="stMetricValue"] { font-family: "Sora", sans-serif !important; font-weight: 600; letter-spacing: -.03em; font-size: 2rem !important; }
[data-testid="stExpander"] { background: var(--panel); border: 0 !important; border-top: 1px solid var(--line2) !important; border-radius: 0 !important; }
[data-testid="stExpander"] details { border: 0 !important; }
[data-testid="stExpander"] summary p { font-weight: 600; }
.stButton > button, .stDownloadButton > button { border-radius: 8px !important; font-weight: 600 !important; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg,#a78bfa,#5ee7ff) !important; border-color: transparent !important; color: #0a0820 !important; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-top: 1px solid var(--line2); }
[data-testid="stImage"] img, [data-testid="stPyplot"] img { border-radius: 0; border-top: 1px solid var(--line2); }
[data-testid="stStatusWidget"], [data-testid="stStatus"] { border-radius: 0 !important; }

/* header band */
.tv-hero { position: relative; margin: -8px 0 14px; padding: 34px 30px 30px; min-height: 230px;
  background: linear-gradient(90deg, rgba(3,5,11,.94) 0%, rgba(3,5,11,.6) 50%, rgba(3,5,11,.1) 100%), url(__BAND__) right center / cover;
  border-bottom: 1px solid var(--line2); display: grid; grid-template-columns: minmax(0,1.5fr) minmax(0,1fr); gap: 24px; align-items: end; }
@media (max-width: 900px) { .tv-hero { grid-template-columns: 1fr; padding: 24px 18px } }
.tv-hero { overflow: hidden; background-size: auto, 112% auto; animation: tv-pan 40s ease-in-out infinite alternate; }
@keyframes tv-pan { from { background-position: 0 0, 100% 40% } to { background-position: 0 0, 70% 60% } }
.tv-hero::before, .tv-hero::after { content: ""; position: absolute; width: 140px; height: 1.5px; border-radius: 2px; opacity: 0;
  background: linear-gradient(90deg, rgba(235,240,255,.95), rgba(160,140,255,0)); transform: rotate(-25deg); pointer-events: none; }
.tv-hero::before { top: 18%; left: 62%; animation: tv-meteor 7s ease-in 2s infinite; }
.tv-hero::after { top: 8%; left: 85%; animation: tv-meteor 11s ease-in 6s infinite; }
@keyframes tv-meteor { 0% { opacity: 0; translate: 0 0 } 3% { opacity: 1 } 12% { opacity: 0; translate: -260px 120px } 100% { opacity: 0; translate: -260px 120px } }
@media (prefers-reduced-motion: reduce) { .tv-hero, .tv-hero::before, .tv-hero::after { animation: none !important } }
.tv-kicker { font-weight: 600; font-size: 14px; color: var(--soft); display: flex; align-items: center; gap: 12px; }
.tv-kicker::before { content: ""; width: 22px; height: 2px; background: linear-gradient(90deg,var(--star),var(--ice)); }
.tv-title { font-family: "Sora", sans-serif; font-weight: 600; letter-spacing: -.035em; font-size: clamp(30px, 3.1vw, 44px); line-height: 1.1; margin: 6px 0 12px; color: var(--ink); }
.tv-title em { font-style: normal; font-weight: 400; background: linear-gradient(90deg,#b69cff,#5ee7ff 55%,#ff8bd1); -webkit-background-clip: text; background-clip: text; color: transparent; }
.tv-lede { color: var(--soft); max-width: 60ch; font-size: 16.5px; line-height: 1.6; }
.tv-stage { position: relative; height: 170px; }
.tv-star { position: absolute; left: 50%; top: 6px; width: 104px; height: 104px; margin-left: -52px; border-radius: 50%;
  background: radial-gradient(circle at 42% 40%, #ffffff, #d6e4ff 50%, #8fb0ff 85%, #5b6be0);
  box-shadow: 0 0 36px 10px rgba(150,180,255,.40), 0 0 100px 30px rgba(120,110,255,.18); }
.tv-planet { position: absolute; top: 42px; left: 50%; width: 30px; height: 30px; margin-left: -15px; border-radius: 50%; background: #05070f;
  animation: tv-cross 7s linear infinite; }
@keyframes tv-cross { from { transform: translateX(-110px) } to { transform: translateX(110px) } }
.tv-curve { position: absolute; left: 0; right: 0; bottom: 0; height: 40px; }
.tv-curve path { fill: none; stroke: var(--star); stroke-width: 2; }
.tv-curve .base { stroke: rgba(214,222,240,.25); stroke-width: 1; }
@media (prefers-reduced-motion: reduce) { .tv-planet { animation: none } }

/* first-visit steps */
.tv-steps { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); column-gap: 32px; margin: 4px 0 10px; }
@media (max-width: 900px) { .tv-steps { grid-template-columns: 1fr } }
.tv-step { display: grid; grid-template-columns: 44px minmax(0,1fr); gap: 10px; border-top: 1px solid var(--line); padding: 14px 0; }
.tv-step .n { font-family: "Sora", sans-serif; font-weight: 600; font-size: 14px; color: var(--star); border: 1px solid rgba(167,139,250,.55); border-radius: 50%; width: 32px; height: 32px; display: grid; place-items: center; }
.tv-step .t { font-weight: 600; color: var(--ink); margin-bottom: 2px; }
.tv-step p { font-size: 14.5px; color: var(--soft); margin: 0; line-height: 1.55; }
</style>
""".replace('__DEEP__', _DEEP).replace('__BAND__', _BAND)

HERO = """
<div class="tv-hero">
  <div>
    <div class="tv-kicker">TransitVet · Kepler &amp; TESS</div>
    <div class="tv-title">Pick a star. <em>See what's going round it.</em></div>
    <div class="tv-lede">Type a Kepler (KIC) or TESS (TIC) number, or upload a light curve. TransitVet looks for dips in the starlight that repeat,
    questions each one, and tells you which could be <b>planet candidates</b>. A score here is the model's estimate, not a confirmation.</div>
  </div>
  <div class="tv-stage" aria-hidden="true">
    <div class="tv-star"></div><div class="tv-planet"></div>
    <svg class="tv-curve" viewBox="0 0 300 40" preserveAspectRatio="none"><path class="base" d="M0 12 H300"/>
      <path d="M0 12 H100 L114 26 H186 L200 12 H300"/></svg>
  </div>
</div>
"""

WELCOME = """
<div class="tv-steps">
  <div class="tv-step"><span class="n">1</span><div><div class="t">Choose a star</div><p>Type an ID above or tap one of the examples. KIC numbers are Kepler stars, TIC numbers are TESS stars.</p></div></div>
  <div class="tv-step"><span class="n">2</span><div><div class="t">Give it a few minutes</div><p>The data comes from NASA's MAST archive. A TESS star takes a minute or two; a Kepler star, with four years of data, can take up to nine.</p></div></div>
  <div class="tv-step"><span class="n">3</span><div><div class="t">Read the report card</div><p>Every dip gets a verdict, the reasons behind it in plain words, and the next nights you could watch it from your site.</p></div></div>
</div>
"""
