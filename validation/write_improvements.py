import json, pandas as pd
L=['# TransitVet v2: what the new checks changed','',
'All numbers are on held-out TEST stars, never used to train, calibrate or tune. Same splits as the experiments: Kepler DR24 split by star 70/15/15; TESS labelled TOIs split by star 50/10/40. '
'Gradient boosting is fitted on training stars; the combiner (logistic regression on the three models\' log-odds) and thresholds on validation stars.','',
'"False alarms caught" and "planets kept" use the threshold that maximised F1 on validation stars.','']
for m in ['kepler','tess']:
    r=json.load(open(f'/home/user/tv/imp/eval_new_{m}.json'))[m]
    L+=[f"## {m.upper()} (train {r['n']['train']}, validation {r['n']['val']}, test {r['n']['test']} signals)",'',pd.DataFrame(r['rows']).to_markdown(index=False),'',
        'Test PR-AUC of the new gradient boosting when one new check is left out: '+', '.join(f'{k} {v}' for k,v in r['drop_one_gb_new_pr_auc'].items()),'']
L+=['## Notes','',
'- gb_old / combined_old = the previous models; gb_new / combined_new = with the new checks (cen_sigma, cen_offset_arcsec = light-position check; rp_est_re = planet size from depth and star radius; contratio = light from neighbouring stars).',
'- On TESS, planet size and crowding carry most of the gain (TESS test PR-AUC with the old features plus only those two: 0.940; plus only the light-position check: 0.890). TESS pixels are 21 arcsec wide, so the light-position check sees less there.',
'- On Kepler (4 arcsec pixels) each new check adds a little; with the combiner the test PR-AUC goes from 0.947 (AstroNet CNN alone) to 0.968, and planets missed at the chosen threshold drop from 50 to 24 of 519.',
'- Caution: the official labels were partly decided by people looking at the same kinds of checks (centroid offsets, planet size). So part of the gain means "agrees better with the official vetting", which is what the tool is for, but it is not independent proof.',
'- Star radii are from the TESS Input Catalog v8.2 (MAST), available for 93% of Kepler and 95% of TESS signals; gradient boosting handles missing values.',
'- Code: imp/run_cen.py, imp/fetch_tic.py, imp/eval_new.py, build_models2.py; feature code in transitvet/tv_centroid.py, extras.py, fit.py.']
open('/home/user/tv/app/validation/IMPROVEMENTS.md','w').write('\n'.join(L)+'\n')
