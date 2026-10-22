from pathlib import Path
import json,re
import numpy as np
P=Path(__file__).resolve().parent
allrows=[];total=0
for name in ['strong','strong_trained']:
 d=json.loads((P/('neural_path_'+name+'_results.json')).read_text());total+=len(d['records'])
 for s in d['summary']:
  rr=[r for r in d['records'] if (r['amp'],r['k'],r['n'])==(s['amp'],s['k'],s['n'])];assert len(rr)==256
  for key,value in s['means'].items():assert abs(value-np.mean([r['risks'][key] for r in rr]))<1e-14
  for key,st in s['paired_gains'].items():
   a=np.array([r['risks'][key]-r['risks']['ls'] for r in rr]);assert abs(a.mean()-st['mean'])<1e-14;assert abs(a.std(ddof=1)/16-st['se'])<1e-14
  allrows.append(s)
 assert np.max(abs(np.array(d['geometry']['Hz'])-np.diag([.75,1])))<1e-12
s=(P/'TEST_submission/supplement.tex').read_text();b=s.split(r'\label{tab:strongcurvature}',1)[1].split(r'\end{table}',1)[0].split(r'\midrule',1)[1].split(r'\bottomrule')[0]
rows=[r.replace(r'\midrule','').strip() for r in b.split(r'\\') if '&' in r];assert len(rows)==18
for row,r in zip(rows,allrows):
 for c,key in zip(row.split('&')[3:],['gradient','full_gradient','equal','full_equal']):
  vals=re.findall(r'-?\d+\.\d+',c);assert vals==[f"{r['paired_gains'][key]['mean']*1e4:.2f}",f"{r['paired_gains'][key]['se']*1e4:.2f}"]
assert total==4608
out=dict(records=total,table_mean_and_se_cells=144,summary_recalculation='passed',hessian_coordinate_check='passed',claim='Mechanism-specific mean comparisons with MC standard errors, not universal superiority or a new aggregation principle')
(P/'neural_path_strong_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
