"""Exploratory 2x2 ablation: inverse-size transport and fold-disagreement damping."""
from pathlib import Path
import json
import numpy as np
from neural_path_average import solve_qp
from neural_path_scale import signal
P=Path(__file__).resolve().parent

def run(ds):
 d=json.loads((P/f'neural_path_{ds}_results.json').read_text());p=d['protocol'];z=np.load(P/f'neural_path_{ds}_predictions.npz');rows=[];gap=0.;inputs={}
 for b in range(p['banks']):
  rng=np.random.default_rng(np.random.SeedSequence([p['seed'],b,0]));inputs[b]=rng.uniform(-1,1,(96,2))
 for r in d['records']:
  n=r['n'];s2=r['sigma']**2;tag=f'b{r["bank"]}_n{n}_{r["signal"]}_s{r["sigma"]}';ix=[j for j,t in enumerate(p['times']) if t<=r['horizon']];F=z[tag+'_F'][:,ix];df=z[tag+'_df'][ix];y=z[tag+'_y'];m=len(ix);D=[]
  for j in range(3):
   ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];B=z[tag+f'_fold{j}_F'][:,ix];V=z[tag+f'_fold{j}_V'][:,ix];dd=z[tag+f'_fold{j}_df'][ix]
   delta=np.mean((V-y[vi,None])**2,axis=0)-np.mean((B-y[ti,None])**2,axis=0)-2*s2*dd/len(ti)
   D.append(delta-delta.mean())
  D=np.array(D);delta=D.mean(0);energy=float(delta@delta);disagreement=float(np.mean(np.sum((D-delta)**2,axis=1)));a=energy/(energy+disagreement) if energy+disagreement>0 else 0.
  W={k:np.array(v) for k,v in r['weights'].items()};W['last_or_uniform']=W['last'] if r['use_sure'] else W['uniform']
  scales={'raw_correction':1.,'transport':2/3,'damped':a,'transport_damped':2*a/3}
  for k,c in scales.items():
   W[k],g=solve_qp(F,y,s2*df/n+c*delta/2,tol=1e-8,max_iter=200000);gap=max(gap,g)
  # All deployable choices made before truth is accessed.
  Ft=z[tag+'_Ft'][:,ix];mut=z[tag+'_mut'];mu=signal(inputs[r['bank']][:n],r['signal']);tr=np.mean((F-mu[:,None])**2,axis=0);te=np.mean((Ft-mut[:,None])**2,axis=0);target=te-tr;target-=target.mean();C=np.mean((F-y[:,None])**2,axis=0)+2*s2*df/n-s2;C-=C.mean();tec=te-te.mean()
  cal={k:float(np.mean((C+c*delta-tec)**2)) for k,c in {'sure':0.,**scales}.items()}
  rows.append({k:r[k] for k in ['bank','n','signal','sigma','horizon']}|{'a':a,'risks':{k:float(np.mean((Ft@w-mut)**2)) for k,w in W.items()},'calibration':cal,'target_gap':target.tolist(),'delta':delta.tolist(),'weights':{k:W[k].tolist() for k in scales}})
 z.close();groups=[]
 for n in p['sample_sizes']:
  for sig in ['all']+p['signals']:
   rr=[r for r in rows if r['n']==n and (sig=='all' or r['signal']==sig)];groups.append({'n':n,'signal':sig,'a':float(np.mean([r['a'] for r in rr])),'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']},'calibration':{k:float(np.mean([r['calibration'][k] for r in rr])) for k in rr[0]['calibration']}})
 cells=[]
 for key in sorted({(r['n'],r['signal'],r['sigma'],r['horizon']) for r in rows}):
  rr=[r for r in rows if (r['n'],r['signal'],r['sigma'],r['horizon'])==key];cells.append({'condition':key,'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']}})
 comparisons={}
 for k in scales:
  comparisons[k]={}
  for base in ['sure','cv3','adaptive','last_or_uniform']:
   v=np.array([np.mean([r['risks'][base]-r['risks'][k] for r in rows if r['bank']==b]) for b in range(p['banks'])]);comparisons[k][base]={'gain':float(v.mean()),'cluster_se':float(v.std(ddof=1)/np.sqrt(len(v))),'cell_mean_wins':sum(c['risks'][k]<c['risks'][base] for c in cells),'cells':len(cells)}
 law=[]
 if p['sample_sizes']==[48,96]:
  for sig in p['signals']:
   for s in p['sigmas']:
    for T in p['horizons']:
     q={n:np.array([r['target_gap'] for r in rows if r['n']==n and r['signal']==sig and r['sigma']==s and r['horizon']==T]) for n in [48,96]};res=q[96]-.5*q[48];mean=res.mean(0);se=res.std(0,ddof=1)/np.sqrt(len(res));law.append({'signal':sig,'sigma':s,'horizon':T,'mean_gap48':q[48].mean(0).tolist(),'mean_gap96':q[96].mean(0).tolist(),'inverse_size_residual':mean.tolist(),'paired_cluster_se':se.tolist(),'residual_rms':float(np.sqrt(np.mean(mean**2))),'gap96_rms':float(np.sqrt(np.mean(q[96].mean(0)**2)))})
 out={'status':'Exploratory reuse; four prespecified-in-code ablations, no tuning on test outcomes. Inverse-size transport is an unproved hypothesis. Fold disagreement is a damping heuristic, NOT a variance estimate: folds overlap. No new significance claim.','groups':groups,'cells':cells,'comparisons':comparisons,'inverse_size_diagnostic':law,'max_qp_gap':gap,'records':rows};(P/f'neural_path_transport_{ds}.json').write_text(json.dumps(out,indent=2));print(ds,json.dumps({'groups':[g for g in groups if g['signal']=='all'],'comparisons':comparisons,'law':[{k:r[k] for k in ['signal','sigma','horizon','residual_rms','gap96_rms']} for r in law]}),flush=True)
if __name__=='__main__':
 for ds in ['n96','predictability']:run(ds)
