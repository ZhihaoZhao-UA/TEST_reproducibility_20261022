"""Exploratory/frozen stabilization evaluation on saved predictions; no network retraining."""
from pathlib import Path
import json,time
import numpy as np
from neural_path_average import solve_qp
ROOT=Path(__file__).resolve().parent

def fit(A,y,penalty,sigma,kind,value):
 n,m=A.shape;u=np.ones(m)/m
 if kind=='ridge':
  tau=value*sigma**2*m/n;N=n+m
  B=np.vstack([np.sqrt(N/n)*A,np.sqrt(N*tau)*np.eye(m)])
  t=np.r_[np.sqrt(N/n)*y,np.sqrt(N*tau)*u]
  w,gap=solve_qp(B,t,penalty,tol=1e-7,max_iter=100000)
  # Independent analytic gradient certifies original, unaugmented objective.
  grad=2*A.T@(A@w-y)/n+2*penalty+2*tau*(w-u)
 else:
  nu=.5
  w,gap=solve_qp(np.sqrt(1-nu)*A,y/np.sqrt(1-nu),penalty+nu*np.mean(A*A,axis=0)/2,tol=1e-7,max_iter=100000)
  grad=2*(1-nu)*A.T@A@w/n-2*A.T@y/n+nu*np.mean(A*A,axis=0)+2*penalty
 cert=float(grad@w-grad.min());assert cert<=1.00001e-7 and abs(w.sum()-1)<1e-10 and min(w)>=-1e-12
 return w,max(float(gap),cert)

def run(prefix='neural_path_n_time',output='neural_path_stabilize_explore',methods=None):
 d=json.loads((ROOT/(prefix+'_results.json')).read_text());z=np.load(ROOT/(prefix+'_predictions.npz'));cache={k:z[k] for k in z.files};z.close()
 if methods is None:methods=[(src,'ridge',lam) for src in ['sure','cv'] for lam in [.1,1.,10.]]+[(src,'q',.5) for src in ['sure','cv']]
 out=[];maxgap=0.;start=time.perf_counter()
 for i,r in enumerate(d['records']):
  tag=f'b{r["bank"]}_n{r["n"]}_{r["signal"]}_s{r["sigma"]}';idx=[j for j,t in enumerate(d['protocol']['checkpoints']) if t<=r['horizon']]
  A=cache[tag+'_F'][:,idx];At=cache[tag+'_Ft'][:,idx];O=cache[tag+'_oof'][:,idx];y=cache[tag+'_y'];mu=cache[tag+'_mut'];df=cache[tag+'_df'][idx]
  risks=dict(r['risks']['test']);weights={}
  for src,kind,val in methods:
   name=f'{src}_{kind}_{val:g}';B=A if src=='sure' else O;penalty=r['sigma']**2*df/r['n'] if src=='sure' else np.zeros(len(idx))
   w,gap=fit(B,y,penalty,r['sigma'],kind,val);maxgap=max(maxgap,gap);risks[name]=float(np.mean((At@w-mu)**2));weights[name]=w.tolist()
  out.append({k:r[k] for k in ['bank','n','signal','sigma','horizon']}|{'risks':risks,'weights':weights})
  if (i+1)%144==0:print(f'Evaluated {i+1}/{len(d["records"])} records {time.perf_counter()-start:.1f}s',flush=True)
 names=list(out[0]['risks']);summary=[]
 for signal in ['all']+d['protocol']['signals']:
  rr=out if signal=='all' else [r for r in out if r['signal']==signal]
  summary.append({'signal':signal,'mean_risks':{name:float(np.mean([r['risks'][name] for r in rr])) for name in names}})
 result={'input_prefix':prefix,'protocol_sha256':d['protocol_sha256'],'methods':methods,'note':'Q-shaped objective is heuristic on same-data SURE/OOF dictionaries; no automatic transfer of fixed-dictionary Q-aggregation theorem. Ridge uses knownsigma. Exploratory scores cannot establish independent generalization.','max_qp_gap':maxgap,'seconds':time.perf_counter()-start,'summary':summary,'records':out}
 (ROOT/(output+'_results.json')).write_text(json.dumps(result,indent=2))
 for row in summary:print(row['signal'],json.dumps(row['mean_risks']))
 print('gap',maxgap)
 return result

if __name__=='__main__':run()
