"""Exploratory cached-data risk correction; no test-dependent tuning."""
from pathlib import Path
import json
import numpy as np
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent

def matrix(F,y,df,s2):
 n=len(y);E=y[:,None]-F
 return E.T@E/n+s2/n*(df[:,None]+df[None,:])-s2

def run(dataset):
 d=json.loads((P/f'neural_path_{dataset}_results.json').read_text());p=d['protocol'];z=np.load(P/f'neural_path_{dataset}_predictions.npz');rows=[];maxgap=0.;maxidentity=0.
 for r in d['records']:
  tag=f'b{r["bank"]}_n{r["n"]}_{r["signal"]}_s{r["sigma"]}';idx=[j for j,t in enumerate(p['times']) if t<=r['horizon']]
  F=z[tag+'_F'][:,idx];O=z[tag+'_O'][:,idx];y=z[tag+'_y'];df=z[tag+'_df'][idx];s2=r['sigma']**2;n=len(y);m=len(idx)
  G=matrix(F,y,df,s2);Go=matrix(O,y,np.zeros(m),s2);Gfold=np.zeros_like(G)
  for j in range(3):
   ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];B=z[tag+f'_fold{j}_F'][:,idx];dd=z[tag+f'_fold{j}_df'][idx]
   Gfold+=len(vi)/n*matrix(B,y[ti],dd,s2)
  D=Go-Gfold;delta=np.diag(D);Gc=G+D
  w,g=solve_qp(F,y,s2*df/n+delta/2,tol=1e-8,max_iter=200000);maxgap=max(maxgap,g)
  W={k:np.array(v) for k,v in r['weights'].items()};W['linear_corrected']=w;W['corrected_select']=np.eye(m)[np.argmin(np.diag(Gc))];W['last_or_uniform']=W['last'] if r['use_sure'] else W['uniform']
  # Weight computation is complete before loading evaluation truth.
  Ft=z[tag+'_Ft'][:,idx];mut=z[tag+'_mut'];E=mut[:,None]-Ft;Gt=E.T@E/len(mut)
  risks={k:float(np.mean((Ft@v-mut)**2)) for k,v in W.items()}
  for k in r['risks']:maxidentity=max(maxidentity,abs(risks[k]-r['risks'][k]))
  # Fixed weights: every vertex and uniform, chosen without outcomes.
  V=np.vstack([np.eye(m),np.ones(m)/m]);truth=np.einsum('bi,ij,bj->b',V,Gt,V);est={k:np.einsum('bi,ij,bj->b',V,H,V) for k,H in [('sure',G),('oof',Go),('full_corrected',Gc)]}
  metrics={}
  for k,e in est.items():
   metrics[k]={'mse':float(np.mean((e-truth)**2)),'bias':float(np.mean(e-truth)),'contrast_mse':float(np.mean(((e-e[-1])-(truth-truth[-1]))**2))}
  Q=np.eye(m)-np.ones((m,m))/m;mineig=float(np.linalg.eigvalsh(Q@Gc@Q)[0])
  rows.append({k:r[k] for k in ['bank','n','signal','sigma','horizon']}|{'risks':risks,'calibration':metrics,'contrast_min_eigenvalue':mineig,'weights':{k:W[k].tolist() for k in ['linear_corrected','corrected_select']},'delta':delta.tolist()})
 z.close();groups=[]
 for n in [None]+p['sample_sizes']:
  for s in ['all']+p['signals']:
   rr=[r for r in rows if (n is None or r['n']==n) and (s=='all' or r['signal']==s)]
   group={'n':n,'signal':s,'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']},'calibration':{k:{q:float(np.mean([r['calibration'][k][q] for r in rr])) for q in rr[0]['calibration'][k]} for k in rr[0]['calibration']},'indefinite_fraction':float(np.mean([r['contrast_min_eigenvalue'] < -1e-8 for r in rr])),'paired_gains':{}}
   for k in ['sure','cv3','adaptive','last_or_uniform']:
    v=np.array([np.mean([r['risks'][k]-r['risks']['linear_corrected'] for r in rr if r['bank']==b]) for b in range(p['banks'])]);group['paired_gains'][k]={'mean':float(v.mean()),'cluster_se':float(v.std(ddof=1)/np.sqrt(len(v)))}
   groups.append(group)
 cells=[]
 for key in sorted({(r['n'],r['signal'],r['sigma'],r['horizon']) for r in rows}):
  rr=[r for r in rows if (r['n'],r['signal'],r['sigma'],r['horizon'])==key];cells.append(dict(zip(['n','signal','sigma','horizon'],key))|{'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']}})
 out={'status':'Exploratory reuse of previously examined data; no independent confirmation or significance claim. Known noise variance. No candidate changes or test-dependent tuning.','formula':'D=OOF risk matrix - mean fold training SURE matrix. Linear corrected objective = full SURE(w) + diag(D)^T w. Full matrix correction is assessed only at fixed weights, not optimized.','limitations':'Fold-size gap need not transfer to full sample. Linear correction is exact at vertices but omits mixture correction interactions. Fixed-weight SURE does not account for response-adaptive weight divergence.','checks':{'max_qp_gap':maxgap,'existing_risk_reconstruction':maxidentity},'groups':groups,'cells':cells,'records':rows}
 assert maxidentity<1e-12 and maxgap<1.1e-8
 (P/f'neural_path_risk_correction_{dataset}.json').write_text(json.dumps(out,indent=2))
 print(dataset,json.dumps(groups[0]),flush=True)
if __name__=='__main__':
 for dataset in ['n96','predictability']:run(dataset)
