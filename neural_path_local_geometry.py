"""Conditional-design geometry of independent fold predictions, all32 banks."""
from pathlib import Path
import json
import numpy as np
from neural_path_scale import signal
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent

def run():
 d=json.loads((P/'neural_path_n96_results.json').read_text());p=d['protocol'];z=np.load(P/'neural_path_n96_predictions.npz');rows=[];gap=0.;violation=-1e9
 for b in range(p['banks']):
  rng=np.random.default_rng(np.random.SeedSequence([p['seed'],b,0]));X=rng.uniform(-1,1,(96,2))
  for sig in p['signals']:
   for sigma in p['sigmas']:
    tag=f'b{b}_n96_{sig}_s{sigma}';y=z[tag+'_y'];mu=signal(X,sig)
    for j in range(3):
     vi=z[tag+f'_fold{j}_vi'];V=z[tag+f'_fold{j}_V'];n=len(vi);eps=y[vi]-mu[vi]
     for T in p['horizons']:
      ix=[k for k,t in enumerate(p['times']) if t<=T];A=V[:,ix];m=A.shape[1];H=A-A.mean(1,keepdims=True);U,s,_=np.linalg.svd(H,full_matrices=False);bounds=[];resids=[]
      for k in range(m+1):
       QH=H-U[:,:k]@(U[:,:k].T@H);delta=float(np.max(np.linalg.norm(QH,axis=0)));resids.append(delta);bounds.append(sigma*sigma*k/n+2*sigma*delta*np.sqrt(2*np.log(m))/n)
      k=int(np.argmin(bounds));w,g=solve_qp(A,y[vi],np.zeros(m),tol=1e-8,max_iter=200000);wo,go=solve_qp(A,mu[vi],np.zeros(m),tol=1e-8,max_iter=200000);gap=max(gap,g,go);dd=A@(w-wo);ex=float(np.mean((A@w-mu[vi])**2)-np.mean((A@wo-mu[vi])**2));upper=float((2*eps@dd-dd@dd)/n+g);violation=max(violation,ex-upper)
      rows.append({'bank':b,'signal':sig,'sigma':sigma,'horizon':T,'fold':j,'M':m,'best_k':k,'local_expected_bound':min(bounds),'dimension_only_bound':sigma*sigma*(m-1)/n,'residual_radius':resids[k],'singular_values':s.tolist(),'all_bounds':bounds,'realized_design_excess':ex})
  if b%8==7:print('geometry banks',b+1,flush=True)
 groups=[]
 for sig in ['all']+p['signals']:
  rr=rows if sig=='all' else [r for r in rows if r['signal']==sig];groups.append({'signal':sig,**{k:float(np.mean([r[k] for r in rr])) for k in ['best_k','local_expected_bound','dimension_only_bound','residual_radius','realized_design_excess']}})
 checks={'max_qp_gap':gap,'max_variational_inequality_violation':violation};assert violation<1e-10 and len(rows)==1152
 out={'status':'Exploratory geometry audit. Frozen fold candidates independent of validation response noise. Bounds concern fixed validation-input risk, not new-input risk; expected bound is not a pointwise confidence bound. No new improvement experiment.','groups':groups,'checks':checks,'records':rows};(P/'neural_path_local_geometry.json').write_text(json.dumps(out,indent=2));print(json.dumps({'groups':groups,'checks':checks}),flush=True)
if __name__=='__main__':run()
