"""Post-outcome diagnosis only: fold/full test-grid coefficient oracles."""
from pathlib import Path
import json,time
import numpy as np
from neural_path_nested import correction
from neural_path_factor_train import train
from neural_path_scale import signal
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent

def run():
 start=time.time();d=json.loads((P/'neural_path_nested_results.json').read_text());p=json.loads((P/'neural_path_n96_protocol.json').read_text());z=np.load(P/'neural_path_n96_predictions.npz');cs=d['protocol']['coefficients'];out=[];checks={'prediction_reconstruction':0.,'identity_error':0.,'max_qp_gap':0.}
 for b in d['protocol']['banks']:
  cache=np.load(P/f'neural_path_nested_bank{b}_cache.npz');rng=np.random.default_rng(np.random.SeedSequence([p['seed'],b,0]));X=rng.uniform(-1,1,(96,2));Xt=rng.uniform(-1,1,(p['test_points'],2));ir=np.random.default_rng(np.random.SeedSequence([p['seed'],b,1]));h=p['width'];init=(2*ir.normal(0,.7,(h,2)),ir.normal(0,.1,h),ir.uniform(-.5,.5,h))
  for sig in p['signals']:
   for sigma in p['sigmas']:
    tag=f'b{b}_n96_{sig}_s{sigma}';y=z[tag+'_y'];mut=signal(Xt,sig);foldrisks={T:np.zeros(len(cs)) for T in p['horizons']}
    for j in range(3):
     ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];F=z[tag+f'_fold{j}_F'];df=z[tag+f'_fold{j}_df'];B,_,Bt=train(X[ti],y[ti],init,times=p['times'],lr=p['lr'],X_eval=Xt,first_layer_multiplier=4.)
     checks['prediction_reconstruction']=max(checks['prediction_reconstruction'],float(np.max(abs(B-F))))
     inner=[tuple(cache[tag+f'_outer{j}_inner{k}_'+name] for name in ['ti','vi','F','df','V']) for k in range(3)]
     for T in p['horizons']:
      ix=[k for k,t in enumerate(p['times']) if t<=T];delta=correction(inner,y[ti],sigma*sigma,ix)
      for k,c in enumerate(cs):
       w,g=solve_qp(F[:,ix],y[ti],sigma*sigma*df[ix]/len(ti)+c*delta/2,tol=1e-8,max_iter=200000);checks['max_qp_gap']=max(checks['max_qp_gap'],g);foldrisks[T][k]+=len(vi)/96*float(np.mean((Bt[:,ix]@w-mut)**2))
    for T in p['horizons']:
     r=next(r for r in d['records'] if r['bank']==b and r['signal']==sig and r['sigma']==sigma and r['horizon']==T);q=foldrisks[T];f=np.array([r['risks'][str(c)] for c in cs]);sel=cs.index(r['coefficient']);fo=int(np.argmin(q));fu=int(np.argmin(f))
     a=float(q[sel]-q[fo]);bterm=float((f[sel]-q[sel])-(f[fo]-q[fo]));cterm=float(f[fo]-f[fu]);total=float(f[sel]-f[fu]);base=float(f[fu]-r['risks']['cv3']);checks['identity_error']=max(checks['identity_error'],abs(total-a-bterm-cterm),abs(f[sel]-r['risks']['cv3']-a-bterm-cterm-base));assert a>=-1e-12 and cterm>=-1e-12
     out.append({k:r[k] for k in ['bank','signal','sigma','horizon']}|{'selected_coefficient':cs[sel],'fold_oracle_coefficient':cs[fo],'full_oracle_coefficient':cs[fu],'fold_risks':q.tolist(),'full_risks':f.tolist(),'components':{'fold_selection_excess':a,'differential_transfer_signed':bterm,'oracle_mismatch':cterm,'total_coefficient_selection_excess':total,'full_oracle_minus_cv':base},'risks':{'nested':float(f[sel]),'fold_oracle_on_full':float(f[fo]),'full_coefficient_oracle':float(f[fu]),'cv3':r['risks']['cv3'],'last_or_uniform':r['risks']['last_or_uniform']}})
  cache.close();print('diagnosed bank',b,round(time.time()-start,1),flush=True)
 z.close();groups=[]
 for sig in ['all']+p['signals']:
  rr=out if sig=='all' else [r for r in out if r['signal']==sig];group={'signal':sig,'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']},'components':{}}
  for k in rr[0]['components']:
   v=np.array([np.mean([r['components'][k] for r in rr if r['bank']==bank]) for bank in d['protocol']['banks']]);group['components'][k]={'mean':float(v.mean()),'cluster_se':float(v.std(ddof=1)/np.sqrt(len(v)))}
  groups.append(group)
 assert checks['prediction_reconstruction']<1e-12 and checks['identity_error']<1e-12
 result={'status':'Post-outcome diagnosis on 8 old banks. Fold and full coefficient oracles use the same finite1024-point test truth and are optimistic, not deployable. No causal decomposition or population guarantee. Differential transfer is signed. Fold risk is average risk of three separately fitted models, NOT risk of their averaged prediction.','groups':groups,'records':out,'checks':checks};(P/'neural_path_nested_decomposition.json').write_text(json.dumps(result,indent=2));print(json.dumps({'groups':groups,'checks':checks}),flush=True)
if __name__=='__main__':run()
