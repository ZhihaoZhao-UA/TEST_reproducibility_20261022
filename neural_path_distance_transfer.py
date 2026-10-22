"""Finite-test-grid diagnostic; test covariance is NOT population covariance."""
from pathlib import Path
import json,time
import numpy as np
from neural_path_factor_train import train
P=Path(__file__).resolve().parent

def run():
 d=json.loads((P/'neural_path_n96_results.json').read_text());p=d['protocol'];z=np.load(P/'neural_path_n96_predictions.npz');rows=[];maxdiff=0.;start=time.time()
 for bank in range(p['banks']):
  rng=np.random.default_rng(np.random.SeedSequence([p['seed'],bank,0]));X=rng.uniform(-1,1,(96,2));Xt=rng.uniform(-1,1,(p['test_points'],2));ir=np.random.default_rng(np.random.SeedSequence([p['seed'],bank,1]));h=p['width'];init=(2*ir.normal(0,.7,(h,2)),ir.normal(0,.1,h),ir.uniform(-.5,.5,h))
  for sig in p['signals']:
   for sigma in p['sigmas']:
    tag=f'b{bank}_n96_{sig}_s{sigma}';y=z[tag+'_y']
    for j in range(3):
     ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];F,_,Ft=train(X[ti],y[ti],init,times=p['times'],lr=p['lr'],X_eval=Xt,first_layer_multiplier=4.);maxdiff=max(maxdiff,float(np.max(abs(F-z[tag+f'_fold{j}_F']))));V=z[tag+f'_fold{j}_V']
     for T in p['horizons']:
      ix=[k for k,t in enumerate(p['times']) if t<=T];A=V[:,ix];B=Ft[:,ix];H=A[:,:-1]-A[:,-1,None];G=B[:,:-1]-B[:,-1,None];Sn=H.T@H/len(H);St=G.T@G/len(G);lam=sigma*sigma/len(H);ev,U=np.linalg.eigh(St+lam*np.eye(St.shape[0]));inv=(U/np.sqrt(ev))@U.T;eps=float(np.max(abs(np.linalg.eigvalsh(inv@(Sn-St)@inv))));deff=float(np.trace(St@inv@inv));leverage=float(np.max(np.sum((G@inv)**2,axis=1)));add=2*eps*lam/(1-eps) if eps<1 else None
      rows.append({'bank':bank,'signal':sig,'sigma':sigma,'horizon':T,'fold':j,'lambda':lam,'relative_deviation':eps,'effective_dimension_test':deff,'max_test_leverage_NOT_supremum':leverage,'transfer_factor':1/(1-eps) if eps<1 else None,'simplex_additive_bound':add})
  if bank%8==7:print('distance banks',bank+1,round(time.time()-start,1),flush=True)
 z.close();groups=[]
 for sig in ['all']+p['signals']:
  rr=rows if sig=='all' else [r for r in rows if r['signal']==sig];v=np.array([r['relative_deviation'] for r in rr]);groups.append({'signal':sig,'epsilon_mean':float(v.mean()),'epsilon_median':float(np.median(v)),'epsilon_q90':float(np.quantile(v,.9)),'fraction_epsilon_below1':float(np.mean(v<1)),'effective_dimension_mean':float(np.mean([r['effective_dimension_test'] for r in rr])),'conditional_mean_additive_where_valid':float(np.mean([r['simplex_additive_bound'] for r in rr if r['simplex_additive_bound'] is not None]))})
 out={'status':'Exploratory finite-grid geometry diagnostic. 1024 test inputs approximate population covariance but do not certify it. Max test leverage is not a supremum bound. Lambda=sigma²/32 fixed without test tuning; no new method or risk advantage claim.','groups':groups,'records':rows,'max_prediction_reconstruction':maxdiff};assert maxdiff<1e-12 and len(rows)==1152;(P/'neural_path_distance_transfer.json').write_text(json.dumps(out,indent=2));print(json.dumps(groups),flush=True)
if __name__=='__main__':run()
