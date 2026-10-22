"""Unclipped independent convex aggregation aligned with manuscript theorem."""
from pathlib import Path
import json,time
import numpy as np
from neural_path_factor_train import train
from neural_path_average import solve_qp
from neural_path_scale import signal
P=Path(__file__).resolve().parent

def run():
 d=json.loads((P/'neural_path_n96_results.json').read_text());p=d['protocol'];z=np.load(P/'neural_path_n96_predictions.npz');rows=[];gap=0.;reconstruction=0.;start=time.time()
 for bank in range(32):
  rng=np.random.default_rng(np.random.SeedSequence([p['seed'],bank,0]));X=rng.uniform(-1,1,(96,2));Xt=rng.uniform(-1,1,(1024,2));ir=np.random.default_rng(np.random.SeedSequence([p['seed'],bank,1]));h=32;initial=(2*ir.normal(0,.7,(h,2)),ir.normal(0,.1,h),ir.uniform(-.5,.5,h))
  for sig in p['signals']:
   for sigma in p['sigmas']:
    tag=f'b{bank}_n96_{sig}_s{sigma}';y=z[tag+'_y'];truth=signal(Xt,sig);folds=[]
    for j in range(3):
     ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];F,_,Ft=train(X[ti],y[ti],initial,times=p['times'],lr=.25,X_eval=Xt,first_layer_multiplier=4.);reconstruction=max(reconstruction,float(np.max(abs(F-z[tag+f'_fold{j}_F']))));V=z[tag+f'_fold{j}_V']
     for T in p['horizons']:
      ix=[k for k,t in enumerate(p['times']) if t<=T];A=V[:,ix];At=Ft[:,ix];m=len(ix);w,g=solve_qp(A,y[vi],np.zeros(m),tol=1e-8,max_iter=200000);gap=max(gap,g);sel=np.argmin(np.mean((A-y[vi,None])**2,axis=0));pred={'convex':At@w,'selection':At[:,sel],'last':At[:,-1],'uniform':At.mean(1)};folds.append({'fold':j,'horizon':T,'risks':{k:float(np.mean((v-truth)**2)) for k,v in pred.items()},'weights':w.tolist()})
    for T in p['horizons']:
     rr=[r for r in folds if r['horizon']==T];old=next(r for r in d['records'] if r['bank']==bank and r['signal']==sig and r['sigma']==sigma and r['horizon']==T);risks={k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']};risks['full_cv']=old['risks']['cv3'];risks['full_last_equal']=old['risks']['last'] if old['use_sure'] else old['risks']['uniform'];rows.append({'bank':bank,'signal':sig,'sigma':sigma,'horizon':T,'risks':risks,'folds':rr})
  if bank%8==7:print('unclipped banks',bank+1,flush=True)
 out={'status':'Exploratory reuse of all32 existing seed20261018 banks, all12conditions. Unclipped Gaussian labels and predictions, frozen64-trained dictionary, independent32-sample convex weights; no finalrefit, no tuning. Average separate fold risks, not ensemble fold predictions. Main estimator of revised TEST manuscript.','records':rows,'means':{k:float(np.mean([r['risks'][k] for r in rows])) for k in rows[0]['risks']},'comparisons':{},'max_qp_gap':gap,'prediction_reconstruction':reconstruction}
 for k in out['means']:
  if k=='convex':continue
  v=np.array([np.mean([r['risks'][k]-r['risks']['convex'] for r in rows if r['bank']==bank]) for bank in range(32)]);out['comparisons'][k]={'gain':float(v.mean()),'se':float(v.std(ddof=1)/np.sqrt(32))}
 assert len(rows)==384 and reconstruction<1e-12;z.close();(P/'neural_path_TEST_independent.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)
if __name__=='__main__':run()
