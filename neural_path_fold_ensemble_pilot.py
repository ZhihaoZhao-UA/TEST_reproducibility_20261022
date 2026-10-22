from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from neural_path_factor_train import train
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent;old=json.loads((P/'neural_path_n96_results.json').read_text());cfg=old['protocol'];z=np.load(P/'neural_path_n96_predictions.npz');records=[]
for b in range(8):
 rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],b,0]));X=rng.uniform(-1,1,(96,2));Xt=rng.uniform(-1,1,(1024,2));rr=np.random.default_rng(np.random.SeedSequence([cfg['seed'],b,1]));initial=(2*rr.normal(0,.7,(32,2)),rr.normal(0,.1,32),rr.uniform(-.5,.5,32))
 for sig in cfg['signals']:
  for sigma in cfg['sigmas']:
   tag=f'b{b}_n96_{sig}_s{sigma}';y=z[tag+'_y'];mu=z[tag+'_mut'];Ft=z[tag+'_Ft'];O=z[tag+'_O'];folds=[]
   for j in range(3):
    ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];_,_,T=train(X[ti],y[ti],initial,cfg['times'],X_eval=Xt,first_layer_multiplier=4.);folds.append((vi,T))
   for horizon in cfg['horizons']:
    ix=np.where(np.array(cfg['times'])<=horizon)[0];late=np.where(np.array(cfg['times'])[ix]>=horizon/2)[0];A=O[:,ix];w,g=solve_qp(A,y,np.zeros(len(ix)),tol=1e-8,max_iter=200000);select=np.argmin(np.mean((A-y[:,None])**2,axis=0));foldavg=np.mean([T[:,ix] for vi,T in folds],axis=0);local=[]
    for vi,T in folds:
     sw,_=solve_qp(A[vi],y[vi],np.zeros(len(ix)),tol=1e-8,max_iter=200000);local.append(T[:,ix]@sw)
    ps={'full_cv':Ft[:,ix]@w,'oof_weight_foldmean':foldavg@w,'independent_foldmean':np.mean(local,axis=0),'foldmean_last':foldavg[:,-1],'foldmean_late':foldavg[:,late].mean(1),'foldmean_cvselect':foldavg[:,select],'full_last':Ft[:,ix][:,-1],'full_late':Ft[:,ix][:,late].mean(1)}
    records.append(dict(bank=b,signal=sig,sigma=sigma,horizon=horizon,risks={k:float(np.mean((v-mu)**2)) for k,v in ps.items()}))
 print('bank',b,flush=True)
means={k:float(np.mean([r['risks'][k] for r in records])) for k in records[0]['risks']};out=dict(status='Exploratory pilot reusing first8 historical banks; not confirmation',records=records,means=means);(P/'neural_path_fold_ensemble_pilot.json').write_text(json.dumps(out,indent=2));print(means)
