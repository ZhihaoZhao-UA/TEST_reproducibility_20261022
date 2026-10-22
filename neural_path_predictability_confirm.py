from pathlib import Path
import json,time,hashlib
from itertools import product
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from neural_path_factor_train import train
from neural_path_scale import signal
from neural_path_average import solve_qp
from neural_path_stabilize import fit
P=Path(__file__).resolve().parent

def worker(bank):
 p=json.loads((P/'neural_path_predictability_protocol.json').read_text());rng=np.random.default_rng(np.random.SeedSequence([p['seed'],bank,0]));Xall=rng.uniform(-1,1,(96,2));Xt=rng.uniform(-1,1,(p['test_points'],2));eps=rng.normal(size=96);ir=np.random.default_rng(np.random.SeedSequence([p['seed'],bank,1]));h=p['width'];init=(2*ir.normal(0,.7,(h,2)),ir.normal(0,.1,h),ir.uniform(-.5,.5,h));rr=[];pred={};gapmax=0.
 def qp(A,y,pen):
  nonlocal gapmax
  w,g=solve_qp(A,y,pen,tol=1e-7,max_iter=100000);gapmax=max(gapmax,g);return w
 for n,s,sigma in product(p['sample_sizes'],p['signals'],p['sigmas']):
  X=Xall[:n];mu=signal(X,s);mut=signal(Xt,s);y=mu+sigma*eps[:n];kw={'times':p['times'],'lr':p['lr'],'first_layer_multiplier':4.}
  F,J,Ft=train(X,y,init,directions=np.eye(n),X_eval=Xt,**kw);df=np.trace(J,axis1=1,axis2=2);O=np.empty_like(F);folds=[];tag=f'b{bank}_n{n}_{s}_s{sigma}'
  for j in range(3):
   vi=np.arange(n)[np.arange(n)%3==j];ti=np.arange(n)[np.arange(n)%3!=j];A,D,V=train(X[ti],y[ti],init,directions=np.eye(len(ti)),X_eval=X[vi],**kw);dd=np.trace(D,axis1=1,axis2=2);O[vi]=V;folds.append((ti,vi,A,dd,V))
   for name,v in [('ti',ti),('vi',vi),('F',A),('df',dd),('V',V)]:pred[tag+f'_fold{j}_'+name]=v
  for name,v in [('F',F),('df',df),('Ft',Ft),('O',O),('y',y),('mut',mut)]:pred[tag+'_'+name]=v
  for T in p['horizons']:
   idx=[j for j,t in enumerate(p['times']) if t<=T];A=F[:,idx];At=Ft[:,idx];m=len(idx);pen=sigma**2*df[idx]/n;val=np.empty(n)
   for ti,vi,B,dd,V in folds:
    w=qp(B[:,idx],y[ti],sigma**2*dd[idx]/len(ti));val[vi]=V[:,idx]@w
   loss=float(np.mean((val-y)**2));null=float(np.mean(y*y));use=bool(null-loss>p['rho']*max(null-sigma**2,0));Oc=O[:,idx]
   W={'sure':qp(A,y,pen),'sure_select':np.eye(m)[int(np.argmin(np.mean((A-y[:,None])**2,axis=0)+2*pen))],'cv3':qp(Oc,y,np.zeros(m)),'cv3_select':np.eye(m)[int(np.argmin(np.mean((Oc-y[:,None])**2,axis=0)))],'uniform':np.ones(m)/m,'last':np.eye(m)[-1]}
   W['cv3_ridge1'],g=fit(Oc,y,np.zeros(m),sigma,'ridge',1);gapmax=max(gapmax,g);W['adaptive']=W['sure'].copy() if use else W['uniform'].copy()
   risks={k:float(np.mean((At@w-mut)**2)) for k,w in W.items()};assert np.isfinite(list(risks.values())).all()
   for w in W.values():assert min(w)>=-1e-12 and abs(w.sum()-1)<1e-10
   rr.append({'bank':bank,'n':n,'signal':s,'sigma':sigma,'horizon':T,'use_sure':use,'validation_sure_loss':loss,'null_loss':null,'risks':risks,'weights':{k:w.tolist() for k,w in W.items()}})
 return rr,pred,gapmax

def run():
 pb=(P/'neural_path_predictability_protocol.json').read_bytes();p=json.loads(pb);rr=[];pred={};gap=0.;start=time.perf_counter();ex=ProcessPoolExecutor(max_workers=4)
 try:
  fs=[ex.submit(worker,b) for b in range(p['banks'])]
  for i,f in enumerate(as_completed(fs)):
   a,b,g=f.result();rr.extend(a);pred.update(b);gap=max(gap,g);print(f'Completed {i+1}/32 banks {time.perf_counter()-start:.1f}s',flush=True)
 except BaseException:
  for proc in ex._processes.values():proc.terminate()
  ex.shutdown(wait=True,cancel_futures=True);raise
 ex.shutdown(wait=True);rr.sort(key=lambda r:(r['bank'],r['n'],r['signal'],r['sigma'],r['horizon']));out={'protocol':p,'protocol_sha256':hashlib.sha256(pb).hexdigest(),'records':rr,'max_qp_gap':gap,'seconds':time.perf_counter()-start};(P/'neural_path_predictability_results.json').write_text(json.dumps(out,indent=2));np.savez_compressed(P/'neural_path_predictability_predictions.npz',**pred);print('DONE',out['seconds'],gap)
if __name__=='__main__':run()
