from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
from neural_path_factor_train import train
from neural_path_scale import signal
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent

def correction(folds,y,s2,ix):
 d=np.zeros(len(ix));n=len(y)
 for ti,vi,F,df,V in folds:
  d+=len(vi)/n*(np.mean((V[:,ix]-y[vi,None])**2,axis=0)-np.mean((F[:,ix]-y[ti,None])**2,axis=0)-2*s2*df[ix]/len(ti))
 return d-d.mean()

def worker(b):
 p=json.loads((P/'neural_path_n96_protocol.json').read_text());pr=json.loads((P/'neural_path_nested_protocol.json').read_text());d=json.loads((P/'neural_path_n96_results.json').read_text());rows=[r for r in d['records'] if r['bank']==b];z=np.load(P/'neural_path_n96_predictions.npz');rng=np.random.default_rng(np.random.SeedSequence([p['seed'],b,0]));X=rng.uniform(-1,1,(96,2));rng.uniform(-1,1,(p['test_points'],2));eps=rng.normal(size=96);ir=np.random.default_rng(np.random.SeedSequence([p['seed'],b,1]));h=p['width'];init=(2*ir.normal(0,.7,(h,2)),ir.normal(0,.1,h),ir.uniform(-.5,.5,h));result=[];cache={};gap=0.
 def qp(F,y,pen):
  nonlocal gap
  w,g=solve_qp(F,y,pen,tol=1e-8,max_iter=200000);gap=max(gap,g);return w
 for sig in p['signals']:
  for sigma in p['sigmas']:
   tag=f'b{b}_n96_{sig}_s{sigma}';y=z[tag+'_y'];assert np.max(abs(y-signal(X,sig)-sigma*eps))<1e-12;outer=[];inner=[]
   for j in range(3):
    ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];F=z[tag+f'_fold{j}_F'];df=z[tag+f'_fold{j}_df'];V=z[tag+f'_fold{j}_V'];outer.append((ti,vi,F,df,V));n=len(ti);folds=[]
    for k in range(3):
     iv=np.arange(n)[np.arange(n)%3==k];it=np.arange(n)[np.arange(n)%3!=k]
     assert not set(ti[it]).intersection(vi) and not set(ti[iv]).intersection(vi)
     A,J,B=train(X[ti[it]],y[ti[it]],init,times=p['times'],lr=p['lr'],directions=np.eye(len(it)),X_eval=X[ti[iv]],first_layer_multiplier=4.)
     dd=np.trace(J,axis1=1,axis2=2);folds.append((it,iv,A,dd,B))
     for name,v in [('ti',it),('vi',iv),('F',A),('df',dd),('V',B)]:cache[tag+f'_outer{j}_inner{k}_'+name]=v
    inner.append(folds)
   for T in p['horizons']:
    ix=[j for j,t in enumerate(p['times']) if t<=T];coef=pr['coefficients'];val=np.empty((96,len(coef)))
    for j,(ti,vi,F,df,V) in enumerate(outer):
     delta=correction(inner[j],y[ti],sigma*sigma,ix)
     for k,c in enumerate(coef):
      w=qp(F[:,ix],y[ti],sigma*sigma*df[ix]/len(ti)+c*delta/2);val[vi,k]=V[:,ix]@w
    losses=np.mean((val-y[:,None])**2,axis=0);sel=int(np.argmin(losses));delta=correction(outer,y,sigma*sigma,ix);F=z[tag+'_F'][:,ix];df=z[tag+'_df'][ix];W={str(c):qp(F,y,sigma*sigma*df/96+c*delta/2) for c in coef};W['nested']=W[str(coef[sel])]
    # Selection complete before evaluation truth.
    Ft=z[tag+'_Ft'][:,ix];mut=z[tag+'_mut'];r=next(r for r in rows if r['signal']==sig and r['sigma']==sigma and r['horizon']==T);risks=dict(r['risks']);risks['last_or_uniform']=r['risks']['last'] if r['use_sure'] else r['risks']['uniform'];risks.update({k:float(np.mean((Ft@w-mut)**2)) for k,w in W.items()});oldw=np.array(r['weights']['sure']);objective=lambda w:float(np.mean((F@w-y)**2)+2*sigma*sigma*df@w/96);assert abs(objective(W['0.0'])-objective(oldw))<2e-7
    result.append({k:r[k] for k in ['bank','n','signal','sigma','horizon']}|{'coefficient':coef[sel],'validation_losses':losses.tolist(),'risks':risks,'weights':{k:v.tolist() for k,v in W.items()}})
 z.close();np.savez_compressed(P/f'neural_path_nested_bank{b}_cache.npz',**cache);(P/f'neural_path_nested_bank{b}.json').write_text(json.dumps({'records':result,'max_qp_gap':gap},indent=2));return result,gap

def run():
 p=json.loads((P/'neural_path_nested_protocol.json').read_text());rows=[];gap=0.;start=time.time()
 with ProcessPoolExecutor(max_workers=3) as ex:
  jobs={}
  for b in p['banks']:
   f=P/f'neural_path_nested_bank{b}.json'
   if f.exists():
    saved=json.loads(f.read_text());rows+=saved['records'];gap=max(gap,saved['max_qp_gap']);print('Reused bank',b,flush=True)
   else:jobs[ex.submit(worker,b)]=b
  for f in as_completed(jobs):
   r,g=f.result();rows+=r;gap=max(gap,g);print('Completed bank',jobs[f],round(time.time()-start,1),'seconds',flush=True)
 rows.sort(key=lambda r:(r['bank'],r['signal'],r['sigma'],r['horizon']));groups=[]
 for sig in ['all','smooth','oscillatory','mixed']:
  rr=rows if sig=='all' else [r for r in rows if r['signal']==sig];groups.append({'signal':sig,'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']},'coefficient_counts':{str(c):sum(r['coefficient']==c for r in rr) for c in p['coefficients']}})
 comparisons={}
 for k in ['sure','cv3','adaptive','last_or_uniform',str(2/3)]:
  v=np.array([np.mean([r['risks'][k]-r['risks']['nested'] for r in rows if r['bank']==b]) for b in p['banks']]);comparisons[k]={'gain':float(v.mean()),'cluster_se':float(v.std(ddof=1)/np.sqrt(len(v)))}
 cells=[]
 for key in sorted({(r['signal'],r['sigma'],r['horizon']) for r in rows}):
  rr=[r for r in rows if (r['signal'],r['sigma'],r['horizon'])==key];cells.append({'condition':key,'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']}})
 out={'protocol':p,'groups':groups,'comparisons':comparisons,'cells':cells,'records':rows,'max_qp_gap':gap,'seconds':time.time()-start};(P/'neural_path_nested_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({'groups':groups,'comparisons':comparisons}),flush=True)
if __name__=='__main__':run()
