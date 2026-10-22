from pathlib import Path
import json,time
import numpy as np
from neural_path_factor_train import train
from neural_path_fast_rate import progressive_weights
from neural_path_average import solve_qp
from neural_path_scale import signal
P=Path(__file__).resolve().parent

def run():
 start=time.time();d=json.loads((P/'neural_path_n96_results.json').read_text());p=d['protocol'];z=np.load(P/'neural_path_n96_predictions.npz');rows=[];maxdiff=0.;maxgap=0.
 for b in range(p['banks']):
  rng=np.random.default_rng(np.random.SeedSequence([p['seed'],b,0]));X=rng.uniform(-1,1,(96,2));Xt=rng.uniform(-1,1,(p['test_points'],2));ir=np.random.default_rng(np.random.SeedSequence([p['seed'],b,1]));h=p['width'];init=(2*ir.normal(0,.7,(h,2)),ir.normal(0,.1,h),ir.uniform(-.5,.5,h))
  for sig in p['signals']:
   for sigma in p['sigmas']:
    tag=f'b{b}_n96_{sig}_s{sigma}';y=z[tag+'_y'];mut=signal(Xt,sig);bound=1.6+sigma*np.sqrt(2*np.log(32));folds=[]
    for j in range(3):
     ti=z[tag+f'_fold{j}_ti'];vi=z[tag+f'_fold{j}_vi'];F,_,Ft=train(X[ti],y[ti],init,times=p['times'],lr=p['lr'],X_eval=Xt,first_layer_multiplier=4.);maxdiff=max(maxdiff,float(np.max(abs(F-z[tag+f'_fold{j}_F']))));V=z[tag+f'_fold{j}_V']
     for T in p['horizons']:
      ix=[k for k,t in enumerate(p['times']) if t<=T];Vc=np.clip(V[:,ix],-bound,bound);Tc=np.clip(Ft[:,ix],-bound,bound);A=np.column_stack([Vc,Vc.mean(1)]);At=np.column_stack([Tc,Tc.mean(1)]);yc=np.clip(y[vi],-bound,bound);w,info=progressive_weights(A,yc,bound);wh,g=solve_qp(A,yc,np.zeros(A.shape[1]),tol=1e-8,max_iter=200000);maxgap=max(maxgap,g);sel=np.argmin(np.mean((A-yc[:,None])**2,axis=0))
      pred={'progressive':At@w,'holdout_convex':At@wh,'holdout_select':At[:,sel],'split_last':Tc[:,-1],'split_uniform':Tc.mean(1)}
      risk={k:float(np.mean((v-mut)**2)) for k,v in pred.items()};pen=8*bound**2*np.log(A.shape[1])/len(vi);tail=4*bound*np.sqrt(2/np.pi)*sigma/32
      folds.append({'fold':j,'horizon':T,'risks':risk,'weights':w.tolist(),'bound':float(bound),'aggregation_penalty':float(pen),'tail_bound':float(tail),'response_clip_fraction':float(np.mean(abs(y[vi])>bound)),'prediction_clip_fraction':float(np.mean(abs(V[:,ix])>bound))})
    for T in p['horizons']:
     rr=[r for r in folds if r['horizon']==T];old=next(r for r in d['records'] if r['bank']==b and r['signal']==sig and r['sigma']==sigma and r['horizon']==T);risk={k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']};risk.update({'full_'+k:old['risks'][k] for k in ['cv3','last','uniform','adaptive']});risk['full_last_or_uniform']=old['risks']['last'] if old['use_sure'] else old['risks']['uniform'];rows.append({k:old[k] for k in ['bank','signal','sigma','horizon']}|{'risks':risk,'folds':rr})
  if b%8==7:print('completed',b+1,'banks',round(time.time()-start,1),flush=True)
 z.close();groups=[]
 for sig in ['all']+p['signals']:
  rr=rows if sig=='all' else [r for r in rows if r['signal']==sig];groups.append({'signal':sig,'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']}})
 cells=[]
 for key in sorted({(r['signal'],r['sigma'],r['horizon']) for r in rows}):
  rr=[r for r in rows if (r['signal'],r['sigma'],r['horizon'])==key];cells.append({'condition':key,'risks':{k:float(np.mean([r['risks'][k] for r in rr])) for k in rr[0]['risks']}})
 comparisons={}
 for k in rows[0]['risks']:
  if k=='progressive':continue
  v=np.array([np.mean([r['risks'][k]-r['risks']['progressive'] for r in rows if r['bank']==b]) for b in range(p['banks'])]);comparisons[k]={'gain':float(v.mean()),'cluster_se':float(v.std(ddof=1)/np.sqrt(len(v))),'cell_mean_wins':sum(c['risks']['progressive']<c['risks'][k] for c in cells)}
 allfolds=[f for r in rows for f in r['folds']];check={'max_prediction_reconstruction':maxdiff,'max_qp_gap':maxgap,'response_clip_fraction':float(np.mean([f['response_clip_fraction'] for f in allfolds])),'prediction_clip_fraction':float(np.mean([f['prediction_clip_fraction'] for f in allfolds])),'theoretical_penalty_range':[min(f['aggregation_penalty']+f['tail_bound'] for f in allfolds),max(f['aggregation_penalty']+f['tail_bound'] for f in allfolds)]};assert maxdiff<1e-12 and len(rows)==384
 out={'protocol':json.loads((P/'neural_path_progressive_protocol.json').read_text()),'groups':groups,'comparisons':comparisons,'cells':cells,'records':rows,'checks':check,'seconds':time.time()-start};(P/'neural_path_progressive_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({'groups':groups,'comparisons':comparisons,'checks':check}),flush=True)
if __name__=='__main__':run()
