from pathlib import Path
import json,hashlib,math
import numpy as np
from neural_path_stabilize import fit
from neural_path_average import solve_qp
ROOT=Path(__file__).resolve().parent
nodes,weights=np.polynomial.legendre.leggauss(256)

def tcdf(t,df):
 x=abs(float(t));v=(nodes+1)*x/2;c=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi);a=x/2*np.dot(weights,c*(1+v*v/df)**(-(df+1)/2));return float(np.clip(.5+np.sign(t)*a,0,1))

def tquantile(prob,df):
 lo,hi=0.,16.
 for _ in range(70):
  mid=(lo+hi)/2
  if tcdf(mid,df)<prob:lo=mid
  else:hi=mid
 return (lo+hi)/2

def run():
 d=json.loads((ROOT/'neural_path_calibrated_results.json').read_text());p=d['protocol'];rr=d['records'];assert len(rr)==768 and d['full_paths']==384 and d['fold_paths']==1152
 assert d['protocol_sha256']==hashlib.sha256((ROOT/'neural_path_calibrated_protocol.json').read_bytes()).hexdigest()
 z=np.load(ROOT/'neural_path_calibrated_predictions.npz');pred={k:z[k] for k in z.files};z.close();assert len(pred)==8448
 riskerr=0.;calerr=0.;weighterr=0.;checks=0
 for r in rr:
  tag=f'b{r["bank"]}_n{r["n"]}_{r["signal"]}_s{r["sigma"]}';idx=[j for j,t in enumerate(p['times']) if t<=r['horizon']];y=pred[tag+'_y'];n=len(y);m=len(idx);At=pred[tag+'_Ft'][:,idx]
  for name,w in r['weights'].items():
   assert abs(sum(w)-1)<1e-10 and min(w)>=-1e-12
   a=float(np.mean((At@w-pred[tag+'_mut'])**2));riskerr=max(riskerr,abs(a-r['risks'][name]))
  A=pred[tag+'_F'][:,idx];df=pred[tag+'_df'][idx];O=pred[tag+'_O'][:,idx];pen=r['sigma']**2*df/n
  for name,lam in [('calibrated',r['selected_lambda']),('sure',0),('fixed1',1)]:
   if lam==0:w,g=solve_qp(A,y,pen,tol=1e-7,max_iter=100000)
   else:w,g=fit(A,y,pen,r['sigma'],'ridge',lam)
   weighterr=max(weighterr,float(np.max(abs(w-r['weights'][name]))))
  w,g=solve_qp(O,y,np.zeros(m),tol=1e-7,max_iter=100000);weighterr=max(weighterr,float(np.max(abs(w-r['weights']['cv3']))))
  expected={'cv3_select':np.eye(m)[int(np.mean((O-y[:,None])**2,axis=0).argmin())],'uniform':np.ones(m)/m,'last':np.eye(m)[-1]}
  for name,w in expected.items():weighterr=max(weighterr,float(np.max(abs(w-r['weights'][name]))))
  losses=[]
  for lam in p['lambdas']:
   err=np.empty(n);coverage=np.zeros(n,int)
   for fold in range(3):
    k=tag+f'_fold{fold}_';ti=pred[k+'train_idx'];vi=pred[k+'val_idx'];A=pred[k+'F'][:,idx];df=pred[k+'df'][idx];V=pred[k+'V'][:,idx]
    assert not set(ti)&set(vi) and len(ti)+len(vi)==n;coverage[vi]+=1;assert np.max(abs(pred[tag+'_O'][vi][:,idx]-V))<1e-12
    penalty=r['sigma']**2*df/len(ti)
    if lam==0:w,g=solve_qp(A,y[ti],penalty,tol=1e-7,max_iter=100000)
    else:w,g=fit(A,y[ti],penalty,r['sigma'],'ridge',lam)
    err[vi]=(V@w-y[vi])**2
   assert np.all(coverage==1);losses.append(float(err.mean()))
  calerr=max(calerr,float(np.max(abs(np.array(losses)-r['calibration_losses']))));assert p['lambdas'][int(np.argmin(losses))]==r['selected_lambda'];checks+=1
 assert riskerr<1e-12 and calerr<1e-12 and weighterr<1e-12
 critical=tquantile(1-.05/3,31);assert abs(tcdf(tquantile(.975,31),31)-.975)<1e-12
 regimes=[];cells=[]
 for kind in p['signals']:
  rows=[r for r in rr if r['signal']==kind];v=np.array([np.mean([r['risks']['cv3']-r['risks']['calibrated'] for r in rows if r['bank']==b]) for b in range(32)]);mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(32));lower=mean-critical*se
  regimes.append({'signal':kind,'mean_risks':{k:float(np.mean([r['risks'][k] for r in rows])) for k in rows[0]['risks']},'gain_vs_cv3':mean,'se':se,'family95_lower':lower,'passes_zero':lower>0,'passes_margin_0.001':lower>.001,'bonferroni_p_zero':min(1.,3*(1-tcdf(mean/se,31))),'bonferroni_p_margin':min(1.,3*(1-tcdf((mean-.001)/se,31))),'bank_gains':v.tolist(),'positive_banks':int(sum(v>0)),'leave_one_out_mean_range':[float(min((v.sum()-v)/31)),float(max((v.sum()-v)/31))],'lambda_counts':{str(lam):sum(r['selected_lambda']==lam for r in rows) for lam in p['lambdas']}})
 for n,kind,sigma,T in sorted({(r['n'],r['signal'],r['sigma'],r['horizon']) for r in rr}):
  rows=[r for r in rr if (r['n'],r['signal'],r['sigma'],r['horizon'])==(n,kind,sigma,T)];v=np.array([r['risks']['cv3']-r['risks']['calibrated'] for r in rows]);cells.append({'n':n,'signal':kind,'sigma':sigma,'horizon':T,'mean_risks':{k:float(np.mean([r['risks'][k] for r in rows])) for k in rows[0]['risks']},'gain_vs_cv3':float(v.mean()),'se':float(v.std(ddof=1)/np.sqrt(32))})
 out={'critical_t':critical,'regimes':regimes,'cells':cells,'risk_reconstruction_error':riskerr,'calibration_reconstruction_error':calerr,'weight_reconstruction_error':weighterr,'records_checked':checks,'inference':'Three prespecified regime means; Bonferroni one-sided paired t approximate confidence, df31. Cell-specific rows descriptive only. Not all CV variants or arbitrary tasks.'};(ROOT/'neural_path_calibrated_analysis.json').write_text(json.dumps(out,indent=2))
 lines=['训练内校准惩罚强度：新种子验证','主检验：三个预设信号组，各平均8条件；32独立簇；单侧Bonferroni校正，近似t推断。','signal | calibrated | matched CV3 | gain | SE | family95 lower | lower>0.001']
 for r in regimes:lines.append(f"{r['signal']} | {r['mean_risks']['calibrated']:.6f} | {r['mean_risks']['cv3']:.6f} | {r['gain_vs_cv3']:+.6f} | {r['se']:.6f} | {r['family95_lower']:+.6f} | {r['passes_margin_0.001']}")
 lines+=['','全部24条件，以下逐条件未作显著性认定：','n | signal | sigma | steps | calibrated | CV3 | CV3select | SURE | fixed1 | uniform | last | gain(SE)']
 for r in cells:
  a=r['mean_risks'];lines.append(f"{r['n']} | {r['signal']} | {r['sigma']} | {r['horizon']} | "+' | '.join(f'{a[k]:.6f}' for k in ['calibrated','cv3','cv3_select','sure','fixed1','uniform','last'])+f" | {r['gain_vs_cv3']:+.6f} ({r['se']:.6f})")
 (ROOT/'neural_path_calibrated_report.txt').write_text('\n'.join(lines));print('critical',critical)
 for r in regimes:print(json.dumps(r))
 print('checks',riskerr,calerr,weighterr,checks)

if __name__=='__main__':run()
