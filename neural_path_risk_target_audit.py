"""Frozen candidates; separate training-target oracle and test-grid oracle.
Truth is used only in this post-outcome diagnosis, never to select deployable weights.
"""
from pathlib import Path
import json
import numpy as np
from neural_path_scale import signal
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent

def run():
 d=json.loads((P/'neural_path_n96_results.json').read_text());p=d['protocol'];z=np.load(P/'neural_path_n96_predictions.npz');rows=[];checks={'max_qp_gap':0.,'data_reconstruction':0.,'risk_reconstruction':0.,'decomposition_error':0.,'sure_identity_error':0.}
 inputs={}
 for b in range(p['banks']):
  rng=np.random.default_rng(np.random.SeedSequence([p['seed'],b,0]));X=rng.uniform(-1,1,(96,2));Xt=rng.uniform(-1,1,(p['test_points'],2));eps=rng.normal(size=96);inputs[b]=(X,Xt,eps)
 for r in d['records']:
  tag=f'b{r["bank"]}_n{r["n"]}_{r["signal"]}_s{r["sigma"]}';idx=[j for j,t in enumerate(p['times']) if t<=r['horizon']];F=z[tag+'_F'][:,idx];Ft=z[tag+'_Ft'][:,idx];df=z[tag+'_df'][idx];y=z[tag+'_y'];mut=z[tag+'_mut'];X,Xt,e=inputs[r['bank']];mu=signal(X,r['signal']);eps=r['sigma']*e;n=len(y);m=len(idx)
  checks['data_reconstruction']=max(checks['data_reconstruction'],float(np.max(abs(y-mu-eps))),float(np.max(abs(mut-signal(Xt,r['signal'])))))
  ws=np.array(r['weights']['sure']);wa=np.array(r['weights']['adaptive'])
  wtr,g=solve_qp(F,mu,np.zeros(m),tol=1e-8,max_iter=200000);checks['max_qp_gap']=max(checks['max_qp_gap'],g)
  wte,g=solve_qp(Ft,mut,np.zeros(m),tol=1e-8,max_iter=200000);checks['max_qp_gap']=max(checks['max_qp_gap'],g)
  W={'sure':ws,'adaptive':wa,'train_oracle':wtr,'test_oracle':wte,'cv3':np.array(r['weights']['cv3']),'last':np.array(r['weights']['last']),'uniform':np.array(r['weights']['uniform'])}
  risks={name:{'train':float(np.mean((F@w-mu)**2)),'test':float(np.mean((Ft@w-mut)**2))} for name,w in W.items()}
  for name in ['sure','adaptive','cv3','last','uniform']:checks['risk_reconstruction']=max(checks['risk_reconstruction'],abs(risks[name]['test']-r['risks'][name]))
  a=risks['sure']['test']-risks['train_oracle']['test'] # signed, not a nonnegative estimation loss
  t=risks['train_oracle']['test']-risks['test_oracle']['test']
  b=risks['adaptive']['test']-risks['sure']['test']
  etr=risks['sure']['train']-risks['train_oracle']['train']
  gd=a-etr
  total=risks['adaptive']['test']-risks['test_oracle']['test']
  checks['decomposition_error']=max(checks['decomposition_error'],abs(total-a-t-b),abs(total-etr-gd-t-b))
  def criterion(w):return float(np.mean((y-F@w)**2)+2*r['sigma']**2*df@w/n-r['sigma']**2)
  cd=criterion(ws)-criterion(wtr);fl=2*float((eps@F-r['sigma']**2*df)@(ws-wtr))/n
  checks['sure_identity_error']=max(checks['sure_identity_error'],abs(etr-cd-fl));assert etr>=-1e-7 and t>=-1e-7 and cd<=1.1e-7
  rows.append({k:r[k] for k in ['bank','n','signal','sigma','horizon']}|{'risks':risks,'components':{'signed_weight_effect_test':a,'target_transfer':t,'fallback_effect':b,'sure_excess_train':etr,'difference_in_train_test_gaps':gd,'total_adaptive_excess':total,'criterion_difference':cd,'selected_noise_fluctuation':fl},'weights':{'train_oracle':wtr.tolist(),'test_oracle':wte.tolist()}})
 z.close();assert checks['data_reconstruction']<1e-12 and checks['risk_reconstruction']<1e-12 and checks['decomposition_error']<1e-12 and checks['sure_identity_error']<1e-12
 groups=[]
 for s in ['all']+p['signals']:
  rr=rows if s=='all' else [r for r in rows if r['signal']==s];summary={'signal':s,'mean_risks':{name:{target:float(np.mean([r['risks'][name][target] for r in rr])) for target in ['train','test']} for name in rr[0]['risks']},'components':{},'negative_weight_effect_fraction':float(np.mean([r['components']['signed_weight_effect_test']< -1e-7 for r in rr]))}
  for k in rr[0]['components']:
   v=np.array([np.mean([r['components'][k] for r in rr if r['bank']==b]) for b in range(p['banks'])]);summary['components'][k]={'mean':float(v.mean()),'cluster_se':float(v.std(ddof=1)/np.sqrt(p['banks']))}
  groups.append(summary)
 cells=[]
 for key in sorted({(r['signal'],r['sigma'],r['horizon']) for r in rows}):
  rr=[r for r in rows if (r['signal'],r['sigma'],r['horizon'])==key];cells.append(dict(zip(['signal','sigma','horizon'],key))|{'components':{k:float(np.mean([r['components'][k] for r in rr])) for k in rr[0]['components']},'mean_test_risks':{k:float(np.mean([r['risks'][k]['test'] for r in rr])) for k in rr[0]['risks']}})
 out={'status':'Post-outcome descriptive diagnosis on existing seed20261018. Test-grid oracle uses truth and is optimistic; no deployable improvement or new significance claim. All methods share the same full-data candidates: target_transfer is NOT CV fold-to-full refit error. Signed weight effect may be negative.','identities':'Adaptive test excess = signed weight effect on test + target transfer + fallback effect = train excess + difference in train-test gaps + target transfer + fallback effect. Train excess = SURE criterion difference + selected noise fluctuation.','checks':checks,'groups':groups,'cells':cells,'records':rows};(P/'neural_path_risk_target_audit.json').write_text(json.dumps(out,indent=2))
 for g in groups:print(json.dumps(g))
 print('checks',checks)
if __name__=='__main__':run()
