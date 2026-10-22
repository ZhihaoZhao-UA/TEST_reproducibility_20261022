"""Post-outcome diagnostics of fixed dictionaries; no new predictor or tuning rule.
Replay stored dictionary seeds, change analysis quantities only, retain all cases.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from neural_path_major_revision import train_states, predict, signal, init, qp
P=Path(__file__).resolve().parent

def replay(C,old):
 q=np.linspace(-1,1,C['grid_side']);a,b=np.meshgrid(q,q);X=np.c_[a.ravel(),b.ravel()]
 for d in old:
  si=C['signals'].index(d['signal']);sj=C['sigmas'].index(d['sigma'])
  rng=np.random.default_rng(np.random.SeedSequence([C['seed'],0,d['bank'],si,sj]));ids=rng.integers(len(X),size=C['training_n']);initial=init(rng);mu=signal(X,d['signal']);y=mu[ids]+d['sigma']*rng.normal(size=len(ids));states=train_states(X[ids],y,initial);F=predict(X,states)
  yield d,X,mu,states,F

def quantities(F,mu,w,g,lam,sigma):
 z=F[:,:-1]-F[:,-1,None];ev,U=np.linalg.eigh(z.T@z/len(F));u=(z@U)/np.sqrt(np.maximum(ev,0)+lam);ell=(u*u).sum(1);deff=float(ell.mean());K=float(ell.max());ur=u*(mu-F@w)[:,None];vr=max(0,float((ur*ur).sum(1).mean()-np.sum(ur.mean(0)**2)));V=(np.sqrt(vr)+np.sqrt(K*max(g,0)))**2+sigma**2*deff
 return deff,K,float(V)
def bound(d,K,R,sigma,n,lam):
 e=float(np.sqrt(2*K*np.log(440)/n)+2*K*np.log(440)/(3*n))
 return None if e>=.5 else float(2*lam+1e-8+(R*R+sigma*sigma)*d/(n*.05*(1-2*e)))
def summarize(rows,key):
 vals=[r[key] for r in rows];finite=[v for v in vals if v is not None]
 return dict(finite=len(finite),nonvacuous=sum(v is not None and v<r['trivial'] for v,r in zip(vals,rows)),median_finite=float(np.median(finite)) if finite else None)

def main():
 final=json.loads((P/'neural_path_final_results.json').read_text());C=final['protocol'];ab=[]
 for old,X,mu,states,F in replay(C,final['dictionaries']):
  cert=next(v for v in old['local'] if v['K']==4)['certificates']['4096']['parameter_envelope'];assert cert is not None
  lam=cert['lambda_value'];deff,K,_=quantities(F,mu,np.ones(F.shape[1])/F.shape[1],0,lam,old['sigma'])
  Rp=old['R_parameter'];Rg=float(np.max(abs(F-mu[:,None])));assert abs(Rg-old['R_finite'])<1e-10
  vals={}
  for label,d,k in [('parameter',cert['d'],cert['K']),('exact',deff,K)]:
   for env,R in [('parameter',Rp),('grid',Rg)]:vals[label+'_'+env]=bound(d,k,R,old['sigma'],4096,lam)
  assert abs(vals['parameter_parameter']-cert['value'])<1e-9
  ab.append(dict(id=old['id'],trivial=old['trivial_excess'],lambda_value=lam,exact_K=K,cert_K=cert['K'],exact_d=deff,cert_d=cert['d'],R_parameter=Rp,R_grid=Rg,A_inf=max(float(abs(st[2]).max()) for st in states),**vals))
 score=json.loads((P/'neural_path_score_results.json').read_text());sens=[]
 for old,X,mu,states,F in replay(score['protocol'],score['dictionaries']):
  w,g=qp(F,mu,1e-10);R=float(np.mean((F@w-mu)**2));assert abs(R-old['oracle_risk'])<1e-11
  for n in [128,512,4096]:
   for label,lam in [(str(c)+'/n',c/n) for c in [.25,.5,1,2,4]]+[('sigma2',old['sigma']**2)]:
    d,K,V=quantities(F,mu,w,g,lam,old['sigma']);e=np.sqrt(2*K*np.log(440)/n)+2*K*np.log(440)/(3*n);b=None if e>=.5 else float(2*lam+1e-8+V/(n*.05*(1-2*e)))
    if label=='1/n':
     prev=old['quantities'][str(n)]['bounds']['numerical_oracle_variance_upper'];assert (b is None)==(prev is None)
     if b is not None:assert abs(b-prev)<1e-8
    sens.append(dict(id=old['id'],n=n,label=label,lam=lam,trivial=old['trivial'],bound=b))
 real=json.loads((P/'neural_path_realdata_results.json').read_text());weights={}
 for dataset in ['airfoil','yacht']:
  rows=[r for r in real['records'] if r['dataset']==dataset];weights[dataset]={}
  for key in ['weights','cv_weights']:
   W=np.array([r[key] for r in rows]);weights[dataset][key]=dict(last_mean=float(W[:,-1].mean()),last_min=float(W[:,-1].min()),last_max=float(W[:,-1].max()),near_last_vertex=int(np.sum(W[:,-1]>=1-1e-6)),mean_count_above_1e_6=float(np.sum(W>1e-6,axis=1).mean()),mean_effective_count=float(np.mean(1/np.sum(W*W,axis=1))))
  weights[dataset]['max_rmse_difference_from_split_last']=max(abs(r['rmse']['split_convex']-r['rmse']['split_last']) for r in rows)
 out=dict(status='Post-outcome diagnostic; reuses existing dictionary seeds; not independent confirmation; no weight-estimation changes',
 inputs_sha256={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['neural_path_final_results.json','neural_path_score_results.json','neural_path_realdata_results.json']},
 ablation_summary={k:summarize(ab,k) for k in ['parameter_parameter','parameter_grid','exact_parameter','exact_grid']},
 ablation_medians={k:float(np.median([r[k] for r in ab])) for k in ['lambda_value','exact_K','cert_K','exact_d','cert_d','R_parameter','R_grid','A_inf']},
 ablation_records=ab,sensitivity_summary={str(n):{l:summarize([r for r in sens if r['n']==n and r['label']==l],'bound') for l in ['0.25/n','0.5/n','1/n','2/n','4/n','sigma2']} for n in [128,512,4096]},sensitivity_records=sens,realdata_weights=weights)
 (P/'neural_path_clarity_diagnostics.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in out.items() if not k.endswith('records')},indent=2))
if __name__=='__main__':main()
