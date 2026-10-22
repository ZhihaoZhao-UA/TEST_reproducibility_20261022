"""Variance-sensitive population oracle audit, with finite-population diagnostics."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from neural_path_major_revision import train_states,predict,signal,init,qp
P=Path(__file__).resolve().parent;path=P/'neural_path_score_protocol.json';C=json.loads(path.read_text());SHA=hashlib.sha256(path.read_bytes()).hexdigest()
def val(V,n,delta,e,lam):
 return float(2*lam+1e-8+V/(n*delta*(1-2*e))) if e<.5 else None
def main():
 q=np.linspace(-1,1,C['grid_side']);a,b=np.meshgrid(q,q);X=np.c_[a.ravel(),b.ravel()];N=len(X);ds=[];records=[];start=time.time();maxgap=0.
 for bank in range(C['banks']):
  for si,sig in enumerate(C['signals']):
   mu=signal(X,sig)
   for sj,sigma in enumerate(C['sigmas']):
    rng=np.random.default_rng(np.random.SeedSequence([C['seed'],0,bank,si,sj]));ix=rng.integers(N,size=C['training_n']);initial=init(rng);y=mu[ix]+sigma*rng.normal(size=len(ix));states=train_states(X[ix],y,initial);F=predict(X,states);wo,og=qp(F,mu,1e-10);Rstar=float(np.mean((F@wo-mu)**2));rr=mu-F@wo
    z=F[:,:-1]-F[:,-1,None];Sigma=z.T@z/N;eig,V=np.linalg.eigh(Sigma);p=z.shape[1];Rp=float(np.max(abs(mu)))+(max(float(np.sum(abs(v[2]))) for v in states)+float(np.sum(abs(states[0][2]))))/np.sqrt(32);Rzero=float(np.mean(mu**2));trivial=float(np.max(np.mean((F-mu[:,None])**2,axis=0))-Rstar)
    did=f'{bank}_{sig}_{sigma}';di=dict(id=did,bank=bank,signal=sig,sigma=sigma,oracle_risk=Rstar,oracle_gap=og,trivial=trivial,parameter_envelope=Rp,zero_reference_risk=Rzero,quantities={})
    for n in C['aggregation_n']:
     lam=1/n;root=(V/np.sqrt(np.maximum(eig,0)+lam))@V.T;u=z@root;lev=np.sum(u*u,axis=1);d=float(np.mean(lev));K=float(lev.max());Ar=u.T@u/N;ur=u*rr[:,None];vr=max(0.,float(np.mean(np.sum(ur*ur,axis=1))-np.sum(ur.mean(0)**2)));exact_upper=(np.sqrt(vr)+np.sqrt(K*max(og,0)))**2+sigma*sigma*d
     vars={'old_parameter_envelope':(Rp*Rp+sigma*sigma)*d,'oracle_risk_score':K*Rstar+sigma*sigma*d,'zero_reference_score':K*Rzero+sigma*sigma*d,'numerical_oracle_variance_upper':exact_upper}
     assert exact_upper<=vars['oracle_risk_score']+1e-5
     t=np.log(2*p/.05);ep=float(np.sqrt(2*K*t/n)+2*K*t/(3*n));bounds={k:val(v,n,.05,ep,lam) for k,v in vars.items()}
     di['quantities'][n]=dict(lambda_value=lam,deff=d,K=K,residual_variance=vr,variance_upper=exact_upper,predicted_e=ep,variances=vars,bounds=bounds)
     for ni,noise in enumerate(C['aggregation_noise']):
      for rep in range(C['aggregation_repeats']):
       rng2=np.random.default_rng(np.random.SeedSequence([C['seed'],1,bank,si,sj,n,ni,rep]));ids=rng2.integers(N,size=n);eps=rng2.normal(size=n) if ni==0 else rng2.standard_t(3,size=n)/np.sqrt(3);yy=mu[ids]+sigma*eps;counts=np.bincount(ids,minlength=N);sums=np.bincount(ids,weights=yy,minlength=N);A=F*np.sqrt(N*counts/n)[:,None];Y=np.zeros(N);mask=counts>0;Y[mask]=np.sqrt(N/n)*sums[mask]/np.sqrt(counts[mask]);w,g=qp(A,Y);maxgap=max(maxgap,g);risk=float(np.mean((F@w-mu)**2));excess=risk-Rstar
       gram=(u.T*counts)@u/n-Ar;eo=float(np.max(abs(np.linalg.eigvalsh(gram))));bo=val(exact_upper,n,.1,eo,lam)
       records.append(dict(id=did,bank=bank,n=n,noise=noise,rep=rep,risk=risk,excess=excess,gap=g,observed_e=eo,observed_bound=bo,observed_nonvacuous=bo is not None and bo<trivial,observed_violation=bo is not None and excess>bo+1e-10))
    ds.append(di);print(did,'elapsed',round(time.time()-start),flush=True)
 summary={}
 for n in C['aggregation_n']:
  summary[n]={'bounds':{},'replications':{}}
  for name in C['bounds'][:-1]:
   vv=[d['quantities'][n]['bounds'][name] for d in ds];ok=[v for v in vv if v is not None];wins=sum(v is not None and v<d['trivial'] for v,d in zip(vv,ds));summary[n]['bounds'][name]=dict(finite=len(ok),nonvacuous=wins,median_finite=float(np.median(ok)) if ok else None)
  for noise in C['aggregation_noise']:
   rows=[r for r in records if r['n']==n and r['noise']==noise];summary[n]['replications'][noise]=dict(count=len(rows),mean_excess=float(np.mean([r['excess'] for r in rows])),finite=sum(r['observed_bound'] is not None for r in rows),nonvacuous=sum(r['observed_nonvacuous'] for r in rows),violations=sum(r['observed_violation'] for r in rows),median_finite=float(np.median([r['observed_bound'] for r in rows if r['observed_bound'] is not None])))
 out=dict(protocol=C,protocol_sha256=SHA,dictionaries=ds,records=records,summary=summary,max_gap=maxgap)
 (P/'neural_path_score_results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
