"""Recheck score audit totals and deterministic inequalities (Python+NumPy)."""
from pathlib import Path
import json,hashlib
import numpy as np
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent
r=json.loads((P/'neural_path_score_results.json').read_text());c=json.loads((P/'neural_path_score_protocol.json').read_text());assert r['protocol_sha256']==hashlib.sha256((P/'neural_path_score_protocol.json').read_bytes()).hexdigest();assert len(r['dictionaries'])==48 and len(r['records'])==4608
byid={v['id']:v for v in r['dictionaries']};counts={}
for n in c['aggregation_n']:
 for noise in c['aggregation_noise']:
  rows=[v for v in r['records'] if v['n']==n and v['noise']==noise];assert len(rows)==768
  counts[f'{n}_{noise}']=sum(v['observed_nonvacuous'] for v in rows)
  for v in rows:
   if v['observed_bound'] is not None:assert v['excess']+max(byid[v['id']]['oracle_gap'],0)<=v['observed_bound']+1e-9
assert r['max_gap']<1e-8
rng=np.random.default_rng(66791);bridge=[];variance=[];approx=[];finite=0
for k in range(100):
 N=50;M=6;n=512;F=rng.normal(size=(N,M));F[:,0]=0
 if k%2==0:F[:,-2]=F[:,-1]
 mu=rng.normal(size=N)*rng.uniform(.1,5);wo,og=solve_qp(F,mu,np.zeros(M),tol=1e-11,max_iter=200000);rr=mu-F@wo;R=float(rr@rr/N);z=F[:,:-1]-F[:,-1,None];S=z.T@z/N;ev,U=np.linalg.eigh(S);lam=1/n;root=(U/np.sqrt(np.maximum(ev,0)+lam))@U.T;u=z@root;lev=np.sum(u*u,axis=1);K=lev.max();d=lev.mean();sig=.8
 ur=u*rr[:,None];Vres=np.mean(np.sum(ur*ur,axis=1))-np.sum(ur.mean(0)**2);V=Vres+sig*sig*d;variance.append(V-(K*R+sig*sig*d))
 wp=.8*wo+.2*np.full(M,1/M);rp=mu-F@wp;gp=2*(F.T@F@wp/N-F.T@mu/N);eta=gp@wp-gp.min();urp=u*rp[:,None];vp=max(0.,float(np.mean(np.sum(urp*urp,axis=1))-np.sum(urp.mean(0)**2)));bound=(np.sqrt(vp)+np.sqrt(K*max(eta,0)))**2+sig*sig*d;approx.append(V-bound)
 ids=rng.integers(N,size=n);eps=sig*rng.standard_t(3,size=n)/np.sqrt(3);yy=mu[ids]+eps;w,g=solve_qp(F[ids],yy,np.zeros(M),tol=1e-9,max_iter=200000);e=float(max(abs(np.linalg.eigvalsh(u[ids].T@u[ids]/n-u.T@u/N))));s=(u[ids]*(rr[ids]+eps)[:,None]).mean(0)-ur.mean(0)
 if e<.5:
  finite+=1;rex=np.mean((F@w-mu)**2)-R;rhs=s@s/(1-2*e)+2*lam+g;bridge.append(rex-rhs)
assert max(variance)<1e-8 and max(approx)<1e-8 and max(bridge)<1e-8
out={'protocol_hash_matches':True,'dictionaries':48,'fits':4608,'nonvacuous_counts':counts,'random_cases':100,'finite_gram_cases':finite,'max_variance_inequality_violation':float(max(variance)),'max_oracle_variance_correction_violation':float(max(approx)),'max_deterministic_bridge_violation':float(max(bridge)),'max_optimization_gap':r['max_gap']}
(P/'neural_path_score_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
