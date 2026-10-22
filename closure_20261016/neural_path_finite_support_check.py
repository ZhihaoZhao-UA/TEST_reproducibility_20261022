"""Numerical checks of S11 leverage and deterministic bridge; not a proof or coverage experiment."""
from pathlib import Path
import json
import numpy as np
rng=np.random.default_rng(2026101609);records=[]
for i in range(120):
 L=3+i%8;p=2+i%14;k=1+i%min(L,p)
 q=rng.uniform(.1,1,L);q/=q.sum()
 z=rng.normal(size=(L,k))@rng.normal(size=(k,p))
 if i%3==0:z[:,-1]=z[:,0]
 A=np.sqrt(q)[:,None]*z;U,s,V=np.linalg.svd(A,full_matrices=False);r=int(np.sum(s>1e-11*s[0]))
 u=U[:,:r]/np.sqrt(q)[:,None]
 lev=(u*u).sum(axis=1)
 assert np.all(q*lev<=1+1e-10)
 assert np.max(abs(u.T@(q[:,None]*u)-np.eye(r)))<1e-10
 # Direct verification of the completed-square bridge for arbitrary b, score and Gram event.
 h=rng.normal(size=(r,r));h=(h+h.T)/2;h*=.2/max(np.linalg.norm(h,2),1e-12)
 b=rng.normal(size=r);score=rng.normal(size=r);e=np.linalg.norm(h,2)
 expression=-b@b-2*b@h@b+2*b@score
 upper=score@score/(1-2*e)
 assert expression<=upper+1e-9
 records.append(dict(L=L,p=p,rank=r,max_weighted_leverage=float(max(q*lev)),bridge_slack=float(upper-expression)))
result=dict(scope=__doc__,cases=len(records),maximum_weighted_leverage=max(x['max_weighted_leverage'] for x in records),minimum_bridge_slack=min(x['bridge_slack'] for x in records),records=records)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='records'})
