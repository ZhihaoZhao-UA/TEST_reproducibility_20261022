"""Fixed fresh-randomness validation after exploratory construction.
Oracle-designed coordinate map, common to all methods. Not a practical tuning rule.
"""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
protocol=dict(seed=14102064,replications=256,n=[516,2064,8256],amplitudes=[.2],checkpoint_pairs=[[2,3],[3,4]],noise_sd=.05,training_sites=129,rho=.4,eta_over_A=.7,step=1.4,status='Second-stage fresh draws, added after first-stage results to check pairs with both members trained; all six conditions retained',full_budget='Pool training and aggregation labels, rerun same updates, use training-gradient weights; no held-out weights',geometry='Population-designed fixed invertible coordinate map, common to every comparator')
(P/'neural_path_strong_trained_protocol.json').write_text(json.dumps(protocol,indent=2))
x=np.linspace(-4,4,129);Q=len(x);h=np.tanh(x);ss=1-h*h;J=np.c_[h,4*x*ss];G=J.T@J/Q;e,V=np.linalg.eigh(G);Gi=V@np.diag(1/np.sqrt(e))@V.T
T=np.zeros((Q,2,2));T[:,0,1]=T[:,1,0]=x*ss;T[:,1,1]=-8*x*x*h*ss
u=Gi[:,1];D=np.einsum('i,qij,j->q',u,T,u);kap=D-J@np.linalg.solve(G,J.T@D/Q)
C=np.mean(kap[:,None,None]*T,axis=0);ec,U=np.linalg.eigh(Gi@C@Gi);eps=.7/ec[-1];mu=4*h+eps*kap;H=G-eps*C
W=Gi@U@np.diag(np.sqrt(np.array([.75,1.])/(1-eps*ec)))
def fg(z):
 a,u=np.array([4.,1.])+W@z;hh=np.tanh(u*x);s=1-hh*hh
 return a*hh,np.c_[hh,a*x*s]@W
def grad(z,count,ysum):
 f,j=fg(z);return j.T@(count*f-ysum)/count.sum()
def path(amp,count,ysum):
 z=np.array([0.,amp]);out=[z.copy()]
 for _ in range(4):z=z-1.4*grad(z,count,ysum);out.append(z.copy())
 return out
def gradient_mix(z0,z1,count,ysum):
 g0,g1=grad(z0,count,ysum),grad(z1,count,ysum);d=g0-g1
 return float(np.clip(-g1@d/(d@d),0,1))
rng=np.random.default_rng(protocol['seed']);records=[]
for amp in protocol['amplitudes']:
 for k in [2,3]:
  for n in protocol['n']:
   for rep in range(protocol['replications']):
    r=n//Q;count=np.full(Q,r);ysum=count*mu+rng.normal(size=Q)*.05*np.sqrt(r)
    inds=rng.integers(Q,size=n);cnt=np.bincount(inds,minlength=Q);sy=np.bincount(inds,weights=mu[inds]+rng.normal(size=n)*.05,minlength=Q)
    pp=path(amp,count,ysum);f0,f1=fg(pp[k])[0],fg(pp[k+1])[0];d=f0-f1
    lw=float(np.clip(d@(sy-cnt*f1)/np.dot(cnt,d*d),0,1));gw=gradient_mix(pp[k],pp[k+1],count,ysum);ow=float(np.clip(np.mean(d*(mu-f1))/np.mean(d*d),0,1))
    full=path(amp,count+cnt,ysum+sy);fw=gradient_mix(full[k],full[k+1],count+cnt,ysum+sy);ff0,ff1=fg(full[k])[0],fg(full[k+1])[0]
    risk=lambda f:float(np.mean((f-mu)**2))
    risks=dict(ls=risk(f1+lw*d),gradient=risk(f1+gw*d),oracle=risk(f1+ow*d),equal=risk((f0+f1)/2),last=risk(f1),gradient_parameter=risk(fg(gw*pp[k]+(1-gw)*pp[k+1])[0]),full_gradient=risk(fw*ff0+(1-fw)*ff1),full_equal=risk((ff0+ff1)/2),full_last=risk(ff1))
    h2=float(np.mean(d*d));vscore=float(np.mean(d*d*((mu-(f1+ow*d))**2+.05**2)));k4=float(np.mean(d**4)/h2**2);gap=risks['gradient']-risks['oracle']
    threshold=max(160*vscore/(h2*gap),160*(k4-1)) if 0<ow<1 and gap>0 else None
    certificate=dict(H=h2,V=vscore,K4=k4,oracle_gain=gap,sufficient_n_for_95=threshold,passes=bool(threshold is not None and n>threshold))
    records.append(dict(certificate=certificate,amp=amp,k=k,n=n,rep=rep,ls_weight=lw,gradient_weight=gw,oracle_weight=ow,risks=risks))
summary=[]
for amp in protocol['amplitudes']:
 for k in [2,3]:
  for n in protocol['n']:
   rr=[z for z in records if (z['amp'],z['k'],z['n'])==(amp,k,n)];means={key:float(np.mean([z['risks'][key] for z in rr])) for key in rr[0]['risks']};diffs={}
   for key in ['gradient','full_gradient','equal','full_equal']:
    dd=np.array([z['risks'][key]-z['risks']['ls'] for z in rr]);se=float(dd.std(ddof=1)/np.sqrt(len(dd)));diffs[key]=dict(mean=float(dd.mean()),se=se,normal_approx_95=[float(dd.mean()-1.96*se),float(dd.mean()+1.96*se)],fraction_positive=float(np.mean(dd>0)))
   summary.append(dict(amp=amp,k=k,n=n,means=means,paired_gains=diffs))
A=float(W[:,-1]@G@W[:,-1]);eta=float(W[:,-1]@(eps*C)@W[:,-1]);B=eta*eta*(1-.4)**2/(4*A)
checks=dict(Hz=(W.T@H@W).tolist(),beta=.05,rho=.4,A=A,eta=eta,B=B,relative_extra_gain=B/(.4*eta),W=W.tolist(),baseline_risk=float(np.mean((mu-4*h)**2)))
assert len(records)==1536
for z in records:assert z['risks']['oracle']<=min(z['risks'][a] for a in ['ls','gradient','equal','last'])+1e-12
out=dict(protocol=protocol,geometry=checks,summary=summary,records=records)
(P/'neural_path_strong_trained_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(geometry=checks,summary=summary),indent=2))
