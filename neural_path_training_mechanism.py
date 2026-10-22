"""Exploratory mechanism check: fully trained width-one bias-free tanh network.
Fixed training design, independent aggregation labels, fixed late pair; no test tuning.
The 400-step endpoint is a diagnostic/extra-computation comparator, never a candidate.
"""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
P=Path(__file__).resolve().parent
protocol=dict(seed=14102028,replications=128,training_sites=129,domain=[-4.,4.],teacher=[4.,1.],initial=[4.02,1.01],learning_rates=[1.5,.5],training_noise_sds=[0.,.02],aggregation_noise_sd=.02,aggregation_sizes=[512,2048,8192],checkpoint_rule='ceil(5 log(n)), and the next iterate',training_replicates_per_site='ceil(n/129)',diagnostic_endpoint=400,gradient_control='Added after initial mechanism results; minimize the norm of the convex combination of the two training gradients',status='Exploratory mechanism experiment after analytic derivation; no claim of independent confirmation or CV superiority.')
(P/'neural_path_training_protocol.json').write_text(json.dumps(protocol,indent=2))
x=np.linspace(-4,4,129);mu=4*np.tanh(x)
xq,wq=leggauss(256);xq*=4;wq/=2;muq=4*np.tanh(xq)
def pred(th,xx):return th[0]*np.tanh(th[1]*xx)
def grad_hess(th,y):
 a,u=th;hh=np.tanh(u*x);ss=1-hh*hh;r=a*hh-y;J=np.c_[hh,a*x*ss]
 H=J.T@J/129;H[0,1]+=np.mean(r*x*ss);H[1,0]=H[0,1];H[1,1]+=np.mean(r*(-2*a*x*x*hh*ss))
 return J.T@r/129,H
def train(y,lr,k):
 th=np.array(protocol['initial']);saved={}
 for t in range(401):
  if t in [k,k+1,2*k+1,400]:saved[t]=th.copy()
  if t<400:th-=lr*grad_hess(th,y)[0]
 return saved
rng=np.random.default_rng(protocol['seed']);records=[]
for lr in protocol['learning_rates']:
 for sd in protocol['training_noise_sds']:
  for n in protocol['aggregation_sizes']:
   k=int(np.ceil(5*np.log(n)));r=int(np.ceil(n/129));m=129*r
   for rep in range(protocol['replications']):
    yy=mu+rng.normal(size=129)*sd/np.sqrt(r);saved=train(yy,lr,k)
    F=np.array([pred(saved[k],xq),pred(saved[k+1],xq)]).T;d=F[:,0]-F[:,1]
    denom=wq@(d*d);ow=float(np.clip((wq@((muq-F[:,1])*d))/denom,0,1))
    xx=rng.uniform(-4,4,n);yagg=4*np.tanh(xx)+rng.normal(scale=.02,size=n)
    FF=np.array([pred(saved[k],xx),pred(saved[k+1],xx)]).T;dd=FF[:,0]-FF[:,1]
    ww=float(np.clip(np.dot(dd,yagg-FF[:,1])/np.dot(dd,dd),0,1))
    sr=np.array([wq@((F[:,j]-muq)**2) for j in [0,1]])
    val_losses=np.mean((FF-yagg[:,None])**2,axis=0);sel=int(np.argmin(val_losses))
    g0=grad_hess(saved[k],yy)[0];g1=grad_hess(saved[k+1],yy)[0];gd=g0-g1
    gw=float(np.clip(-g1@gd/(gd@gd),0,1))
    means=dict(gradient_mixture=float(wq@((F[:,1]+gw*d-muq)**2)),aggregate=float(wq@((F[:,1]+ww*d-muq)**2)),oracle=float(wq@((F[:,1]+ow*d-muq)**2)),best_checkpoint=float(sr.min()),last=float(sr[1]),selection=float(sr[sel]),equal=float(wq@((F.mean(1)-muq)**2)),continued=float(wq@((pred(saved[2*k+1],xq)-muq)**2)),converged=float(wq@((pred(saved[400],xq)-muq)**2)))
    ev,V=np.linalg.eigh(grad_hess(saved[400],yy)[1]);rho=lr*ev[-1]-1;beta=abs(1-lr*ev[0]);err=saved[k]-saved[400];v=V[:,-1];perp=err-(err@v)*v
    records.append(dict(lr=lr,training_noise=sd,n=n,m=m,k=k,rep=rep,weight=ww,oracle_weight=ow,gradient_weight=gw,rho=float(rho),beta=float(beta),spectral_condition=bool(0<rho<1 and beta<rho*rho),gradient_at_endpoint=float(np.linalg.norm(grad_hess(saved[400],yy)[0])),perpendicular_over_squared_error=float(np.linalg.norm(perp)/max(err@err,1e-30)),risks=means))
summary=[]
for lr in protocol['learning_rates']:
 for sd in protocol['training_noise_sds']:
  for n in protocol['aggregation_sizes']:
   rs=[v for v in records if v['lr']==lr and v['training_noise']==sd and v['n']==n]
   keys=rs[0]['risks'];stats={key:float(np.mean([v['risks'][key] for v in rs])) for key in keys}
   gains=np.array([v['risks']['best_checkpoint']-v['risks']['aggregate'] for v in rs]);stats.update(mean_gain=float(gains.mean()),gain_se=float(gains.std(ddof=1)/np.sqrt(len(rs))),fraction_beating_best=float(np.mean(gains>0)),spectral_pass=sum(v['spectral_condition'] for v in rs),interior_oracles=sum(1e-8<v['oracle_weight']<1-1e-8 for v in rs))
   summary.append(dict(lr=lr,training_noise=sd,n=n,m=rs[0]['m'],k=rs[0]['k'],statistics=stats))
out=dict(protocol=protocol,summary=summary,records=records)
(P/'neural_path_training_mechanism.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
