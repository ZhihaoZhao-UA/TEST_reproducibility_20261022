"""Exploratory deterministic checks of a misspecified finite-width path.
No new performance benchmark; no test-set tuning or fitted-weight claim.
All risk calculations use the same 129-point population as the training loss.
"""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
x=np.linspace(-4,4,129);Q=len(x);star=np.array([4.,1.]);h=np.tanh(x);s=1-h*h
J=np.c_[h,4*x*s];G=J.T@J/Q;ev,V=np.linalg.eigh(G);v0=V[:,-1]
curv=2*v0[0]*v0[1]*x*s-8*v0[1]**2*x*x*h*s
kap=curv-J@np.linalg.solve(G,J.T@curv/Q)
res=.1*kap;mu=4*h+res

def pred(t):return t[0]*np.tanh(t[1]*x)
def gh(t):
 a,u=t;hh=np.tanh(u*x);ss=1-hh*hh;rr=a*hh-mu;j=np.c_[hh,a*x*ss]
 H=j.T@j/Q;H[0,1]+=np.mean(rr*x*ss);H[1,0]=H[0,1];H[1,1]+=np.mean(rr*(-2*a*x*x*hh*ss))
 return j.T@rr/Q,H
H=gh(star)[1];ev,V=np.linalg.eigh(H);v=V[:,-1];rho=.8;lr=(1+rho)/ev[-1];beta=abs(1-lr*ev[0]);A=np.mean((J@v)**2)
D=2*v[0]*v[1]*x*s-8*v[1]**2*x*x*h*s;eta=np.mean(res*D)
w0=rho/(1+rho);wstar=w0+eta*(1-rho)/(2*A*(1+rho));B=eta**2*(1-rho)**2/(4*A)
# Start sufficiently near the stationary point for an asymptotic numerical audit.
theta=star+.01*v;path=[theta.copy()]
for t in range(66):theta=theta-lr*gh(theta)[0];path.append(theta.copy())
base=pred(star)
def difference(f):
 d=f-base
 return float(np.mean(d*d)-2*np.mean(res*d))
records=[]
for t in [4,8,12,16,20,24,28]:
 th0,th1=path[t:t+2];f0,f1=pred(th0),pred(th1);dd=f0-f1
 ow=float(np.clip(np.mean((mu-f1)*dd)/np.mean(dd*dd),0,1))
 g0,g1=gh(th0)[0],gh(th1)[0];dg=g0-g1;wg=float(np.clip(-g1@dg/(dg@dg),0,1))
 amp=float(v@(th0-star));scale=amp*amp
 ws=[0,w0,wstar,1];errors=[]
 for w in ws:
  theory=A*((1+rho)*w-rho)**2-eta*(rho*rho+(1-rho*rho)*w)
  errors.append(abs(difference(w*f0+(1-w)*f1)/scale-theory))
 records.append(dict(t=t,amplitude=amp,oracle_weight=ow,gradient_weight=wg,limiting_weight=wstar,cancellation_weight=w0,max_scaled_expansion_error=max(errors),pair_risk_excess=min(difference(f0),difference(f1)),oracle_risk_excess=difference(ow*f0+(1-ow)*f1),gradient_prediction_excess=difference(wg*f0+(1-wg)*f1),gradient_parameter_excess=difference(pred(wg*th0+(1-wg)*th1)),scaled_oracle_advantage=(difference(wg*f0+(1-wg)*f1)-difference(ow*f0+(1-ow)*f1))/scale))
assert np.linalg.norm(gh(star)[0])<1e-14 and ev.min()>0 and beta<rho*rho and 0<eta<A
assert all(z['pair_risk_excess']>0 and z['oracle_risk_excess']<0 for z in records)
assert abs(records[-1]['oracle_weight']-wstar)<2e-5
assert abs(records[-1]['scaled_oracle_advantage']/B-1)<.1
out=dict(protocol='Exploratory deterministic theorem check; not evidence of finite-sample fitted-weight superiority',population_size=Q,teacher='4 tanh(x) + 0.1 times normal projection of curvature',stationary_gradient_norm=float(np.linalg.norm(gh(star)[0])),hessian_eigenvalues=ev.tolist(),rho=rho,beta=beta,A=A,eta=eta,step_size=lr,initial_parameter=path[0].tolist(),asymptotic_weight=wstar,cancellation_weight=w0,asymptotic_scaled_advantage=B,records=records)
(P/'neural_path_curvature_separation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
