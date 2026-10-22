"""Independent numerical checks of complementarity identities and scale reparametrization."""
import json
from pathlib import Path
import numpy as np
from neural_path_average import train, solve_qp


def effective_train(X,y,initial,steps,input_scale,lr=.25):
    W,b,a=[p.copy() for p in initial];W*=input_scale
    n=len(y);norm=np.sqrt(len(a));base=np.tanh(X@W.T+b)@a/norm;pred=[]
    for t in range(max(steps)+1):
        H=np.tanh(X@W.T+b);P=1-H*H;f=H@a/norm-base;r=f-y
        if t in steps: pred.append(f.copy())
        if t==max(steps): break
        q=r[:,None]*a*P
        W,b,a=W-lr*input_scale**2*q.T@X/(n*norm),b-lr*q.sum(0)/(n*norm),a-lr*H.T@r/(n*norm)
    return np.array(pred).T


def run():
    rng=np.random.default_rng(20261011); worst=0.; max_violation=0.
    for case in range(100):
        F=rng.normal(size=(24,7));mu=rng.normal(size=24)
        if case%4==0:F[:,1]=F[:,0]
        if case%4==1:F[:]=F[:,0,None]
        if case%4==2:mu=F[:,2].copy()
        R=np.mean((F-mu[:,None])**2,axis=0);b=R.argmin();D=F-F[:,b,None]
        g=np.mean(D*D,axis=0);c=-(F[:,b]-mu)@D/len(mu);G=D.T@D/len(mu)
        w,gap=solve_qp(F,mu,np.zeros(F.shape[1]),tol=1e-9,max_iter=100000)
        H=R[b]-np.mean((F@w-mu)**2)
        err=abs(H-(2*c@w-w@G@w));worst=max(worst,float(err))
        lower=np.max(np.divide(np.maximum(c,0)**2,g,out=np.zeros_like(g),where=g>1e-24))
        max_violation=max(max_violation,float(lower-H),float(H-2*max(0,c.max())))
        assert err<1e-12 and lower-H<2e-9 and H<=2*max(0,c.max())+2e-9
        assert (H>1e-7)==(c.max()>1e-7)
    X=rng.uniform(-1,1,(24,2));y=rng.normal(size=24);init=(rng.normal(size=(16,2)),rng.normal(size=16),rng.normal(size=16));steps=[0,1,10,80,200]
    errors={}
    for s in [.5,1.,2.]:
        A,_=train(s*X,y,init,steps);B=effective_train(X,y,init,steps,s)
        errors[str(s)]=float(np.max(abs(A-B)));assert errors[str(s)]<1e-10
    result={'random_geometry_cases':100,'max_identity_error':worst,'max_bound_violation':max_violation,'scale_reparametrization_errors':errors,'note':'These are numerical checks, not replacements for the written proof or statistical validation.'}
    (Path(__file__).parent/'neural_path_geometry_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run()
