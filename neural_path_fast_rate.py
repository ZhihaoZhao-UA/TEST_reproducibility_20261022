"""Progressive mixture of frozen neural predictions; no reuse/refit guarantee."""
import numpy as np

def progressive_weights(predictions, y, bound, prior=None):
    A=np.asarray(predictions,float);y=np.asarray(y,float)
    if A.ndim!=2 or y.shape!=(len(A),) or len(A)==0 or bound<=0:raise ValueError('Invalid dimensions or bound')
    if not np.isfinite(A).all() or not np.isfinite(y).all():raise ValueError('Nonfinite input')
    if np.max(abs(A))>bound or np.max(abs(y))>bound:raise ValueError('The bounded-loss guarantee requires bounded predictions and responses')
    n,m=A.shape;p=np.ones(m)/m if prior is None else np.asarray(prior,float)
    if p.shape!=(m,) or (p<=0).any() or not np.isclose(p.sum(),1):raise ValueError('Invalid prior')
    eta=1/(8*bound**2);logw=np.log(p);avg=np.zeros(m);loss=0.
    for row,target in zip(A,y):
        u=np.exp(logw-logw.max());w=u/u.sum();avg+=w;loss+=(target-row@w)**2
        logw-=eta*(target-row)**2
    return avg/n,{'online_loss':float(loss),'eta':eta,'expert_losses':np.sum((A-y[:,None])**2,axis=0).tolist()}

def check():
    rng=np.random.default_rng(20261004);worst=-np.inf
    for n in [1,8,64]:
        for m in [1,3,12]:
            for _ in range(20):
                A=rng.uniform(-2,2,(n,m));y=rng.uniform(-2,2,n);p=rng.dirichlet(np.ones(m));w,info=progressive_weights(A,y,2,p)
                rhs=min(np.array(info['expert_losses'])+np.log(1/p)/info['eta']);v=info['online_loss']-rhs;worst=max(worst,v)
                assert v<1e-10 and abs(w.sum()-1)<1e-12 and min(w)>=0
    # Exact one-expert case and guard against silently applying bounded theorem to Gaussian data.
    w,_=progressive_weights(np.array([[.2],[.1]]),np.array([.4,-.3]),1);assert w[0]==1
    try:progressive_weights(np.array([[0.]]),np.array([2.]),1)
    except ValueError:pass
    else:raise AssertionError('Unbounded input accepted')
    print('180 regret checks passed; maximum violation:',worst)
if __name__=='__main__':check()
