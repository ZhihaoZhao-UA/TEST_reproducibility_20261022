"""Deterministic checks for S10; designed parameter paths, not GD or predictive trials."""
from pathlib import Path
import numpy as np,json
rng=np.random.default_rng(14102026)
out={'seed':14102026,'scope':'Algebra and finite-population certification; no new prediction experiment.'}
worst_d=worst_K=worst_ell=0.
for rep in range(120):
 p=int(rng.integers(2,14));k=int(rng.integers(1,4));N=51
 phi=rng.normal(size=(N,k));B=rng.normal(size=(p,k));B[:,0]*=10.**(-(rep%7))
 if k>1 and rep%3==0:B[:,-1]=B[:,0]
 A=rng.normal(size=(p,p));Gamma=A@A.T+.1*np.eye(p);C=np.linalg.cholesky(Gamma)
 e=rng.normal(size=(N,p));e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1);e=e@C.T
 z=phi@B.T+e;Sigma=z.T@z/N;inv=np.linalg.inv(Sigma+Gamma)
 cf=phi.T@phi/N;L=np.einsum('ni,ij,nj->n',phi,np.linalg.pinv(cf),phi).max()
 d=np.trace(Sigma@inv);K=np.einsum('ni,ij,nj->n',z,inv,z).max()
 worst_d=max(worst_d,d-k-1);worst_K=max(worst_K,K-L-(np.sqrt(L)+1)**2)
 # Maximum quadratic diameter is attained at a pair of simplex vertices.
 Q=np.vstack([np.eye(p),np.zeros(p)]);diff=Q[:,None,:]-Q[None,:,:];D=np.einsum('abi,ij,abj->ab',diff,Gamma,diff).max()
 w=rng.dirichlet(np.ones(p+1),size=40);v=rng.dirichlet(np.ones(p+1),size=40);a=(w-v)[:,:p]
 assert np.max(np.einsum('ni,ij,nj->n',a,Gamma,a))<=D+1e-9
out['random_matrix_cases']=120;out['max_dimension_violation']=worst_d;out['max_leverage_violation']=worst_K
h=4;X=np.c_[np.linspace(-1,1,401),np.ones(401)];S=np.sqrt(2.)
a=rng.uniform(-.5,.5,h);u=rng.normal(scale=.4,size=(h,2));direction=rng.normal(size=3*h);direction/=np.linalg.norm(direction);da=direction[:h];du=direction[h:].reshape(h,2)
s=X@u.T;b=X@du.T;th=np.tanh(s);sech=1-th**2
f0=th@a/np.sqrt(h);phi=(th@da+(sech*b)@a)/np.sqrt(h)
psi=(2*(sech*b)@da+(-2*th*sech*b*b)@a)/(2*np.sqrt(h))
L=float(np.max(phi**2)/np.mean(phi**2));n=1000000;T=n**(-.25)
Ainf=max(np.max(abs(a-T*da)),np.max(abs(a+T*da)))
H=(S+2*Ainf*S*S)/np.sqrt(h);J=(6*S*S+2*Ainf*S**3)/np.sqrt(h)
records=[];max_taylor=0.;max_residual_ellipsoid=0.
for p in [8,16,32,64,128]:
 t=np.arange(1,p+1)*T/p;c=t*t;delta=J*abs(t)**3/6
 F=np.array([np.tanh(X@(u+tt*du).T)@(a+tt*da)/np.sqrt(h)-f0 for tt in t]).T
 e=F-phi[:,None]*t;third=e-psi[:,None]*c
 max_taylor=max(max_taylor,float(np.max(abs(third)-delta)))
 Gamma=H*H/2*np.outer(c,c)+2*p*np.diag(delta**2)+np.eye(p)/n
 ell=np.einsum('ni,ij,nj->n',e,np.linalg.inv(Gamma),e);max_residual_ellipsoid=max(max_residual_ellipsoid,float(max(ell)))
 Sigma=F.T@F/len(X);inv=np.linalg.inv(Sigma+Gamma);d=float(np.trace(Sigma@inv));K=float(np.einsum('ni,ij,nj->n',F,inv,F).max())
 q=np.vstack([np.eye(p),np.zeros(p)]);diff=q[:,None,:]-q[None,:,:];D=float(np.einsum('abi,ij,abj->ab',diff,Gamma,diff).max())
 upper=H*H*T**4/2+p*J*J*T**6/9+2/n
 oldDelta=H*H/4*np.sum(t**4);old=2*max(oldDelta,1/n)
 assert d<=2+1e-8 and K<=L+(np.sqrt(L)+1)**2+1e-8 and D<=upper+1e-10 and max(ell)<=1+1e-8
 records.append(dict(M=p+1,scalar_additive_allowance=old,matrix_diameter=D,matrix_upper_allowance=upper,exact_effective_dimension=d,exact_leverage=K,scalar_to_matrix_ratio=old/D))
assert max_taylor<1e-10 and worst_d<1e-8 and worst_K<1e-8
out.update(n=n,T=T,H=float(H),J=float(J),tangent_leverage=L,max_third_order_violation=max_taylor,max_residual_ellipsoid=max_residual_ellipsoid,records=records)
# Independent deterministic population-bridge checks, including misspecification.
brng=np.random.default_rng(14102027);max_bridge=0.;bridge_count=0
for case in range(80):
 pp=5;NN=101;nn=3000;FF=brng.normal(size=(NN,pp));SS=FF.T@FF/NN
 AA=brng.normal(size=(pp,pp));GG=AA@AA.T+.5*np.eye(pp)
 vals,vecs=np.linalg.eigh(SS+GG);W=(vecs/np.sqrt(vals))@vecs.T
 mu=brng.normal(size=NN);ws=brng.dirichlet(np.ones(pp+1));wh=brng.dirichlet(np.ones(pp+1));rr=mu-FF@ws[:pp]
 ii=brng.integers(NN,size=nn);FN=FF[ii];Y=mu[ii]+brng.normal(size=nn)
 EE=np.linalg.norm(W@(FN.T@FN/nn-SS)@W,2)
 if EE>=.5:continue
 grad0=2*FN.T@(FN@wh[:pp]-Y)/nn;grad=np.r_[grad0,0.];gap=float(grad@wh-grad.min())
 sc=W@(FN.T@(Y-FN@ws[:pp])/nn-FF.T@rr/NN)
 qq=np.vstack([np.eye(pp),np.zeros(pp)]);dif=qq[:,None,:]-qq[None,:,:];diam=np.einsum('abi,ij,abj->ab',dif,GG,dif).max()
 excess=np.mean((FF@wh[:pp]-mu)**2)-np.mean(rr**2)
 max_bridge=max(max_bridge,float(excess-diam-gap-sc@sc/(1-2*EE)));bridge_count+=1
assert bridge_count==80 and max_bridge<1e-9
out.update(bridge_cases=bridge_count,max_bridge_violation=max_bridge)
aa=np.tanh(1.);ss=1-aa*aa;Phi=np.array([[1/np.sqrt(2),.5],[(aa+ss)/np.sqrt(2),ss*(1-aa)/2]])
separation=[]
for qq in [1e-2,1e-3,1e-4,1e-5]:
 probs=np.array([1-qq,qq]);CC=Phi.T@(Phi*probs[:,None]);joint=np.einsum('ni,ij,nj->n',Phi,np.linalg.inv(CC),Phi)
 tangent=Phi[:,0]**2/(probs@Phi[:,0]**2)
 assert max(tangent)<=25/16+1e-10 and np.allclose(joint,1/probs,rtol=1e-8)
 separation.append(dict(q=qq,tangent_leverage=float(max(tangent)),joint_leverage=float(max(joint)),joint_exact=1/qq))
out['tangent_vs_joint_separation']=separation
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
