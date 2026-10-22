"""Major-revision experiments. Python+NumPy; mode fixed or budget.
Protocol must exist beside this script. No outcome-dependent selection.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from neural_path_factor_train import train
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent
PROTOCOL=P/'neural_path_major_revision_protocol.json'
cfg=json.loads(PROTOCOL.read_text()); SHA=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
def signal(x,s):
 if s=='smooth':return np.sin(np.pi*x[:,0])+.4*x[:,1]
 if s=='oscillatory':return np.sin(3*np.pi*x[:,0])+.6*np.cos(2*np.pi*x[:,1])
 return np.sin(np.pi*x[:,0])+.6*np.sin(3*np.pi*x[:,1])
def init(rng):return (2*rng.normal(0,.7,(32,2)),rng.normal(0,.1,32),rng.uniform(-.5,.5,32))
def qp(A,y,tol=1e-8):return solve_qp(A,y,np.zeros(A.shape[1]),tol=tol,max_iter=200000)
def train_states(X,y,initial):
 W,b,a=[v.copy() for v in initial];h=len(a);n=len(y);root=np.sqrt(h);base=np.tanh(X@W.T+b)@a/root;states=[]
 for t in range(max(cfg['times'])+1):
  H=np.tanh(X@W.T+b);D=1-H*H;r=H@a/root-base-y
  if t in cfg['times']:states.append(tuple(v.copy() for v in (W,b,a)))
  if t==max(cfg['times']):break
  q=r[:,None]*a*D
  W,b,a=W-cfg['lr']*4*q.T@X/(n*root),b-cfg['lr']*q.sum(0)/(n*root),a-cfg['lr']*H.T@r/(n*root)
 return states
def predict(X,states):
 out=np.column_stack([np.tanh(X@W.T+b)@a/np.sqrt(len(a)) for W,b,a in states]);return out-out[:,0,None]
def geometry(X,states,F,k,mu):
 W,b,a=states[-1];h=len(a);root=np.sqrt(h);H=np.tanh(X@W.T+b);D=(1-H*H)*a/root
 J=np.c_[(D[:,:,None]*X[:,None,:]).reshape(len(X),-1),D,H/root]
 theta=np.array([np.concatenate([W.ravel(),b,a]) for W,b,a in states]);disp=theta-theta[-1];_,_,Vt=np.linalg.svd(disp,full_matrices=False);U=Vt[:k].T
 phi=J@U;C=phi.T@phi/len(X);eig,V=np.linalg.eigh(C);assert eig.min()>0
 lev=np.sum(((phi@V)/np.sqrt(eig))**2,axis=1);L=float(lev.max())
 coords=disp[:-1]@U;z=F[:,:-1]-F[:,-1,None];res=z-phi@coords.T
 delta_exact=float(np.max(np.sum(res**2,axis=1)))
 S=float(np.sqrt(1+np.max(np.sum(X*X,axis=1))));Ainf=max(float(np.max(abs(v[2]))) for v in states)
 G=np.sqrt(1+S*S*np.sum(a*a)/h);HH=(S+2*Ainf*S*S)/root
 q=np.linalg.norm(disp,axis=1);eta=np.linalg.norm(disp-(disp@U)@U.T,axis=1);bj=G*eta+.5*HH*q*q;delta_taylor=float(np.sum((bj[:-1]+bj[-1])**2))
 Bf=max(float(np.sum(abs(v[2]))) for v in states)/root+float(np.sum(abs(states[0][2])))/root;R0=float(np.max(abs(mu)))+Bf
 Sigma=z.T@z/len(X);vals,V=np.linalg.eigh(Sigma);checks=[]
 for n in cfg['fixed_dictionary']['aggregation_n']:
  lam=1/n;inv=(V/np.sqrt(np.maximum(vals,0)+lam))@V.T
  deff=float(np.trace(Sigma@inv@inv));K=float(np.max(np.sum((z@inv)**2,axis=1)))
  d_bound=k+delta_exact/lam;K_bound=L+(np.sqrt(L)+1)**2*delta_exact/lam
  checks.append(dict(n=n,lambda_value=lam,deff=deff,K=K,d_bound_exact_remainder=d_bound,K_bound_exact_remainder=K_bound))
  assert deff<=d_bound+1e-7 and K<=K_bound+1e-7
 assert delta_exact<=delta_taylor+1e-8
 return dict(k=k,tangent_leverage=L,delta2_exact=delta_exact,delta2_taylor=delta_taylor,
             min_tangent_eigenvalue=float(eig.min()),R0_bound=R0,G=float(G),H=float(HH),q=q.tolist(),eta=eta.tolist(),checks=checks)
def fixed():
 c=cfg['fixed_dictionary'];v=np.linspace(-1,1,c['population_grid_side']);x1,x2=np.meshgrid(v,v);X=np.c_[x1.ravel(),x2.ravel()];N=len(X);records=[];dicts=[];checkerr=0.;maxgap=0.;oraclemax=0.
 for b in range(c['banks']):
  for si,sig in enumerate(cfg['signals']):
   mu=signal(X,sig)
   for sj,sigma in enumerate(cfg['sigmas']):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],1,b,si,sj]));ti=rng.integers(N,size=c['training_n']);y=mu[ti]+sigma*rng.normal(size=len(ti));initial=init(rng)
    states=train_states(X[ti],y,initial);F=predict(X,states)
    if not dicts:
     _,_,check=train(X[ti],y,initial,cfg['times'],X_eval=X,first_layer_multiplier=4.)
     checkerr=float(np.max(abs(F-check)));assert checkerr<1e-11
    wo,og=qp(F,mu,c['oracle_gap']);oraclemax=max(oraclemax,og);riskstar=float(np.mean((F@wo-mu)**2));key=f'{b}_{sig}_{sigma}'
    dd=dict(id=key,bank=b,signal=sig,sigma=sigma,oracle_risk=riskstar,oracle_gap=og,
       geometry=[geometry(X,states,F,k,mu) for k in c['tangent_dimensions']]);dicts.append(dd)
    for rep in range(c['aggregation_repeats']):
     rr=np.random.default_rng(np.random.SeedSequence([cfg['seed'],2,b,si,sj,rep]));ids=rr.integers(N,size=max(c['aggregation_n']));yy=mu[ids]+sigma*rr.normal(size=len(ids))
     for n in c['aggregation_n']:
      w,g=qp(F[ids[:n]],yy[:n]);maxgap=max(maxgap,g);risk=float(np.mean((F@w-mu)**2));assert risk-riskstar>=-2*c['oracle_gap']
      records.append(dict(dictionary=key,bank=b,signal=sig,sigma=sigma,rep=rep,n=n,risk=risk,excess=risk-riskstar,gap=g))
    print('fixed',key,flush=True)
  (P/'neural_path_fixed_progress.json').write_text(json.dumps({'completed_banks':b+1,'dictionaries':len(dicts),'records':len(records)}))
 summary=[]
 for n in c['aggregation_n']:
  rows=[r for r in records if r['n']==n];av=np.mean([r['excess'] for r in rows]);banks=[np.mean([r['excess'] for r in rows if r['bank']==b]) for b in range(c['banks'])]
  # Conditional Monte Carlo SE: independent aggregation replicates within dictionaries.
  vars=[np.var([r['excess'] for r in rows if r['dictionary']==d['id']],ddof=1)/c['aggregation_repeats'] for d in dicts]
  summary.append(dict(n=n,mean_excess=float(av),n_times_excess=float(n*av),conditional_mc_se=float(np.sqrt(sum(vars))/len(dicts)),bank_se=float(np.std(banks,ddof=1)/np.sqrt(c['banks']))))
 out=dict(protocol_sha256=SHA,protocol=cfg,summary=summary,records=records,dictionaries=dicts,max_gap=maxgap,oracle_maxgap=oraclemax,training_replay_error=checkerr)
 (P/'neural_path_fixed_dictionary_results.json').write_text(json.dumps(out,indent=2));print('FIXED SUMMARY',json.dumps(summary),flush=True)
def budget():
 c=cfg['fresh_budget'];records=[];maxgap=0.;start=time.time()
 for bank in range(c['banks']):
  for N in c['total_n']:
   rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],3,bank,N]));X=rng.uniform(-1,1,(N,2));Xt=rng.uniform(-1,1,(c['test_n'],2));noise=rng.normal(size=N);initial=init(rng);foldids=np.arange(N)%3
   for sig in cfg['signals']:
    mu=signal(X,sig);mut=signal(Xt,sig)
    for sigma in cfg['sigmas']:
     y=mu+sigma*noise;_,_,full=train(X,y,initial,cfg['times'],X_eval=Xt,first_layer_multiplier=4.);folds=[];O=np.zeros((N,len(cfg['times'])))
     for j in range(3):
      vi=np.where(foldids==j)[0];ti=np.where(foldids!=j)[0];_,_,E=train(X[ti],y[ti],initial,cfg['times'],X_eval=np.r_[X[vi],Xt],first_layer_multiplier=4.);O[vi]=E[:len(vi)];folds.append((vi,E[len(vi):]))
     for T in c['horizons']:
      ix=np.where(np.array(cfg['times'])<=T)[0];late=np.where(np.array(cfg['times'])[ix]>=T/2)[0];ff=full[:,ix];oo=O[:,ix];w,g=qp(oo,y);maxgap=max(maxgap,g)
      pred={'full_cv_convex':ff@w,'full_cv_select':ff[:,np.argmin(np.mean((oo-y[:,None])**2,axis=0))], 'full_last':ff[:,-1],'full_equal':ff.mean(1),'full_late_equal':ff[:,late].mean(1)}
      risks={k:float(np.mean((v-mut)**2)) for k,v in pred.items()};splits=[]
      for vi,ft in folds:
       A=oo[vi];E=ft[:,ix];sw,g=qp(A,y[vi]);maxgap=max(maxgap,g);sp={'split_convex':E@sw,'split_select':E[:,np.argmin(np.mean((A-y[vi,None])**2,axis=0))],'split_last':E[:,-1],'split_equal':E.mean(1),'split_late_equal':E[:,late].mean(1)}
       splits.append({k:float(np.mean((v-mut)**2)) for k,v in sp.items()})
      risks.update({k:float(np.mean([v[k] for v in splits])) for k in splits[0]});records.append(dict(bank=bank,N=N,signal=sig,sigma=sigma,horizon=T,risks=risks))
  print('budget bank',bank+1,'seconds',round(time.time()-start),flush=True)
 summary={};comparisons={}
 for N in c['total_n']:
  rr=[r for r in records if r['N']==N];summary[N]={k:float(np.mean([r['risks'][k] for r in rr])) for k in c['methods']};comparisons[N]={}
  for k in c['methods']:
   if k=='split_convex':continue
   gains=np.array([np.mean([r['risks'][k]-r['risks']['split_convex'] for r in rr if r['bank']==b]) for b in range(c['banks'])]);comparisons[N][k]=dict(mean_gain=float(gains.mean()),bank_se=float(gains.std(ddof=1)/np.sqrt(c['banks'])))
 out=dict(protocol=cfg,protocol_sha256=SHA,records=records,summary=summary,comparisons=comparisons,max_gap=maxgap)
 (P/'neural_path_fresh_budget_results.json').write_text(json.dumps(out,indent=2));print('BUDGET SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fixed','budget']);args=ap.parse_args();globals()[args.mode]()
