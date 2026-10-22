"""Frozen final-round path-certificate audit; Python + NumPy only.
Run with all files alongside the previous major-revision scripts.
Only analysis bounds change; fitted weights retain every checkpoint.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from neural_path_major_revision import train_states,predict,signal,init,geometry,qp
P=Path(__file__).resolve().parent
protocol=P/'neural_path_final_protocol.json'; C=json.loads(protocol.read_text()); SHA=hashlib.sha256(protocol.read_bytes()).hexdigest()
def sup_first(l,r):
 lo=np.minimum(l,r);hi=np.maximum(l,r);near=np.where(lo>0,lo,np.where(hi<0,hi,0.));return 1-np.tanh(near)**2
def sup_second(l,r):
 lo=np.minimum(l,r);hi=np.maximum(l,r)
 def f(z):t=np.tanh(z);return 2*abs(t)*(1-t*t)
 v=np.maximum(f(lo),f(hi));a=np.arctanh(1/np.sqrt(3));hit=((lo<=a)&(hi>=a))|((lo<=-a)&(hi>=-a));return np.where(hit,4/(3*np.sqrt(3)),v)
def local(X,states,F,K):
 M=len(states);anchors=np.rint(np.linspace(0,M-1,K+1)).astype(int);assert len(set(anchors))==K+1
 theta=np.array([np.r_[W.ravel(),b,a] for W,b,a in states]);BB=np.zeros((M,K+1));bound=np.zeros_like(F)
 root=np.sqrt(len(states[0][2]));taus=[]
 for aa,bb in zip(anchors[:-1],anchors[1:]):
  Wa,ba,va=states[aa];Wb,bbias,vb=states[bb];za=X@Wa.T+ba;zb=X@Wb.T+bbias;dz=zb-za;da=vb-va
  curvature=(2*abs(da*dz)*sup_first(za,zb)+np.maximum(abs(va),abs(vb))*dz*dz*sup_second(za,zb)).sum(1)/root
  chord=theta[bb]-theta[aa];q=float(chord@chord)
  for j in range(aa,bb+1):
   tau=float(np.clip((theta[j]-theta[aa])@chord/q,0,1)) if q else 0.
   if j==aa:tau=0.
   if j==bb:tau=1.
   BB[j,np.where(anchors==aa)[0][0]]=1-tau;BB[j,np.where(anchors==bb)[0][0]]=tau
   Wj,bj,aj=states[j];zt=(1-tau)*za+tau*zb;at=(1-tau)*va+tau*vb;zj=X@Wj.T+bj
   mismatch=(abs(aj-at)*abs(np.tanh(zj))+abs(at)*sup_first(zt,zj)*abs(zj-zt)).sum(1)/root
   bound[:,j]=mismatch+.5*tau*(1-tau)*curvature
   taus.append([int(j),tau])
 for j in anchors:bound[:,j]=0.
 residual=F-F[:,anchors]@BB.T
 assert np.max(abs(residual)-bound)<1e-10
 phi=F[:,anchors[:-1]]-F[:,-1,None];U,s,_=np.linalg.svd(phi,full_matrices=False)
 L=float(len(X)*np.max(np.sum(U*U,axis=1)))*(1+1e-8)+1e-10
 delta=float(np.max(np.sum(bound[:,:-1]**2,axis=1)));exact=float(np.max(np.sum(residual[:,:-1]**2,axis=1)))
 z=F[:,:-1]-F[:,-1,None];assert np.max(abs(z-(phi@BB[:-1,:-1].T+residual[:,:-1])))<1e-10
 return dict(K=K,anchors=anchors.tolist(),L=L,delta2=delta,exact_delta2=exact,min_singular=float(s.min()),max_singular=float(s.max()),taus=taus)
def certificate(delta,L,k,n,sigma,R,p):
 best=None
 for lam in 10.**np.linspace(-6,1,71):
  d=min(float(p),k+delta/lam);leverage=L+(np.sqrt(L)+1)**2*delta/lam
  tt=np.log(2*p/C['delta_x']);e=np.sqrt(2*leverage*tt/n)+2*leverage*tt/(3*n)
  if e>=.5:continue
  val=2*lam+C['optimization_gap']+(R*R+sigma*sigma)*d/(n*C['delta_s']*(1-2*e))
  row=dict(value=float(val),lambda_value=float(lam),e=float(e),d=float(d),K=float(leverage))
  if best is None or val<best['value']:best=row
 return best

def main():
 v=np.linspace(-1,1,C['grid_side']);x,y=np.meshgrid(v,v);X=np.c_[x.ravel(),y.ravel()];N=len(X);dictionaries=[];fits=[];start=time.time();maxgap=0.
 for b in range(C['banks']):
  for si,sig in enumerate(C['signals']):
   mu=signal(X,sig)
   for sj,sigma in enumerate(C['sigmas']):
    rng=np.random.default_rng(np.random.SeedSequence([C['seed'],0,b,si,sj]));ids=rng.integers(N,size=C['training_n']);initial=init(rng);yy=mu[ids]+sigma*rng.normal(size=len(ids))
    states=train_states(X[ids],yy,initial);F=predict(X,states);w,og=qp(F,mu,C['oracle_gap']);riskstar=float(np.mean((F@w-mu)**2));rs=np.mean((F-mu[:,None])**2,axis=0)
    g=geometry(X,states,F,4,mu);localrows=[local(X,states,F,K) for K in C['piece_counts']]
    Rfinite=float(np.max(abs(F-mu[:,None])));Rp=g['R0_bound'];key=f'{b}_{sig}_{sigma}';trivial=float(rs.max()-riskstar)
    row=dict(id=key,bank=b,signal=sig,sigma=sigma,oracle_risk=riskstar,oracle_gap=og,initial_risk=float(rs[0]),trivial_excess=trivial,training_gain_fraction=float((rs[0]-riskstar)/rs[0]),global_geometry=g,R_finite=Rfinite,R_parameter=Rp,local=localrows)
    for l in localrows:
     l['ratio_to_global']=l['delta2']/g['delta2_taylor'];l['certificates']={}
     for n in C['certificate_n']:
      l['certificates'][n]={'parameter_envelope':certificate(l['delta2'],l['L'],l['K'],n,sigma,Rp,len(states)-1),'finite_envelope':certificate(l['delta2'],l['L'],l['K'],n,sigma,Rfinite,len(states)-1)}
    row['global_certificates']={n:certificate(g['delta2_taylor'],g['tangent_leverage'],4,n,sigma,Rp,len(states)-1) for n in C['certificate_n']}
    dictionaries.append(row)
    for n in C['certificate_n']:
     for rep in range(C['aggregation_repeats']):
      rr=np.random.default_rng(np.random.SeedSequence([C['seed'],1,b,si,sj,n,rep]));counts=rr.multinomial(n,np.full(N,1/N));noise=rr.normal(size=N);mask=counts>0
      A=F*np.sqrt(N*counts/n)[:,None];Y=np.sqrt(N/n)*(np.sqrt(counts)*mu+sigma*noise*mask)
      ww,gap=qp(A,Y);maxgap=max(maxgap,gap)
      risks={'convex':float(np.mean((F@ww-mu)**2)), 'selection':float(rs[np.argmin(np.mean((A-Y[:,None])**2,axis=0))]), 'last':float(rs[-1]),'equal':float(np.mean((F.mean(1)-mu)**2)),'late_equal':float(np.mean((F[:,-2:].mean(1)-mu)**2))}
      fits.append(dict(id=key,bank=b,n=n,rep=rep,excess=risks['convex']-riskstar,gap=gap,risks=risks))
    print(key,'seconds',round(time.time()-start),'ratio4',localrows[1]['ratio_to_global'],flush=True)
 summary={};gates={}
 for K in C['piece_counts']:
  lr=[next(v for v in d['local'] if v['K']==K) for d in dictionaries]
  summary[K]={'median_delta2':float(np.median([v['delta2'] for v in lr])),'median_exact_delta2':float(np.median([v['exact_delta2'] for v in lr])),'median_L':float(np.median([v['L'] for v in lr])),'median_ratio':float(np.median([v['ratio_to_global'] for v in lr])),'improvement_passes':sum(v['ratio_to_global']<=.1 for v in lr),'certificates':{}}
  for n in C['certificate_n']:
   summary[K]['certificates'][n]={}
   for env in ['parameter_envelope','finite_envelope']:
    vals=[v['certificates'][n][env] for v in lr];valid=[v['value'] for v in vals if v is not None];passed=sum(v is not None and v['value']<d['trivial_excess'] for v,d in zip(vals,dictionaries))
    summary[K]['certificates'][n][env]=dict(valid=len(valid),nonvacuous=passed,median=float(np.median(valid)) if valid else None)
 pred={n:{k:float(np.mean([r['risks'][k] for r in fits if r['n']==n])) for k in fits[0]['risks']} for n in C['certificate_n']}
 gates['improvement']=summary[4]['improvement_passes']/len(dictionaries)>=.8
 gates['nonvacuity']=summary[4]['certificates'][4096]['parameter_envelope']['nonvacuous']/len(dictionaries)>=.8
 gates['meaningful_training_count']=sum(d['training_gain_fraction']>=.1 for d in dictionaries)
 gates['meaningful_training']=gates['meaningful_training_count']/len(dictionaries)>=.8
 out=dict(protocol=C,protocol_sha256=SHA,dictionaries=dictionaries,records=fits,summary=summary,prediction_means=pred,gates=gates,max_gap=maxgap)
 (P/'neural_path_final_results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'gates':gates,'prediction_means':pred},indent=2),flush=True)
if __name__=='__main__':main()
