from pathlib import Path
import json,hashlib
import numpy as np
from neural_path_calibrated_analysis import tquantile,tcdf
from neural_path_average import solve_qp
from neural_path_stabilize import fit
P=Path(__file__).resolve().parent

def run():
 d=json.loads((P/'neural_path_predictability_results.json').read_text());p=d['protocol'];rr=d['records'];assert len(rr)==768;assert d['protocol_sha256']==hashlib.sha256((P/'neural_path_predictability_protocol.json').read_bytes()).hexdigest();z=np.load(P/'neural_path_predictability_predictions.npz');assert len(z.files)==8064;err=0.;checks=0
 def qp(A,y,pen):return solve_qp(A,y,pen,tol=1e-7,max_iter=100000)[0]
 for r in rr:
  tag=f'b{r["bank"]}_n{r["n"]}_{r["signal"]}_s{r["sigma"]}';idx=[j for j,t in enumerate(p['times']) if t<=r['horizon']];A=z[tag+'_F'][:,idx];At=z[tag+'_Ft'][:,idx];O=z[tag+'_O'][:,idx];y=z[tag+'_y'];mu=z[tag+'_mut'];n=len(y);m=len(idx);pen=r['sigma']**2*z[tag+'_df'][idx]/n;val=np.empty(n);cover=np.zeros(n,int)
  for j in range(3):
   key=tag+f'_fold{j}_';ti=z[key+'ti'];vi=z[key+'vi'];B=z[key+'F'][:,idx];df=z[key+'df'][idx];V=z[key+'V'][:,idx];assert not set(ti)&set(vi) and len(ti)+len(vi)==n;assert np.max(abs(O[vi]-V))<1e-12;cover[vi]+=1;val[vi]=V@qp(B,y[ti],r['sigma']**2*df/len(ti))
  assert np.all(cover==1);loss=float(np.mean((val-y)**2));null=float(np.mean(y*y));use=bool(null-loss>p['rho']*max(null-r['sigma']**2,0));assert use==r['use_sure'];err=max(err,abs(loss-r['validation_sure_loss']),abs(null-r['null_loss']))
  W={'sure':qp(A,y,pen),'sure_select':np.eye(m)[int(np.argmin(np.mean((A-y[:,None])**2,axis=0)+2*pen))],'cv3':qp(O,y,np.zeros(m)),'cv3_select':np.eye(m)[int(np.argmin(np.mean((O-y[:,None])**2,axis=0)))],'uniform':np.ones(m)/m,'last':np.eye(m)[-1]};W['cv3_ridge1']=fit(O,y,np.zeros(m),r['sigma'],'ridge',1)[0];W['adaptive']=W['sure'] if use else W['uniform']
  for k,w in W.items():
   err=max(err,float(np.max(abs(w-r['weights'][k]))),abs(float(np.mean((At@w-mu)**2))-r['risks'][k]));assert min(w)>=-1e-12 and abs(w.sum()-1)<1e-10
  # Diagnostic references, not additional primary hypotheses.
  r['diagnostic_zero']=float(np.mean(mu*mu));r['diagnostic_constant']=float(np.mean((y.mean()-mu)**2));r['branch_oracle']=min(r['risks']['sure'],r['risks']['uniform']);r['branch_selection_loss']=r['risks']['adaptive']-r['branch_oracle'];assert r['branch_selection_loss']>=-1e-12;checks+=1
  if checks%192==0:print('audited',checks,flush=True)
 assert err<1e-12;z.close();keys=sorted({(r['n'],r['signal'],r['sigma'],r['horizon']) for r in rr});crit=tquantile(1-.05/24,31);cells=[]
 for key in keys:
  rows=[r for r in rr if (r['n'],r['signal'],r['sigma'],r['horizon'])==key];v=np.array([r['risks']['cv3']-r['risks']['adaptive'] for r in rows]);se=float(v.std(ddof=1)/np.sqrt(32));cells.append(dict(zip(['n','signal','sigma','horizon'],key))|{'means':{k:float(np.mean([r['risks'][k] for r in rows])) for k in rows[0]['risks']},'gain':float(v.mean()),'se':se,'simultaneous_lower':float(v.mean()-crit*se),'sure_fraction':float(np.mean([r['use_sure'] for r in rows])),'zero':float(np.mean([r['diagnostic_zero'] for r in rows])),'constant':float(np.mean([r['diagnostic_constant'] for r in rows]))})
 v=np.array([np.mean([r['risks']['cv3']-r['risks']['adaptive'] for r in rr if r['bank']==b]) for b in range(32)]);mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(32));lower=mean-tquantile(.95,31)*se;wins=sum(r['gain']>1e-10 for r in cells);sigwins=sum(r['simultaneous_lower']>0 for r in cells)
 primary={'overall_gain':mean,'se':se,'one_sided95_lower':lower,'p_zero':1-tcdf(mean/se,31),'p_margin0.001':1-tcdf((mean-.001)/se,31),'mean_winning_cells':wins,'simultaneously_positive_cells':sigwins,'passes_prespecified_success':wins>=16 and lower>.001,'bank_gains':v.tolist(),'positive_banks':int(sum(v>0)),'leave_one_out_mean_range':[float(min((v.sum()-v)/31)),float(max((v.sum()-v)/31))]}
 groups=[]
 for s in ['all']+p['signals']:
  rows=rr if s=='all' else [r for r in rr if r['signal']==s];cs=cells if s=='all' else [r for r in cells if r['signal']==s];groups.append({'signal':s,'means':{k:float(np.mean([r['risks'][k] for r in rows])) for k in rows[0]['risks']},'wins_by_baseline':{k:sum(c['means'][k]-c['means']['adaptive']>1e-10 for c in cs) for k in ['cv3','cv3_ridge1','cv3_select','sure','sure_select','last','uniform']},'ties_by_baseline':{k:sum(abs(c['means'][k]-c['means']['adaptive'])<=1e-10 for c in cs) for k in ['cv3','cv3_ridge1','cv3_select','sure','sure_select','last','uniform']},'sure_fraction':float(np.mean([r['use_sure'] for r in rows])),'diagnostic_zero':float(np.mean([r['diagnostic_zero'] for r in rows])),'diagnostic_constant':float(np.mean([r['diagnostic_constant'] for r in rows])),'branch_oracle_mean':float(np.mean([r['branch_oracle'] for r in rows])),'branch_selection_loss':float(np.mean([r['branch_selection_loss'] for r in rows])),'branch_wrong_fraction':float(np.mean([r['branch_selection_loss']>1e-10 for r in rows]))})
 out={'primary':primary,'groups':groups,'cells':cells,'audit_records':checks,'reconstruction_error':err,'critical_cell_t':crit,'notes':'Exploration was24rules on oldseed20261016. This newseed20261017 kept all24cells. Counts of mean wins are descriptive; simultaneous lowerbounds separately qualify any stronger cellwise claim. Approximate t inference; other baselines descriptive.'};(P/'neural_path_predictability_analysis.json').write_text(json.dumps(out,indent=2))
 lines=['新种子确认：可预测性驱动的SURE/等权规则','整体主要检验：'+json.dumps(primary),'','signal | adaptive | CV3 | ridgeCV1 | SURE | SUREselect | last | uniform | CVmeanwin | SUREbranch']
 for r in groups:lines.append(r['signal']+' | '+' | '.join(f"{r['means'][k]:.6f}" for k in ['adaptive','cv3','cv3_ridge1','sure','sure_select','last','uniform'])+f" | {r['wins_by_baseline']['cv3']} | {r['sure_fraction']:.3f}")
 lines+=['','全部24条件（正gain表示超过CV；lower为24条件同时校正下界）','n | signal | sigma | T | adaptive | CV3 | gain | SE | simultaneous lower | sure fraction']
 for r in cells:lines.append(f"{r['n']} | {r['signal']} | {r['sigma']} | {r['horizon']} | {r['means']['adaptive']:.6f} | {r['means']['cv3']:.6f} | {r['gain']:+.6f} | {r['se']:.6f} | {r['simultaneous_lower']:+.6f} | {r['sure_fraction']:.3f}")
 (P/'neural_path_predictability_report.txt').write_text('\n'.join(lines));print(json.dumps(primary));print(json.dumps(groups));print('AUDIT',checks,err)
if __name__=='__main__':run()
