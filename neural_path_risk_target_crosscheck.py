from pathlib import Path
import json,numpy as np
from neural_path_average import solve_qp
P=Path(__file__).resolve().parent

def run():
 d=json.loads((P/'neural_path_n96_results.json').read_text());audit=json.loads((P/'neural_path_risk_target_audit.json').read_text());z=np.load(P/'neural_path_n96_predictions.npz');rows=[];gapmax=0.
 for r,a in zip(d['records'],audit['records']):
  for k in ['bank','signal','sigma','horizon']:assert r[k]==a[k]
  tag=f'b{r["bank"]}_n{r["n"]}_{r["signal"]}_s{r["sigma"]}';idx=[j for j,t in enumerate(d['protocol']['times']) if t<=r['horizon']];A=z[tag+'_Ft'][:,idx];mu=z[tag+'_mut'];N=len(mu);loss=np.empty(N);reference=[]
  for fold in range(2):
   vi=np.arange(N)[np.arange(N)%2==fold];ti=np.arange(N)[np.arange(N)%2!=fold];w,g=solve_qp(A[ti],mu[ti],np.zeros(len(idx)),tol=1e-8,max_iter=200000);gapmax=max(gapmax,g);loss[vi]=(A[vi]@w-mu[vi])**2;reference.append(w.tolist())
  cross=float(loss.mean());tr=a['risks']['train_oracle']['test'];full=a['risks']['test_oracle']['test'];rows.append({k:r[k] for k in ['bank','signal','sigma','horizon']}|{'cross_reference_risk':cross,'full_grid_oracle_risk':full,'cross_minus_full_oracle':cross-full,'target_transfer_crosschecked':tr-cross,'train_oracle_test_risk':tr,'cross_reference_weights':reference})
 z.close();groups=[]
 for s in ['all']+d['protocol']['signals']:
  rr=rows if s=='all' else [r for r in rows if r['signal']==s];g={'signal':s}
  for k in ['cross_reference_risk','full_grid_oracle_risk','cross_minus_full_oracle','target_transfer_crosschecked','train_oracle_test_risk']:
   v=np.array([np.mean([r[k] for r in rr if r['bank']==b]) for b in range(d['protocol']['banks'])]);g[k]={'mean':float(v.mean()),'cluster_se':float(v.std(ddof=1)/np.sqrt(len(v)))}
  groups.append(g)
 out={'status':'Post-outcome diagnostic: fit weights with true mean on512test inputs and evaluate on the other512, swap and average. It is not deployable and not a new confirmation. Cross-reference is not a full-grid oracle; transfer can be negative. Difference from full-grid oracle includes both finite evaluation optimism and smaller reference fitting sample.','max_qp_gap':gapmax,'groups':groups,'records':rows};(P/'neural_path_risk_target_crosscheck.json').write_text(json.dumps(out,indent=2))
 for g in groups:print(json.dumps(g))
if __name__=='__main__':run()
