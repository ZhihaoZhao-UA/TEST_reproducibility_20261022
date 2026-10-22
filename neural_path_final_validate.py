import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from neural_path_final_round import local,sup_second
from neural_path_average import solve_qp
from neural_path_major_revision import cfg
P=Path(__file__).resolve().parent;r=json.loads((P/'neural_path_final_results.json').read_text());c=json.loads((P/'neural_path_final_protocol.json').read_text())
assert r['protocol_sha256']==hashlib.sha256((P/'neural_path_final_protocol.json').read_bytes()).hexdigest()
assert cfg['times']==c['times'] and cfg['width']==c['width'] and cfg['lr']==c['learning_rate'] and cfg['first_layer_multiplier']==c['first_layer_multiplier']
assert len(r['dictionaries'])==48 and len(r['records'])==2304
rng=np.random.default_rng(99712);worst=-np.inf
for rep in range(100):
 X=rng.normal(size=(43,2));h=7;states=[(rng.normal(size=(h,2)),rng.normal(size=h),rng.normal(size=h)) for _ in range(7)]
 F=np.column_stack([np.tanh(X@W.T+b)@a/np.sqrt(h) for W,b,a in states]);local(X,states,F,3)
 l=rng.normal(size=(30,));rr=rng.normal(size=30);z=l[:,None]+(rr-l)[:,None]*np.linspace(0,1,101);t=np.tanh(z);err=np.max(2*abs(t)*(1-t*t)-sup_second(l,rr)[:,None]);worst=max(worst,err)
# Sufficient-statistic and literal-sample losses differ only by a weight-independent constant.
F=rng.normal(size=(17,5));ids=rng.integers(17,size=128);y=rng.normal(size=128);counts=np.bincount(ids,minlength=17);sums=np.bincount(ids,weights=y,minlength=17);A=F*np.sqrt(17*counts/128)[:,None];Y=np.zeros(17);mask=counts>0;Y[mask]=np.sqrt(17/128)*sums[mask]/np.sqrt(counts[mask])
Qerr=np.max(abs(A.T@A/17-F[ids].T@F[ids]/128));berr=np.max(abs(A.T@Y/17-F[ids].T@y/128));assert max(Qerr,berr)<1e-12
w1,g1=solve_qp(A,Y,np.zeros(5),tol=1e-10,max_iter=200000);w2,g2=solve_qp(F[ids],y,np.zeros(5),tol=1e-10,max_iter=200000);perr=np.max(abs(F@(w1-w2)));assert perr<1e-8
out=dict(protocol_hash_matches=True,complete_records=True,configuration_matches=True,random_piecewise_checks=100,interval_second_derivative_max_violation=float(worst),sufficient_statistic_Q_error=float(Qerr),sufficient_statistic_score_error=float(berr),compressed_solver_prediction_difference=float(perr),max_optimization_gap=r['max_gap'])
(P/'neural_path_final_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
