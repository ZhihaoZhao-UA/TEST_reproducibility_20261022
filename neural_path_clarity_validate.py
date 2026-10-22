"""Algebra, response-derivative and stored-record checks for the calibration appendix."""
from pathlib import Path
import json,hashlib
import numpy as np
from neural_path_factor_train import train
P=Path(__file__).resolve().parent
r=json.loads((P/'neural_path_clarity_diagnostics.json').read_text())
for f,h in r['inputs_sha256'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h
assert len(r['ablation_records'])==48 and len(r['sensitivity_records'])==864
assert sum(2*x['lambda_value']>=x['trivial'] for x in r['ablation_records'])==46
rng=np.random.default_rng(7102026);max_id=0.;max_j=0.;max_endpoint=0.
for case in range(100):
 u=rng.normal(size=(37,5));res=rng.normal(size=37);v=rng.uniform(0,2,size=37);sig2=2.;ell=(u*u).sum(1);K=ell.max();ur=u*res[:,None];mean=ur.mean(0);V=(ur*ur).sum(1).mean()-mean@mean+(v*ell).mean();proxy=sig2*ell.mean()+K*np.mean(res**2);slack=np.mean((sig2-v)*ell)+np.mean((K-ell)*res**2)+mean@mean;max_id=max(max_id,abs(proxy-V-slack))
 a=rng.normal(size=(7,4));A=abs(a).max();tau=rng.uniform(size=(7,1));max_endpoint=max(max_endpoint,float(np.max(abs((1-tau)*a[0]+tau*a))-A))
for case in range(3):
 X=rng.normal(size=(7,2));y=rng.normal(size=7);initial=(rng.normal(size=(4,2)),rng.normal(size=4),rng.normal(size=4));times=[0,1,2,5]
 F,J=train(X,y,initial,times,directions=np.eye(7),first_layer_multiplier=4.)
 for j in range(7):
  dy=np.eye(7)[j]*1e-6;plus,_=train(X,y+dy,initial,times,first_layer_multiplier=4.);minus,_=train(X,y-dy,initial,times,first_layer_multiplier=4.);fd=(plus-minus)/(2e-6)
  max_j=max(max_j,float(np.max(abs(J[:,:,j]-fd.T))))
assert max_id<1e-10 and max_j<1e-7 and max_endpoint<=1e-12
out=dict(input_hashes_valid=True,ablation_dictionaries=48,sensitivity_calculations=864,additive_floor_obstructions=46,variance_identity_cases=100,max_variance_identity_error=max_id,max_endpoint_bound_violation=max_endpoint,jacobian_cases=3,max_response_jacobian_difference=max_j)
(P/'neural_path_clarity_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
