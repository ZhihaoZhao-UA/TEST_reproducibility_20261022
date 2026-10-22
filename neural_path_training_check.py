"""Independent numerical checks for the local-training mechanism experiment."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent
scope={'__file__':str(P/'neural_path_training_mechanism.py')}
exec((P/'neural_path_training_mechanism.py').read_text().split('rng=np.random.default_rng',1)[0],scope)
d=json.loads((P/'neural_path_training_mechanism.json').read_text());records=d['records']
assert len(records)==1536
for z in records:
 r=z['risks'];assert 0<=z['weight']<=1 and 0<=z['gradient_weight']<=1
 assert r['oracle']<=min(r['best_checkpoint'],r['aggregate'],r['equal'],r['gradient_mixture'])+1e-16
for s in d['summary']:
 rs=[v for v in records if all(v[k]==s[k] for k in ['lr','training_noise','n'])]
 for k in rs[0]['risks']:assert abs(np.mean([v['risks'][k] for v in rs])-s['statistics'][k])<1e-20
rng=np.random.default_rng(3471);maxerr=0.;hesserr=0.
for lr in [1.5,.5]:
 for n in [512,2048,8192]:
  yy=scope['mu']+rng.normal(size=129)*.02/np.sqrt(np.ceil(n/129));k=int(np.ceil(5*np.log(n)));saved=scope['train'](yy,lr,k)
  risks=[]
  for nq in [256,512]:
   xq,wq=np.polynomial.legendre.leggauss(nq);xq*=4;wq/=2
   F=np.array([scope['pred'](saved[k],xq),scope['pred'](saved[k+1],xq)]).T
   risks.append([float(wq@((F@np.array([w,1-w])-4*np.tanh(xq))**2)) for w in [0,.25,.5,.75,1]])
  maxerr=max(maxerr,float(np.max(np.abs(np.array(risks[0])-risks[1]))))
  th=saved[k];H=scope['grad_hess'](th,yy)[1];eps=1e-5
  Hnum=np.column_stack([(scope['grad_hess'](th+eps*e,yy)[0]-scope['grad_hess'](th-eps*e,yy)[0])/(2*eps) for e in np.eye(2)])
  hesserr=max(hesserr,float(np.max(abs(H-Hnum))))
assert maxerr<1e-12 and hesserr<1e-7
checks=dict(records=1536,summary_means='verified',oracle_inequalities='verified',quadrature_max_abs_difference=maxerr,hessian_finite_difference_max_abs_difference=hesserr,scope='Checks numerical implementation; does not certify the local basin or establish novelty.')
(P/'neural_path_training_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
