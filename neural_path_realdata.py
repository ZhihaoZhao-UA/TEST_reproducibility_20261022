"""Fixed-protocol UCI pilot. Run with Python+NumPy; --data points to downloaded ZIPs.
Dependencies: neural_path_factor_train.py and neural_path_average.py beside this file.
Reports repeated-split descriptive variability, not independent-study inference.
"""
import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
from neural_path_factor_train import train
from neural_path_average import solve_qp
P=dict(seed=20261005,repeats=20,test_fraction=.2,aggregation_fraction_of_development=1/3,
       checkpoints=[0,1,2,5,10,20,40,80,120,200,400,800],width=32,lr=.25,first_layer_multiplier=4,
       folds=3,tolerance=1e-8,selection='minimum validation MSE; ties earliest',
       preprocessing='X and y mean/std from each network training subset only',
       datasets={'airfoil':'https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip',
                 'yacht':'https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip'},
       interpretation='random-row interpolation; overlapping splits; no significance claims',
       frozen_before_outcomes=True)
OUT=Path(__file__).resolve().parent

def path(X,y,idx,ev,initial):
    mean=X[idx].mean(0); sd=X[idx].std(0); sd[sd==0]=1
    ym=y[idx].mean(); ys=y[idx].std()
    F,_,E=train((X[idx]-mean)/sd,(y[idx]-ym)/ys,initial,P['checkpoints'],
                X_eval=(X[ev]-mean)/sd,first_layer_multiplier=4)
    assert np.isfinite(E).all()
    return E*ys+ym

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args()
    protocol=OUT/'neural_path_realdata_protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text())==P
    else: protocol.write_text(json.dumps(P,indent=2))
    records=[]; hashes={}
    for di,name in enumerate(P['datasets']):
        blob=(a.data/(name+'.zip')).read_bytes();hashes[name]=hashlib.sha256(blob).hexdigest()
        z=zipfile.ZipFile(io.BytesIO(blob));member=next(x for x in z.namelist() if x.endswith(('.dat','.data')))
        data=np.loadtxt(io.BytesIO(z.read(member)));X,y=data[:,:-1],data[:,-1];N=len(y)
        for rep in range(P['repeats']):
            rng=np.random.default_rng(np.random.SeedSequence([P['seed'],di,rep]))
            order=rng.permutation(N); nt=int(np.ceil(N*.2));test=order[:nt];dev=order[nt:]
            folds=np.array_split(dev,3);agg=folds[0];fit=np.concatenate(folds[1:])
            init=(2*rng.normal(0,.7,(32,X.shape[1])),rng.normal(0,.1,32),rng.uniform(-.5,.5,32))
            # All weights and preprocessing are determined without test labels.
            E=path(X,y,fit,np.r_[agg,test],init); A,T=E[:len(agg)],E[len(agg):]
            scale=y[dev].std();w,gap=solve_qp(A/scale,y[agg]/scale,np.zeros(A.shape[1]),max_iter=200000)
            pred={'split_convex':T@w,'split_select':T[:,np.argmin(((A-y[agg,None])**2).mean(0))],
                  'split_last':T[:,-1],'split_equal':T.mean(1)}
            O=[];oy=[]
            for k,v in enumerate(folds):
                ti=np.concatenate([f for j,f in enumerate(folds) if j!=k])
                V=A if k==0 else path(X,y,ti,v,init)
                O.append(V);oy.append(y[v])
            O=np.vstack(O);oy=np.concatenate(oy)
            cw,cgap=solve_qp(O/scale,oy/scale,np.zeros(O.shape[1]),max_iter=200000)
            F=path(X,y,dev,test,init)
            pred.update(full_cv_convex=F@cw,full_cv_select=F[:,np.argmin(((O-oy[:,None])**2).mean(0))],
                        full_last=F[:,-1],full_equal=F.mean(1))
            xd=np.c_[np.ones(len(dev)),X[dev]];xt=np.c_[np.ones(len(test)),X[test]]
            pred['full_ols']=xt@np.linalg.lstsq(xd,y[dev],rcond=None)[0]
            mse={k:float(np.mean((v-y[test])**2)) for k,v in pred.items()}
            records.append(dict(dataset=name,rep=rep,n=N,d=X.shape[1],n_fit=len(fit),n_agg=len(agg),n_test=len(test),
                                mse=mse,rmse={k:float(np.sqrt(v)) for k,v in mse.items()},
                                normalized_mse={k:v/scale**2 for k,v in mse.items()},gap=gap,cv_gap=cgap,
                                weights=w.tolist(),cv_weights=cw.tolist()))
            print(name,rep+1,flush=True)
    summary={}
    for name in P['datasets']:
        rr=[r for r in records if r['dataset']==name];summary[name]={}
        for k in rr[0]['mse']:
            vals=np.array([r['rmse'][k] for r in rr]);diff=np.array([r['mse'][k]-r['mse']['split_convex'] for r in rr])
            summary[name][k]=dict(mean_rmse=float(vals.mean()),sd_rmse=float(vals.std(ddof=1)),
                  mean_mse_gain_to_split_convex=float(diff.mean()),split_convex_wins=int((diff>0).sum()))
    result=dict(protocol=P,sha256=hashes,records=records,summary=summary,
                max_gap=max(max(r['gap'],r['cv_gap']) for r in records))
    (OUT/'neural_path_realdata_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
