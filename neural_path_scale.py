"""Frozen equal-label-budget and candidate-complementarity experiment.
Full data, holdout, holdout-refit, and 2fold OOF-refit methods; NumPy only.
"""
import hashlib
import json
import time
from itertools import product
from pathlib import Path
import numpy as np
from neural_path_average import train, solve_qp
from neural_path_experiment_matrix import signal as old_signal
from neural_path_equal_budget import cluster_stats

def signal(X,kind):
    return np.sin(np.pi*X[:,0])+.6*np.sin(3*np.pi*X[:,1]) if kind=="mixed" else old_signal(X,kind)

ROOT = Path(__file__).resolve().parent


def run():
    pb = (ROOT/'neural_path_scale_protocol.json').read_bytes()
    p = json.loads(pb)
    N, half, nz = p['total_labels'], p['split_labels'][0], p['n_test_grid']
    steps, lr = p['checkpoints'], p['learning_rate']
    records, timings = [], []
    checks = {'max_qp_gap': 0., 'jvp_fd_error': 0., 'full_prediction_difference': 0., 'fold_training_prediction_difference': 0.}
    start, completed = time.perf_counter(), 0
    total = p['independent_design_and_initialization_banks']*p['noise_replicates_per_bank']*len(p['signals'])*len(p['widths'])*len(p['noise_sd'])
    for bank in range(p['independent_design_and_initialization_banks']):
        drng = np.random.default_rng(np.random.SeedSequence([p['seed'], 1, bank]))
        X, Xt = drng.uniform(-1, 1, (N, 2)), drng.uniform(-1, 1, (nz, 2))
        Xe = np.vstack([X, Xt])
        initial_bank = []
        for k in range(max(p['initializations_per_pool'])):
            irng = np.random.default_rng(np.random.SeedSequence([p['seed'], 2, bank, k]))
            hm = max(p['widths'])
            initial_bank.append((irng.normal(0, .7, (hm, 2)), irng.normal(0, .1, hm), irng.uniform(-.5,.5,hm)))
        for noise_id in range(p['noise_replicates_per_bank']):
            nrng = np.random.default_rng(np.random.SeedSequence([p['seed'], 3, bank, noise_id]))
            eps0 = nrng.normal(size=N)
            for kind,h,sigma in product(p['signals'], p['widths'], p['noise_sd']):
                mu, mut = signal(X, kind), signal(Xt, kind)
                y = mu+sigma*eps0
                paths = []
                for k,(init_id,scale) in enumerate(p["path_specs"]):
                    init=initial_bank[init_id]
                    Xs,Xts,Xes=scale*X,scale*Xt,scale*Xe
                    initial = tuple(a[:h].copy() for a in init)
                    t0=time.perf_counter()
                    F,D,Ft = train(Xs,y,initial,steps,lr=lr,directions=np.eye(N),X_eval=Xts)
                    full_jvp_time=time.perf_counter()-t0
                    t0=time.perf_counter()
                    Fplain,_,Ftplain = train(Xs,y,initial,steps,lr=lr,X_eval=Xts)
                    plain_time=time.perf_counter()-t0
                    err=max(float(np.max(abs(F-Fplain))),float(np.max(abs(Ft-Ftplain))))
                    checks['full_prediction_difference']=max(checks['full_prediction_difference'],err)
                    assert err < 1e-10
                    t0=time.perf_counter()
                    Fa,_,Ea=train(Xs[:half],y[:half],initial,steps,lr=lr,X_eval=Xes)
                    Fb,_,Eb=train(Xs[half:],y[half:],initial,steps,lr=lr,X_eval=Xes)
                    split_time=time.perf_counter()-t0
                    fold_error=max(float(np.max(abs(Fa-Ea[:half]))),float(np.max(abs(Fb-Eb[half:N]))))
                    checks['fold_training_prediction_difference']=max(checks['fold_training_prediction_difference'],fold_error)
                    assert fold_error < 1e-10
                    # First half's OOF predictions come from the second-half fit.
                    oof=np.vstack([Eb[:half],Ea[half:N]])
                    if completed==0:
                        direction=np.random.default_rng(p['seed']+1).normal(size=N)
                        delta=1e-5
                        ap,_=train(Xs,y+delta*direction,initial,steps,lr=lr)
                        am,_=train(Xs,y-delta*direction,initial,steps,lr=lr)
                        checks['jvp_fd_error']=max(checks['jvp_fd_error'],float(np.max(abs((ap-am)/(2*delta)-np.einsum('mij,j->im',D,direction)))))
                        assert checks['jvp_fd_error'] < 1e-6
                    paths.append({'F':F,'Ft':Ft,'df':np.trace(D,axis1=1,axis2=2),'val':Ea[half:N],
                                  'split_test':Ea[N:],'oof':oof})
                    timings.append({'bank':bank,'noise':noise_id,'signal':kind,'width':h,'sigma':sigma,'initialization':init_id,'scale':scale,
                                    'full_jvp_seconds':full_jvp_time,'full_prediction_only_seconds':plain_time,'two_half_fits_seconds':split_time})
                for horizon,pool in product(p['horizons_steps'],p['pools']):
                    path_ids=p['pools'][pool]
                    idx=[(1,0)]+[(k,j) for k in path_ids for j,t in enumerate(steps) if 0<t<=horizon]
                    arrays={key:np.column_stack([paths[k][key][:,j] for k,j in idx]) for key in ['F','Ft','val','split_test','oof']}
                    dfs=np.array([paths[k]['df'][j] for k,j in idx])
                    m=len(idx)
                    def qp(A,target,penalty):
                        w,gap=solve_qp(A,target,penalty,tol=p['qp']['frank_wolfe_gap_tolerance'],max_iter=p['qp']['maximum_iterations'])
                        checks['max_qp_gap']=max(checks['max_qp_gap'],float(gap))
                        return w
                    def choose(A,target,penalty=None):
                        risk=np.mean((A-target[:,None])**2,axis=0)
                        if penalty is not None: risk=risk+penalty
                        return np.eye(m)[int(np.argmin(risk))]
                    F,Ft=arrays['F'],arrays['Ft']
                    ws=qp(F,y,sigma**2*dfs/N)
                    wsingle=choose(F,y,2*sigma**2*dfs/N)
                    wvs=choose(arrays['val'],y[half:])
                    wva=qp(arrays['val'],y[half:],np.zeros(m))
                    wcs=choose(arrays['oof'],y)
                    wca=qp(arrays['oof'],y,np.zeros(m))
                    we=np.array([float(steps[j]==horizon)/len(path_ids) for k,j in idx])
                    assert abs(we.sum()-1)<1e-12
                    weights={'sure_average_full':ws,'sure_select_full':wsingle,
                             'validation_select_holdout':wvs,'validation_stack_holdout':wva,
                             'validation_select_refit':wvs,'validation_stack_refit':wva,
                             'cv_select_refit':wcs,'cv_stack_refit':wca,'uniform_final_full':we}
                    risks={}
                    for name,w in weights.items():
                        A=arrays['split_test'] if name.endswith('_holdout') else Ft
                        risks[name]=float(np.mean((A@w-mut)**2))
                    # Oracles below never enter implementable weight selection.
                    wo=qp(Ft,mut,np.zeros(m)); wos=choose(Ft,mut)
                    wosp=qp(arrays['split_test'],mut,np.zeros(m))
                    risks['oracle_full_test_grid_single']=float(np.mean((Ft@wos-mut)**2))
                    risks['oracle_full_test_grid_convex']=float(np.mean((Ft@wo-mut)**2))
                    risks['oracle_split_test_grid_convex']=float(np.mean((arrays['split_test']@wosp-mut)**2))
                    head=risks['oracle_full_test_grid_single']-risks['oracle_full_test_grid_convex']
                    assert head >= -2e-7
                    full_train={name:float(np.mean((F@w-mu)**2)) for name,w in weights.items() if not name.endswith('_holdout')}
                    z=(y-mu)@F-sigma**2*dfs
                    assert full_train['sure_average_full']-min(full_train.values()) <=2*np.ptp(z)/N+p['qp']['frank_wolfe_gap_tolerance']+1e-8
                    records.append({'bank':bank,'noise':noise_id,'signal':kind,'width':h,'sigma':sigma,
                                    'horizon_steps':horizon,'pool':pool,'candidates':m,'test_grid_risks':risks,
                                    'full_design_risks':full_train,'oracle_headroom':head,
                                    'sure_oracle_gap':risks['sure_average_full']-risks['oracle_full_test_grid_convex'],
                                    'cv_oracle_gap':risks['cv_stack_refit']-risks['oracle_full_test_grid_convex'],
                                    'sure_path_mass':[float(sum(ws[i] for i,(kk,j) in enumerate(idx) if kk==k and j>0)) for k in path_ids],
                                    'sure_zero_weight':float(ws[0])})
                completed+=1
                if completed%4==0:
                    print(f'Completed {completed}/{total} data-condition bundles ({5*completed} full paths), {time.perf_counter()-start:.1f}s',flush=True)
    summary,expansion=summarize(records,p)
    result={'protocol_sha256':hashlib.sha256(pb).hexdigest(),'protocol':p,'numpy':np.__version__,
            'completed_bundles':completed,'full_paths':len(timings),'half_fit_paths':2*len(timings),
            'prediction_only_duplicate_paths_for_timing':len(timings),'checks':checks,
            'elapsed_seconds':time.perf_counter()-start,'summary':summary,'pool_expansion':expansion,'records':records,'timings':timings}
    (ROOT/'neural_path_scale_results.json').write_text(json.dumps(result,indent=2))
    report(result)
    print(json.dumps({'bundles':completed,'full_paths':len(timings),'checks':checks,'seconds':result['elapsed_seconds']},indent=2))


def summarize(records,p):
    summaries=[]; expansions=[]
    for kind,h,sigma,horizon in product(p['signals'],p['widths'],p['noise_sd'],p['horizons_steps']):
        group=[r for r in records if (r['signal'],r['width'],r['sigma'],r['horizon_steps'])==(kind,h,sigma,horizon)]
        base={'signal':kind,'width':h,'sigma':sigma,'horizon_steps':horizon}
        for pool in p['pools']:
            rr=[r for r in group if r['pool']==pool]
            summaries.append({**base,'pool':pool,'replicates':len(rr),
                'mean_test_risks':{m:float(np.mean([r['test_grid_risks'][m] for r in rr])) for m in p['methods']},
                'oracle_headroom':cluster_stats(rr,lambda r:r['oracle_headroom'],p),
                'paired_differences':{a+' minus '+b:cluster_stats(rr,lambda r,a=a,b=b:r['test_grid_risks'][a]-r['test_grid_risks'][b],p) for a,b in [('sure_average_full','cv_stack_refit'),('sure_average_full','sure_select_full'),('cv_stack_refit','cv_select_refit')]}})
        for other in ['unit','random3']:
            paired=[]
            for bank,noise in product(range(p['independent_design_and_initialization_banks']),range(p['noise_replicates_per_bank'])):
                a=next(r for r in group if r['bank']==bank and r['noise']==noise and r['pool']==other)
                b=next(r for r in group if r['bank']==bank and r['noise']==noise and r['pool']=='scale3')
                ar,br=a['test_grid_risks'],b['test_grid_risks']
                g1=ar['oracle_full_test_grid_single']-br['oracle_full_test_grid_single']
                gc=ar['oracle_full_test_grid_convex']-br['oracle_full_test_grid_convex']
                dh=b['oracle_headroom']-a['oracle_headroom']
                assert abs(gc-g1-dh)<1e-12
                if other=='unit': assert min(g1,gc)>=-2e-7
                paired.append({'bank':bank,'single_gain':g1,'convex_gain':gc,'extra_headroom':dh,'sure_gain':ar['sure_average_full']-br['sure_average_full'],'cv_gain':ar['cv_stack_refit']-br['cv_stack_refit']})
            expansions.append({**base,'reference_pool':other,**{key:cluster_stats(paired,lambda r,key=key:r[key],p) for key in paired[0] if key!='bank'}})
    return summaries,expansions


def report(x):
    lines=['输入尺度候选互补性：完整结果','所有方法48标签；每条件6簇×2噪声重复。MCSE按独立簇计算。测试网格oracle仅诊断。',
           'unit=初始化0尺度1；random3=3初始化尺度1；scale3=初始化0尺度0.5/1/2。后两者路径数、候选数相同。',
           '协议SHA256: '+x['protocol_sha256'],
           'signal | width | sigma | steps | pool | SURE | CV | oracle single | oracle convex | headroom (SE) | SURE-CV (SE)']
    for r in x['summary']:
        a=r['mean_test_risks'];d=r['paired_differences']['sure_average_full minus cv_stack_refit'];hh=r['oracle_headroom']
        lines.append(f"{r['signal']} | {r['width']} | {r['sigma']} | {r['horizon_steps']} | {r['pool']} | "+' | '.join(f'{a[k]:.6f}' for k in ['sure_average_full','cv_stack_refit','oracle_full_test_grid_single','oracle_full_test_grid_convex'])+f" | {hh['mean']:.6f} ({hh['cluster_mc_se']:.6f}) | {d['mean']:+.6f} ({d['cluster_mc_se']:.6f})")
    lines+=['','scale3相对reference的改善：gain>0表示尺度池更好；非嵌套random3允许oracle gain为负。', 'signal | width | sigma | steps | reference | single gain | convex gain | extra headroom (SE) | SURE gain (SE) | CV gain (SE)']
    for r in x['pool_expansion']:
        lines.append(f"{r['signal']} | {r['width']} | {r['sigma']} | {r['horizon_steps']} | {r['reference_pool']} | {r['single_gain']['mean']:.6f} | {r['convex_gain']['mean']:.6f} | "+' | '.join(f"{r[k]['mean']:+.6f} ({r[k]['cluster_mc_se']:.6f})" for k in ['extra_headroom','sure_gain','cv_gain']))
    lines+=['',json.dumps(x['checks']),'所有方法逐重复风险见JSON；已知方差、离散GD、小规模实验，不认证总体快率。']
    (ROOT/'neural_path_scale_report.txt').write_text('\n'.join(lines))

if __name__=='__main__':
    run()
