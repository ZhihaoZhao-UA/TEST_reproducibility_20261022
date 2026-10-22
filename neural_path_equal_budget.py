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
from neural_path_experiment_matrix import signal

ROOT = Path(__file__).resolve().parent


def run():
    pb = (ROOT/'neural_path_equal_budget_protocol.json').read_bytes()
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
                for k,init in enumerate(initial_bank):
                    initial = tuple(a[:h].copy() for a in init)
                    t0=time.perf_counter()
                    F,D,Ft = train(X,y,initial,steps,lr=lr,directions=np.eye(N),X_eval=Xt)
                    full_jvp_time=time.perf_counter()-t0
                    t0=time.perf_counter()
                    Fplain,_,Ftplain = train(X,y,initial,steps,lr=lr,X_eval=Xt)
                    plain_time=time.perf_counter()-t0
                    err=max(float(np.max(abs(F-Fplain))),float(np.max(abs(Ft-Ftplain))))
                    checks['full_prediction_difference']=max(checks['full_prediction_difference'],err)
                    assert err < 1e-10
                    t0=time.perf_counter()
                    Fa,_,Ea=train(X[:half],y[:half],initial,steps,lr=lr,X_eval=Xe)
                    Fb,_,Eb=train(X[half:],y[half:],initial,steps,lr=lr,X_eval=Xe)
                    split_time=time.perf_counter()-t0
                    fold_error=max(float(np.max(abs(Fa-Ea[:half]))),float(np.max(abs(Fb-Eb[half:N]))))
                    checks['fold_training_prediction_difference']=max(checks['fold_training_prediction_difference'],fold_error)
                    assert fold_error < 1e-10
                    # First half's OOF predictions come from the second-half fit.
                    oof=np.vstack([Eb[:half],Ea[half:N]])
                    if completed==0 and k==0:
                        direction=np.random.default_rng(p['seed']+1).normal(size=N)
                        delta=1e-5
                        ap,_=train(X,y+delta*direction,initial,steps,lr=lr)
                        am,_=train(X,y-delta*direction,initial,steps,lr=lr)
                        checks['jvp_fd_error']=float(np.max(abs((ap-am)/(2*delta)-np.einsum('mij,j->im',D,direction))))
                        assert checks['jvp_fd_error'] < 1e-6
                    paths.append({'F':F,'Ft':Ft,'df':np.trace(D,axis1=1,axis2=2),'val':Ea[half:N],
                                  'split_test':Ea[N:],'oof':oof})
                    timings.append({'bank':bank,'noise':noise_id,'signal':kind,'width':h,'sigma':sigma,'initialization':k,
                                    'full_jvp_seconds':full_jvp_time,'full_prediction_only_seconds':plain_time,'two_half_fits_seconds':split_time})
                for horizon,pool in product(p['horizons_steps'],p['initializations_per_pool']):
                    idx=[(0,0)]+[(k,j) for k in range(pool) for j,t in enumerate(steps) if 0<t<=horizon]
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
                    we=np.array([float(steps[j]==horizon)/pool for k,j in idx])
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
                                    'horizon_steps':horizon,'initializations':pool,'candidates':m,'test_grid_risks':risks,
                                    'full_design_risks':full_train,'oracle_headroom':head,
                                    'sure_oracle_gap':risks['sure_average_full']-risks['oracle_full_test_grid_convex'],
                                    'cv_oracle_gap':risks['cv_stack_refit']-risks['oracle_full_test_grid_convex'],
                                    'sure_initialization_mass':[float(sum(ws[i] for i,(kk,j) in enumerate(idx) if kk==k and j>0)) for k in range(pool)],
                                    'sure_zero_weight':float(ws[0])})
                completed+=1
                if completed%4==0:
                    print(f'Completed {completed}/{total} data-condition bundles ({3*completed} full paths), {time.perf_counter()-start:.1f}s',flush=True)
    summary,expansion=summarize(records,p)
    result={'protocol_sha256':hashlib.sha256(pb).hexdigest(),'protocol':p,'numpy':np.__version__,
            'completed_bundles':completed,'full_paths':len(timings),'half_fit_paths':2*len(timings),
            'prediction_only_duplicate_paths_for_timing':len(timings),'checks':checks,
            'elapsed_seconds':time.perf_counter()-start,'summary':summary,'pool_expansion':expansion,'records':records,'timings':timings}
    (ROOT/'neural_path_equal_budget_results.json').write_text(json.dumps(result,indent=2))
    report(result)
    print(json.dumps({'bundles':completed,'full_paths':len(timings),'checks':checks,'seconds':result['elapsed_seconds']},indent=2))


def cluster_stats(rows,fn,p):
    values=[float(np.mean([fn(r) for r in rows if r['bank']==b])) for b in range(p['independent_design_and_initialization_banks'])]
    return {'mean':float(np.mean(values)),'cluster_mc_se':float(np.std(values,ddof=1)/np.sqrt(len(values))),'bank_means':values}


def summarize(records,p):
    summaries=[]; expansions=[]
    bases=['validation_stack_holdout','validation_stack_refit','cv_stack_refit','cv_select_refit','uniform_final_full','sure_select_full']
    for kind,h,sigma,horizon in product(p['signals'],p['widths'],p['noise_sd'],p['horizons_steps']):
        group=[r for r in records if (r['signal'],r['width'],r['sigma'],r['horizon_steps'])==(kind,h,sigma,horizon)]
        for pool in p['initializations_per_pool']:
            rr=[r for r in group if r['initializations']==pool]
            summaries.append({'signal':kind,'width':h,'sigma':sigma,'horizon_steps':horizon,'initializations':pool,'replicates':len(rr),
                'mean_test_risks':{m:float(np.mean([r['test_grid_risks'][m] for r in rr])) for m in p['methods']},
                'sure_minus_baseline':{b:cluster_stats(rr,lambda r,b=b:r['test_grid_risks']['sure_average_full']-r['test_grid_risks'][b],p) for b in bases},
                'oracle_headroom':cluster_stats(rr,lambda r:r['oracle_headroom'],p),
                'sure_oracle_gap':cluster_stats(rr,lambda r:r['sure_oracle_gap'],p),
                'cv_oracle_gap':cluster_stats(rr,lambda r:r['cv_oracle_gap'],p)})
        paired=[]
        for b,noise in product(range(p['independent_design_and_initialization_banks']),range(p['noise_replicates_per_bank'])):
            one=next(r for r in group if r['bank']==b and r['noise']==noise and r['initializations']==1)
            many=next(r for r in group if r['bank']==b and r['noise']==noise and r['initializations']==3)
            a,z=one['test_grid_risks'],many['test_grid_risks']
            single_gain=a['oracle_full_test_grid_single']-z['oracle_full_test_grid_single']
            convex_gain=a['oracle_full_test_grid_convex']-z['oracle_full_test_grid_convex']
            extra_head=many['oracle_headroom']-one['oracle_headroom']
            assert single_gain >= -1e-12 and convex_gain >= -2e-7
            assert abs(convex_gain-single_gain-extra_head)<1e-12
            paired.append({'bank':b,'single_oracle_gain':single_gain,'convex_oracle_gain':convex_gain,'additional_headroom':extra_head,
                           'sure_change':z['sure_average_full']-a['sure_average_full'],
                           'cv_change':z['cv_stack_refit']-a['cv_stack_refit'],
                           'uniform_change':z['uniform_final_full']-a['uniform_final_full']})
        expansions.append({'signal':kind,'width':h,'sigma':sigma,'horizon_steps':horizon,
                           **{k:cluster_stats(paired,lambda r,k=k:r[k],p) for k in paired[0] if k!='bank'}})
    return summaries,expansions


def report(x):
    lines=['同总标签预算与候选互补性：完整预设模拟结果','',
           '所有方法总标签48；holdout=24训练+24选权；refit 将索引/权重应用到全48训练路径；CV为两折OOF选权后全数据重训。',
           '真实sigma已知的合成研究；不声称未知方差下的实用优越性。相同标签数不等于相同计算预算。',
           '单路径固定为初始化0；扩展池包含3个预设独立初始化。零模型去重。强弱结论不得按结果筛选候选。',
           '每条件6个独立设计-初始化簇，每簇2个噪声重复。括号内为簇MC标准误，不作显著性结论。',
           '下列风险均为独立512点测试网格无噪声MSE；oracle只用于诊断。',
           '方案SHA256: '+x['protocol_sha256'],'',
           'signal | h | sigma | steps | pool | SURE | val-holdout-stack | val-refit-stack | CV-refit-stack | final-ensemble | SURE-CV (SE) | oracle headroom']
    for r in x['summary']:
        a=r['mean_test_risks']; d=r['sure_minus_baseline']['cv_stack_refit']
        lines.append(f"{r['signal']} | {r['width']} | {r['sigma']} | {r['horizon_steps']} | {r['initializations']} | "+' | '.join(f'{a[k]:.5f}' for k in ['sure_average_full','validation_stack_holdout','validation_stack_refit','cv_stack_refit','uniform_final_full'])+f" | {d['mean']:+.5f} ({d['cluster_mc_se']:.5f}) | {r['oracle_headroom']['mean']:.5f}")
    lines+=['','候选池扩展分解：oracle gain=单模型改善+额外组合空间；SURE/CV change<0 表示实际改善。',
            'signal | h | sigma | steps | single oracle gain | convex oracle gain | additional headroom | SURE change (SE) | CV change (SE)']
    for r in x['pool_expansion']:
        lines.append(f"{r['signal']} | {r['width']} | {r['sigma']} | {r['horizon_steps']} | "+' | '.join(f"{r[k]['mean']:.5f}" for k in ['single_oracle_gain','convex_oracle_gain','additional_headroom'])+f" | {r['sure_change']['mean']:+.5f} ({r['sure_change']['cluster_mc_se']:.5f}) | {r['cv_change']['mean']:+.5f} ({r['cv_change']['cluster_mc_se']:.5f})")
    lines+=['','实现检查：'+json.dumps(x['checks']),
            '候选池扩展不是原单路径谱尾主定理的直接数值认证；一般光滑候选的SURE代数可应用，但多初始化谱/宽度理论需另证。',
            '所有方法、逐重复风险、配对差和计时均在同名JSON；未按结果修改方案。']
    (ROOT/'neural_path_equal_budget_report.txt').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    run()
