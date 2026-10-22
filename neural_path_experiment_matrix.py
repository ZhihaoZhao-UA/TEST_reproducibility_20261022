"""Run the locally frozen small simulation matrix. NumPy only.
No test labels enter implementable weight selection. Oracle diagnostics are explicit.
"""
import hashlib
import json
import platform
import time
from itertools import product
from pathlib import Path
import numpy as np
from neural_path_average import train, solve_qp


ROOT = Path(__file__).resolve().parent


def signal(X, kind):
    if kind == 'smooth':
        return np.sin(np.pi*X[:, 0])+.4*X[:, 1]
    return np.sin(3*np.pi*X[:, 0])+.6*np.cos(2*np.pi*X[:, 1])


def run():
    protocol_path = ROOT/'neural_path_experiment_protocol.json'
    protocol_bytes = protocol_path.read_bytes()
    p = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    records, timings, checks = [], [], {'fd_error': 0., 'max_qp_gap': 0., 'prediction_only_difference': 0.}
    nt, nv, nz = p['n_train'], p['n_validation'], p['n_test_grid']
    steps, lr, tol = p['checkpoints'], p['learning_rate'], p['qp_frank_wolfe_gap_tolerance']
    total_paths = np.prod([p['design_replicates'], p['initializations_per_design'], p['noise_replicates_per_initialization'], len(p['widths']), len(p['noise_sd']), len(p['signals'])])
    completed, start = 0, time.perf_counter()
    for design in range(p['design_replicates']):
        drng = np.random.default_rng(np.random.SeedSequence([p['seed'], 1, design]))
        X, Xv, Xt = [drng.uniform(-1, 1, (n, 2)) for n in [nt, nv, nz]]
        Xe = np.vstack([Xv, Xt])
        for init_id in range(p['initializations_per_design']):
            irng = np.random.default_rng(np.random.SeedSequence([p['seed'], 2, design, init_id]))
            hm = max(p['widths'])
            full_initial = (irng.normal(0, .7, (hm, 2)), irng.normal(0, .1, hm), irng.uniform(-.5, .5, hm))
            for noise_id in range(p['noise_replicates_per_initialization']):
                nrng = np.random.default_rng(np.random.SeedSequence([p['seed'], 3, design, init_id, noise_id]))
                standard_train, standard_val = nrng.normal(size=nt), nrng.normal(size=nv)
                for kind, h, sigma in product(p['signals'], p['widths'], p['noise_sd']):
                    mu, muv, mut = signal(X, kind), signal(Xv, kind), signal(Xt, kind)
                    y, yv = mu+sigma*standard_train, muv+sigma*standard_val
                    initial = tuple(z[:h].copy() for z in full_initial)
                    t0 = time.perf_counter()
                    F, D, Fe = train(X, y, initial, steps, lr=lr, directions=np.eye(nt), X_eval=Xe)
                    exact_time = time.perf_counter()-t0
                    t0 = time.perf_counter()
                    plain, _, plain_eval = train(X, y, initial, steps, lr=lr, X_eval=Xe)
                    plain_time = time.perf_counter()-t0
                    difference = max(float(np.max(np.abs(plain-F))), float(np.max(np.abs(plain_eval-Fe))))
                    assert difference < 1e-10
                    checks['prediction_only_difference'] = max(checks['prediction_only_difference'], difference)
                    if completed == 0:
                        direction = np.random.default_rng(p['seed']+1).normal(size=nt)
                        delta = 1e-5
                        plus, _ = train(X, y+delta*direction, initial, steps, lr=lr)
                        minus, _ = train(X, y-delta*direction, initial, steps, lr=lr)
                        jvp = np.einsum('mij,j->im', D, direction)
                        checks['fd_error'] = float(np.max(np.abs((plus-minus)/(2*delta)-jvp)))
                        assert checks['fd_error'] < 1e-6
                    Fv, Ft = Fe[:nv], Fe[nv:]
                    df = np.trace(D, axis1=1, axis2=2)
                    W, b, a = initial
                    H = np.tanh(X@W.T+b)
                    V = (1-H**2)*a
                    kernel = (H@H.T+(X@X.T+1)*(V@V.T))/(nt*h)
                    gamma, U = np.linalg.eigh(kernel)
                    gamma, U = gamma[::-1], U[:, ::-1]
                    assert gamma[-1] > -1e-10 and lr*gamma[0] < 1
                    filters = 1-(1-lr*gamma[:, None])**np.array(steps)[None, :]
                    S = np.einsum('ia,am,ja->mij', U, filters, U)
                    F0 = np.einsum('mij,j->im', S, y)
                    nu = U.T@mu
                    timings.append({'signal': kind, 'width': h, 'sigma': sigma, 'design': design, 'initialization': init_id,
                                    'noise': noise_id, 'exact_jvp_seconds': exact_time, 'prediction_only_seconds': plain_time})
                    for horizon in p['horizons_steps']:
                        m = sum(t <= horizon for t in steps)
                        A, Av, At = F[:, :m], Fv[:, :m], Ft[:, :m]
                        def qp(a_matrix, target, penalty):
                            ww, gg = solve_qp(a_matrix, target, penalty, tol=tol)
                            checks['max_qp_gap'] = max(checks['max_qp_gap'], float(gg))
                            return ww
                        weights = {}
                        weights['sure_average'] = qp(A, y, sigma**2*df[:m]/nt)
                        index_sure = int(np.argmin(np.mean((A-y[:, None])**2, axis=0)+2*sigma**2*df[:m]/nt))
                        weights['sure_select'] = np.eye(m)[index_sure]
                        weights['validation_early_stop'] = np.eye(m)[np.argmin(np.mean((Av-yv[:, None])**2, axis=0))]
                        weights['validation_stack'] = qp(Av, yv, np.zeros(m))
                        weights['uniform_average'] = np.ones(m)/m
                        weights['last_checkpoint'] = np.eye(m)[-1]
                        weights['oracle_train_convex'] = qp(A, mu, np.zeros(m))
                        weights['oracle_test_grid_single'] = np.eye(m)[np.argmin(np.mean((At-mut[:, None])**2, axis=0))]
                        weights['oracle_test_grid_convex'] = qp(At, mut, np.zeros(m))
                        losses = {name: {'train': float(np.mean((A@w-mu)**2)), 'test_grid': float(np.mean((At@w-mut)**2))} for name,w in weights.items()}
                        # General pathwise SURE bound, independent of the NN flow theorem.
                        z = (y-mu)@A-sigma**2*df[:m]
                        excess = losses['sure_average']['train']-losses['oracle_train_convex']['train']
                        assert excess <= 2*np.ptp(z)/nt+tol+1e-8
                        centered_filters = filters[:, :m]-filters[:, :m].mean(1, keepdims=True)
                        tail = float(np.sum(np.sum(centered_filters[3:]**2, axis=1)*(sigma**2*nu[3:]**2+2*sigma**4)))
                        mechanisms = {'oracle_test_grid_headroom': losses['oracle_test_grid_single']['test_grid']-losses['oracle_test_grid_convex']['test_grid'],
                                      'sure_train_excess_over_convex_oracle': excess,
                                      'max_normalized_prediction_linearization_error_squared': float(np.max(np.mean((F[:, :m]-F0[:, :m])**2, axis=0))),
                                      'max_normalized_jacobian_linearization_error_squared': float(np.max(np.sum((D[:m]-S[:m])**2, axis=(1, 2))/nt)),
                                      'kernel_min_eigenvalue': float(gamma[-1]), 'kernel_max_eigenvalue': float(gamma[0]),
                                      'frozen_tail_energy_q3': tail,
                                      'sure_support_size_1e_6': int(np.sum(weights['sure_average'] > 1e-6)),
                                      'sure_last_weight': float(weights['sure_average'][-1])}
                        records.append({'signal': kind, 'width': h, 'sigma': sigma, 'horizon_steps': horizon,
                                        'design': design, 'initialization': init_id, 'noise': noise_id,
                                        'losses': losses, 'mechanisms': mechanisms})
                    completed += 1
                    if completed % 8 == 0:
                        print(f'Completed {completed}/{total_paths} paths; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    summary = []
    comparisons = ['last_checkpoint', 'sure_select', 'validation_early_stop', 'validation_stack']
    for kind,h,sigma,horizon in product(p['signals'], p['widths'], p['noise_sd'], p['horizons_steps']):
        rows = [r for r in records if (r['signal'],r['width'],r['sigma'],r['horizon_steps'])==(kind,h,sigma,horizon)]
        risk = {method:{metric: float(np.mean([r['losses'][method][metric] for r in rows])) for metric in ['train','test_grid']} for method in p['methods']}
        contrasts = {}
        for baseline in comparisons:
            contrasts[baseline] = {}
            for metric in ['train','test_grid']:
                dm = [float(np.mean([r['losses']['sure_average'][metric]-r['losses'][baseline][metric] for r in rows if r['design']==d])) for d in range(p['design_replicates'])]
                contrasts[baseline][metric] = {'paired_difference': float(np.mean(dm)),
                                              'design_cluster_mc_se': float(np.std(dm, ddof=1)/np.sqrt(len(dm))), 'design_means': dm}
        summary.append({'signal':kind,'width':h,'sigma':sigma,'horizon_steps':horizon,'replicates':len(rows),'mean_risks':risk,
                        'paired_contrasts_sure_minus_baseline':contrasts,
                        'mean_mechanisms':{key:float(np.mean([r['mechanisms'][key] for r in rows])) for key in rows[0]['mechanisms']}})
    result = {'protocol_sha256':protocol_hash,'protocol':p,'python':platform.python_version(),'numpy':np.__version__,
              'elapsed_seconds':time.perf_counter()-start,'checks':checks,'completed_training_paths':completed,
              'evaluation_cells':len(summary),'records':records,'timings':timings,'summary':summary}
    (ROOT/'neural_path_experiment_results.json').write_text(json.dumps(result, indent=2))
    write_report(result)
    print(json.dumps({'paths':completed,'cells':len(summary),'checks':checks,'seconds':result['elapsed_seconds']},indent=2))


def write_report(x):
    lines = ['神经网络路径平均：预先冻结的小规模模拟矩阵', '',
             '范围：24 个条件，每个条件 12 次配对重复；共 96 条训练路径，三个 horizon 复用前缀。',
             '三个独立设计簇，每簇两次初始化、每次两次噪声。标准误按设计簇计算，只有 3 簇，不作显著性宣称。',
             '验证集方法多使用 24 个标签；SURE 已知真实噪声方差。本轮不是同总标签预算比较。',
             'test 表示 256 个独立测试点上的无噪声均值 MSE，不是理论中的固定训练设计损失。',
             '差值定义为 SURE平均 减去基线；负数有利于 SURE。括号内为设计簇 MC 标准误。',
             '方案 SHA256: '+x['protocol_sha256'], '',
             'signal | h | sigma | steps | test SURE | test last | delta last (SE) | delta SURE-select (SE) | delta val-stop (SE) | delta val-stack (SE) | oracle headroom']
    for r in x['summary']:
        vals=[]
        for key in ['last_checkpoint','sure_select','validation_early_stop','validation_stack']:
            v=r['paired_contrasts_sure_minus_baseline'][key]['test_grid']
            vals.append(f"{v['paired_difference']:+.5f} ({v['design_cluster_mc_se']:.5f})")
        risk=r['mean_risks']
        lines.append(f"{r['signal']} | {r['width']} | {r['sigma']} | {r['horizon_steps']} | {risk['sure_average']['test_grid']:.5f} | {risk['last_checkpoint']['test_grid']:.5f} | "+' | '.join(vals)+f" | {r['mean_mechanisms']['oracle_test_grid_headroom']:.5f}")
    lines += ['', '机制诊断：', 'signal | h | sigma | steps | 预测线性化误差 | 响应Jacobian误差 | SURE平均训练oracle差 | 平均末时刻权重']
    for r in x['summary']:
        m=r['mean_mechanisms']
        lines.append(f"{r['signal']} | {r['width']} | {r['sigma']} | {r['horizon_steps']} | {m['max_normalized_prediction_linearization_error_squared']:.6g} | {m['max_normalized_jacobian_linearization_error_squared']:.6g} | {m['sure_train_excess_over_convex_oracle']:.6g} | {m['sure_last_weight']:.4f}")
    lines += ['', '计时：']
    for h in x['protocol']['widths']:
        t=[v for v in x['timings'] if v['width']==h]
        exact=np.mean([v['exact_jvp_seconds'] for v in t]); plain=np.mean([v['prediction_only_seconds'] for v in t])
        lines.append(f'h={h}: 完整训练响应Jacobian平均 {exact:.4f}s，仅预测训练平均 {plain:.4f}s，比值 {exact/plain:.2f}。均包含同样的外部点预测；不是大规模硬件benchmark。')
    lines += ['', '检查：'+json.dumps(x['checks']), '完整逐重复数据、训练/测试风险和基线结果见同名 JSON。',
              '这些结果不证明连续梯度流定理，不认证未知常数，不足以支持普遍优越性。']
    (ROOT/'neural_path_experiment_report.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
