"""Pilot: SURE-weighted prediction averaging along a finite-width tanh path.

Requires Python 3 + NumPy only. Run: python neural_path_average.py --reps 200
Fixed design, known Gaussian noise variance, fixed initialization; all network
parameters are trained. Uses exact response directional derivatives, not NTK.
No test-set or population-risk claim. Oracle baselines use the known simulated
mean and are not implementable methods. Writes a JSON result beside this file.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def simplex(v):
    u = np.sort(v)[::-1]
    s = np.cumsum(u) - 1
    k = np.nonzero(u - s / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - s[k] / (k + 1), 0)


def solve_qp(A, y, penalty, tol=1e-8, max_iter=30000):
    """Min ||y-Aw||^2/n + 2 penalty'w on the simplex; FW gap certificate."""
    n, m = A.shape
    Q, c = A.T @ A / n, -A.T @ y / n + penalty
    L = max(2 * np.linalg.eigvalsh(Q)[-1], 1e-12)
    w = np.ones(m) / m
    z, momentum = w.copy(), 1.0
    for it in range(max_iter):
        new = simplex(z - (2 * (Q @ z + c)) / L)
        grad = 2 * (Q @ new + c)
        gap = float(grad @ new - grad.min())
        if gap < tol:
            return new, gap
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        z = new + (momentum - 1) / next_momentum * (new - w)
        w, momentum = new, next_momentum
    raise RuntimeError(f"QP did not converge: gap={gap}")


def train(X, y, initial, times, lr=0.25, directions=None, X_eval=None):
    """Centered network H a/sqrt(h)-f_init; propagate full training JVPs."""
    W, b, a = [p.copy() for p in initial]
    n, d = X.shape
    h = len(a)
    scale = np.sqrt(h)
    base = np.tanh(X @ W.T + b) @ a / scale
    eval_base = None if X_eval is None else np.tanh(X_eval @ W.T+b)@a/scale
    eval_preds = []
    if directions is not None:
        U = directions.T
        B = len(U)
        dW, db, da = np.zeros((B, h, d)), np.zeros((B, h)), np.zeros((B, h))
    preds, jacobians = [], []
    for t in range(max(times) + 1):
        H = np.tanh(X @ W.T + b)
        P = 1 - H**2
        f = H @ a / scale - base
        r = f - y
        if directions is not None:
            dH = P[None] * (np.einsum('nd,bhd->bnh', X, dW) + db[:, None])
            df = (np.einsum('bnh,h->bn', dH, a) + da @ H.T) / scale
        if t in times:
            preds.append(f.copy())
            if X_eval is not None:
                eval_preds.append(np.tanh(X_eval@W.T+b)@a/scale-eval_base)
            if directions is not None:
                jacobians.append(df.T.copy())
        if t == max(times):
            break
        q = r[:, None] * a * P
        ga, gW, gb = H.T @ r / (n * scale), q.T @ X / (n * scale), q.sum(0) / (n * scale)
        if directions is not None:
            dr = df - U
            dq = (dr[:, :, None] * a * P
                  + r[None, :, None] * da[:, None] * P
                  - 2 * r[None, :, None] * a * H * dH)
            dga = (np.einsum('bnh,n->bh', dH, r) + dr @ H) / (n * scale)
            dgW = np.einsum('bnh,nd->bhd', dq, X) / (n * scale)
            dgb = dq.sum(1) / (n * scale)
            dW, db, da = dW - lr * dgW, db - lr * dgb, da - lr * dga
        W, b, a = W - lr * gW, b - lr * gb, a - lr * ga
    if X_eval is not None:
        return np.array(preds).T, np.array(jacobians), np.array(eval_preds).T
    return np.array(preds).T, np.array(jacobians)


def run(reps):
    rng = np.random.default_rng(20261004)
    n, h, sigma = 32, 16, 0.6
    X = rng.uniform(-1, 1, (n, 2))
    mu = np.sin(np.pi * X[:, 0]) + 0.4 * X[:, 1]
    initial = (rng.normal(0, 0.7, (h, 2)), rng.normal(0, 0.1, h), rng.normal(0, 0.5, h))
    times = [0, 1, 2, 5, 10, 20, 50, 100, 200]
    names = ['sure_average', 'sure_select', 'uniform_average', 'last_iterate',
             'realized_oracle_single', 'realized_oracle_convex']
    losses = {name: [] for name in names}
    weights, dfs, sure_bias, path_terms = [], [], [], []
    max_gap, identity_error, finite_diff_error = 0.0, 0.0, 0.0
    max_pathwise_bound_violation = -float('inf')
    for rep in range(reps):
        eps = rng.normal(0, sigma, n)
        y = mu + eps
        A, D = train(X, y, initial, times, directions=np.eye(n))
        df = np.trace(D, axis1=1, axis2=2)
        w, gap = solve_qp(A, y, sigma**2 * df / n)
        wo, gap_o = solve_qp(A, mu, np.zeros(len(times)))
        max_gap = max(max_gap, gap, gap_o)
        single_sure = ((y[:, None] - A)**2).mean(0) + 2*sigma**2*df/n - sigma**2
        single_loss = ((mu[:, None] - A)**2).mean(0)
        loss = lambda f: float(np.mean((f-mu)**2))
        vals = [loss(A @ w), single_loss[np.argmin(single_sure)], loss(A.mean(1)),
                single_loss[-1], single_loss.min(), loss(A @ wo)]
        for name, value in zip(names, vals):
            losses[name].append(float(value))
        weights.append(w.tolist())
        dfs.append(df.tolist())
        sure_bias.append((single_sure-single_loss).tolist())
        # Check exact algebra behind the pathwise oracle comparison.
        T = eps @ A - sigma**2 * df
        increment_bound = 2 * np.abs(np.diff(T)).sum() / n
        max_pathwise_bound_violation = max(max_pathwise_bound_violation, float(vals[0]-vals[-1]-increment_bound-gap))
        path_terms.append((sigma**2 * np.sum(np.diff(A, axis=1)**2, axis=0)
                           + sigma**4 * np.sum(np.diff(D, axis=0)**2, axis=(1, 2))).tolist())
        common = (eps @ eps - n*sigma**2 + 2*eps @ mu) / n
        identity_error = max(identity_error, float(np.max(np.abs(single_sure-single_loss-common+2*T/n))))
        # Explicit training-JVP verification with a central finite difference.
        if rep == 0:
            direction = rng.normal(size=n)
            delta = 1e-5
            Ap, _ = train(X, y+delta*direction, initial, times)
            Am, _ = train(X, y-delta*direction, initial, times)
            exact = np.einsum('mij,j->im', D, direction)
            finite_diff_error = float(np.max(np.abs((Ap-Am)/(2*delta)-exact)))
    result = {
        'scope': 'Pilot only: finite-width nonlinear tanh network, fixed-design mean estimation, known noise variance',
        'seed': 20261004, 'n': n, 'width': h, 'sigma': sigma, 'times': times,
        'repetitions': reps,
        'mse': {k: {'mean': float(np.mean(v)), 'mc_se': float(np.std(v, ddof=1)/np.sqrt(reps))} for k,v in losses.items()},
        'paired_difference_average_minus_select': {
            'mean': float(np.mean(np.array(losses['sure_average'])-losses['sure_select'])),
            'mc_se': float(np.std(np.array(losses['sure_average'])-losses['sure_select'], ddof=1)/np.sqrt(reps))},
        'paired_difference_average_minus_last': {
            'mean': float(np.mean(np.array(losses['sure_average'])-losses['last_iterate'])),
            'mc_se': float(np.std(np.array(losses['sure_average'])-losses['last_iterate'], ddof=1)/np.sqrt(reps))},
        'empirical_path_bound_remainder_not_a_certified_population_bound': float(2*np.sqrt(np.mean(path_terms, axis=0)).sum()/n),
        'max_pathwise_bound_violation': max_pathwise_bound_violation,
        'mean_weights': np.mean(weights, 0).tolist(), 'mean_divergence': np.mean(dfs, 0).tolist(),
        'fixed_checkpoint_sure_minus_loss_mean': np.mean(sure_bias, 0).tolist(),
        'fixed_checkpoint_sure_minus_loss_mc_se': (np.std(sure_bias, axis=0, ddof=1)/np.sqrt(reps)).tolist(),
        'max_qp_gap': max_gap, 'max_algebra_identity_error': identity_error,
        'max_training_jvp_central_difference_error': finite_diff_error,
        'raw_losses': losses,
    }
    assert max_gap < 1e-8
    assert identity_error < 1e-10
    assert finite_diff_error < 1e-5
    assert max_pathwise_bound_violation < 1e-7
    path = Path(__file__).with_name('neural_path_pilot.json')
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['raw_losses']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=200)
    run(parser.parse_args().reps)
