import numpy as np

def train(X, y, initial, times, lr=0.25, directions=None, X_eval=None, first_layer_multiplier=1.):
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
            dW, db, da = dW - lr * first_layer_multiplier * dgW, db - lr * dgb, da - lr * dga
        W, b, a = W - lr * first_layer_multiplier * gW, b - lr * gb, a - lr * ga
    if X_eval is not None:
        return np.array(preds).T, np.array(jacobians), np.array(eval_preds).T
    return np.array(preds).T, np.array(jacobians)

