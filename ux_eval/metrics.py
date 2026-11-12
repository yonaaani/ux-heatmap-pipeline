"""Метрики валідації карт уваги проти реального айтрекінгу.

P — «еталонна» карта (Ground Truth, з айтрекера), Q / S — прогноз моделі,
fixations — бінарна маска точок фіксацій (для NSS і AUC-Judd).
Цільові значення: KL ≤ 0.42, CC ≥ 0.86, NSS ≥ 2.4, AUC-Judd ≥ 0.89.

Метрики траєкторій (ScanMatch, MultiMatch) тут не реалізовано: для них є
готові пакети (напр., `multimatch_gaze`), які варто підключити під час
експериментів на UEyes.
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float64).eps
TARGETS = {"KL": ("<=", 0.42), "CC": (">=", 0.86), "NSS": (">=", 2.4), "AUC_Judd": (">=", 0.89)}


def _as_dist(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    m = m - m.min()
    return m / (m.sum() + EPS)


def kl_divergence(p_gt: np.ndarray, q_pred: np.ndarray) -> float:
    p, q = _as_dist(p_gt), _as_dist(q_pred)
    return float((p * np.log(EPS + p / (q + EPS))).sum())


def cc(p_gt: np.ndarray, s_pred: np.ndarray) -> float:
    p_gt, s_pred = _as_dist(p_gt), _as_dist(s_pred)
    a = (p_gt - p_gt.mean()) / (p_gt.std() + EPS)
    b = (s_pred - s_pred.mean()) / (s_pred.std() + EPS)
    return float((a * b).mean())


def nss(s_pred: np.ndarray, fixations: np.ndarray) -> float:
    s_pred = _as_dist(s_pred)
    s = (s_pred - s_pred.mean()) / (s_pred.std() + EPS)
    return float(s[fixations.astype(bool)].mean())


def auc_judd(s_pred: np.ndarray, fixations: np.ndarray) -> float:
    s = (s_pred - s_pred.min()) / (s_pred.max() - s_pred.min() + EPS)
    fix = fixations.astype(bool)
    thresholds = np.sort(s[fix])[::-1]
    n_fix, n_pix = fix.sum(), s.size
    above = n_pix - np.searchsorted(np.sort(s, axis=None), thresholds, side="left")
    hits = np.arange(1, n_fix + 1)
    tp = np.concatenate(([0.0], hits / n_fix, [1.0]))
    fp = np.concatenate(([0.0], (above - hits) / (n_pix - n_fix), [1.0]))
    return float(np.trapezoid(tp, fp))


def evaluate_saliency(pred: np.ndarray, gt_map: np.ndarray, fixations: np.ndarray) -> dict:
    res = {
        "KL": round(kl_divergence(gt_map, pred), 4),
        "CC": round(cc(gt_map, pred), 4),
        "NSS": round(nss(pred, fixations), 4),
        "AUC_Judd": round(auc_judd(pred, fixations), 4),
    }
    res["passed"] = {k: (res[k] <= v if op == "<=" else res[k] >= v) for k, (op, v) in TARGETS.items()}
    return res


# ---------------------------------------------------------------- траєкторії погляду (крок 4)

SCANPATH_TARGETS = {"ScanMatch": (">=", 0.62), "MultiMatch": (">=", 0.84)}


def _scanmatch_sub_matrix(xb: int = 10, yb: int = 10, threshold: float = 5.0) -> np.ndarray:
    idx = [(i, j) for i in range(yb) for j in range(xb)]
    d = np.array([[np.hypot(j - jj, i - ii) for (ii, jj) in idx] for (i, j) in idx])
    return np.abs(d - d.max()) - (d.max() - threshold)


_SM_SUB = _scanmatch_sub_matrix()


def scan_match(P: np.ndarray, Q: np.ndarray, h: int, w: int, gap: float = 0.0) -> float:
    """ScanMatch (Cristino et al., 2010) у реалізації з коду UEyes (evaluation/eval_scanpaths.py, з rAm1n/saliency):
    фіксації -> клітинки сітки 10×10, вирівнювання Needleman-Wunsch з матрицею схожості (поріг 5), без штрафу
    за пропуск, нормування на максимальний можливий бал. P, Q — масиви (n, 2) з координатами x, y у пікселях."""
    def bins(p):
        xs = np.clip((p[:, 0] // (w // 10)).astype(int), 0, 9)
        ys = np.clip(((h - p[:, 1]) // (h // 10)).astype(int), 0, 9)
        return ys * 10 + xs
    a, b = bins(np.asarray(P, float)), bins(np.asarray(Q, float))
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1))
    S[0, 1:] = gap * np.arange(m)
    S[1:, 0] = gap * np.arange(n)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i, j] = max(S[i - 1, j - 1] + _SM_SUB[b[j - 1], a[i - 1]], S[i - 1, j] + gap, S[i, j - 1] + gap)
    return float(S.max() / (_SM_SUB.max() * max(n, m) + np.finfo(np.float32).eps))


def multimatch(P: np.ndarray, Q: np.ndarray, h: int, w: int) -> dict[str, float]:
    """MultiMatch (Jarodzka et al., 2010; Dewhurst et al., 2012) як у multimatch-gaze (Wagner et al., 2019),
    без спрощення траєкторій (типові пороги 0). P, Q — масиви (n, 3): x, y (пікселі), тривалість фіксації (с).

    Траєкторія -> вектори саккад між сусідніми фіксаціями. Саккади двох траєкторій вирівнюються найкоротшим шляхом
    у матриці різниць векторів (рухи вправо / вниз / по діагоналі), далі для вирівняних пар — медіани різниць:
      вектор (/ 2·діагональ), напрямок (/ π), довжина (/ діагональ), положення початку (/ діагональ),
      тривалість фіксації (|d1 − d2| / max). Схожість = 1 − нормована різниця; MultiMatch — середнє п'яти."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if len(P) < 2 or len(Q) < 2:
        return {k: float("nan") for k in ("vector", "direction", "length", "position", "duration", "mean")}
    vp, vq = np.diff(P[:, :2], axis=0), np.diff(Q[:, :2], axis=0)
    n, m = len(vp), len(vq)
    M = np.linalg.norm(vp[:, None, :] - vq[None, :, :], axis=2)
    # найкоротший шлях (0,0) -> (n-1,m-1) через клітинки матриці (DAG, тож динамічне програмування = Дейкстра)
    C = np.full((n, m), np.inf)
    C[0, 0] = M[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = min(C[i - 1, j] if i else np.inf, C[i, j - 1] if j else np.inf,
                       C[i - 1, j - 1] if i and j else np.inf)
            C[i, j] = best + M[i, j]
    path, i, j = [(n - 1, m - 1)], n - 1, m - 1
    while (i, j) != (0, 0):
        cand = [(C[i - 1, j - 1], i - 1, j - 1)] if i and j else []
        cand += [(C[i - 1, j], i - 1, j)] if i else []
        cand += [(C[i, j - 1], i, j - 1)] if j else []
        _, i, j = min(cand)
        path.append((i, j))
    ii, jj = np.array(path[::-1]).T
    diag = float(np.hypot(w, h))
    vec = np.linalg.norm(vp[ii] - vq[jj], axis=1)
    ang = np.abs(np.arctan2(vp[ii, 1], vp[ii, 0]) - np.arctan2(vq[jj, 1], vq[jj, 0]))
    ang = np.minimum(ang, 2 * np.pi - ang)
    length = np.abs(np.linalg.norm(vp[ii], axis=1) - np.linalg.norm(vq[jj], axis=1))
    pos = np.linalg.norm(P[ii, :2] - Q[jj, :2], axis=1)
    dur = np.abs(P[ii, 2] - Q[jj, 2]) / np.maximum(np.maximum(P[ii, 2], Q[jj, 2]), 1e-9)
    out = {"vector": 1 - np.median(vec) / (2 * diag), "direction": 1 - np.median(ang) / np.pi,
           "length": 1 - np.median(length) / diag, "position": 1 - np.median(pos) / diag,
           "duration": 1 - np.median(dur)}
    out = {k: float(v) for k, v in out.items()}
    out["mean"] = float(np.mean(list(out.values())))
    return out
