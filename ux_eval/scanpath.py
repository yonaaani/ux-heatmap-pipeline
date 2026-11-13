"""Крок 4. Прогнозування траєкторії погляду (Scanpath).

Трансформерні моделі (Gazeformer / HAT) — окремо; тут реалізовано пошук піків уваги з гасінням
Inhibition of Return (IoR), як у моделі Itti-Koch, плюс «короткі стрибки» з HAT.
На UEyes цей метод не поступається HAT, тому використовується як основний.

Результат — послідовність фіксацій (x_k, y_k, t_k), де t_k — тривалість
фіксації в мс, пропорційна «масі» уваги навколо точки.
"""
from __future__ import annotations

import numpy as np


# Приглушення зон за структурою екрана (structure_weights): прибирає фіксації на системному рядку.
# Притягування до елементів (ELEM_FLOOR < 1) вимкнено.
ZONE_W = 0.5
ELEM_FLOOR = 1.0
FOVEA_FRAC = 0.015  # ≈ центральний зір (~2°): ~29 px на екрані 1920 px


def _element_at(layout, x: float, y: float):
    """Елемент під точкою фіксації.

    Спершу CTA: фіксація зараховується кнопці, якщо точка в її рамці або поруч у межах центрального зору (людина
    бачить чітко не точку, а коло ~2°). Інакше «найглибший» елемент (текст усередині кнопки повертав би текст, і
    погляд на кнопці не зараховувався б як погляд на CTA — так було на скріншоті Amazon).
    """
    if layout is None:
        return None
    m = FOVEA_FRAC * max(layout.width, layout.height) if hasattr(layout, "width") else 0
    for el in layout.elements:
        if el.role == "cta":
            ex, ey, ew, eh = el.box
            dx, dy = max(ex - x, 0, x - ex - ew), max(ey - y, 0, y - ey - eh)
            if dx * dx + dy * dy <= m * m:  # відстань від точки до рамки (0 — всередині)
                return el
    for el in reversed(layout.elements):
        if el.type in ("card", "divider"):
            continue
        ex, ey, ew, eh = el.box
        if ex <= x <= ex + ew and ey <= y <= ey + eh:
            return el
    return None


def structure_weights(layout, H: int, W: int, modals=None, bar_top: int | None = None,
                      zone_w: float = 1.0, elem_floor: float = 1.0, status_frac: float = 0.04) -> np.ndarray | None:
    """Вага кожного пікселя за структурою екрана з кроку 1 (1 — без змін).

    (1) Зони, куди люди майже не дивляться, множаться на zone_w: системна панель Android унизу (bar_top) і рядок
        стану вгорі (status_frac висоти, лише коли знайдено панель Android); сторінка поза спливним вікном.
    (2) Порожнє місце між елементами множиться на elem_floor: люди дивляться на текст, кнопки й картинки,
        а не в проміжки (межі елементів трохи розмиті, щоб фіксація могла впасти на край).
    """
    if layout is None or (zone_w >= 1.0 and elem_floor >= 1.0):
        return None
    import cv2
    wgt = np.ones((H, W), np.float32)
    if elem_floor < 1.0:
        mask = np.zeros((H, W), np.float32)
        for el in layout.elements:
            if el.type in ("card", "divider"):
                continue
            x, y, w, h = (int(v) for v in el.box)
            mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 1.0
        k = max(3, int(0.01 * max(H, W)) | 1)
        mask = cv2.GaussianBlur(mask, (0, 0), k)
        wgt *= elem_floor + (1.0 - elem_floor) * np.clip(mask * 1.5, 0, 1)
    if zone_w < 1.0:
        zone = np.zeros((H, W), bool)
        if bar_top is not None and bar_top < H:
            zone[bar_top:] = True
            zone[:int(status_frac * H)] = True
        if modals:
            inside = np.zeros((H, W), bool)
            for mx, my, mw, mh in modals:
                inside[max(0, int(my)):int(my + mh), max(0, int(mx)):int(mx + mw)] = True
            zone |= ~inside
        wgt[zone] *= zone_w
    return wgt


def predict_scanpath(s_early: np.ndarray, s_late: np.ndarray, layout=None,
                     n_fix: int = 16, ior_radius_frac: float = 0.04,
                     t_base_ms: float = 280.0, t_beta: float = 0.25, dur_radius_frac: float = 0.06,
                     t_min: float = 100.0, t_max: float = 900.0,
                     sacc_sigma_frac: float | None = 0.35, weights: np.ndarray | None = None) -> list[dict]:
    """Перші 3 фіксації беремо з ранньої карти (bottom-up), решту — з пізньої (top-down).

    n_fix=16 — люди за 7 с роблять у середньому 16,5 фіксацій (UEyes). Радіус гасіння 0,04 — компроміс, вибраний
    на UEyes: 0,02 дає найвищий ScanMatch, але всі фіксації скупчуються в
    найгарячішій плямі й погляд не доходить до CTA навіть у добрих макетах; 0,04 — на ~0,02 нижчий ScanMatch,
    зате відрізняє добрий макет від поганого (examples/product_good: CTA на 8-й фіксації, product_bad — ні).
    Тривалість t_k = 280 мс × (маса уваги / медіана)^0,25, у межах 100–900 мс: 280 мс — медіана фіксацій UEyes;
    з β = 0,25 важливіші місця тримають погляд довше.
    Короткі стрибки (ідея з HAT, CVPR 2024): увага далеких точок зважується гаусом відстані від попередньої
    фіксації, σ = 0,35 × більша сторона;
    середній стрибок 0,16 сторони (люди — 0,15, без штрафу було 0,30). sacc_sigma_frac=None вимикає.
    """
    H, W = s_early.shape
    f = max(1, W // 120)  # пошук піків на зменшеній карті, координати потім ×f
    s_early, s_late = s_early[::f, ::f], s_late[::f, ::f]
    if weights is not None:  # структура екрана з кроку 1 (structure_weights)
        wv = weights[::f, ::f]
        s_early, s_late = s_early * wv, s_late * wv
    h, w = s_early.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = ior_radius_frac * max(h, w)
    inhib = np.ones((h, w))
    rd = dur_radius_frac * max(h, w)
    pts, masses = [], []
    for k in range(n_fix):
        s = (s_early if k < 3 else s_late) * inhib
        if sacc_sigma_frac and pts:
            # короткі стрибки (ідея з HAT і з даних айтрекера): люди частіше переводять погляд поруч, ніж через
            # увесь екран, тож увага далеких точок зважується гаусом відстані від попередньої фіксації
            px, py = pts[-1]
            sg = sacc_sigma_frac * max(h, w)
            s = s * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sg ** 2))
        y, x = np.unravel_index(np.argmax(s), s.shape)
        # «маса» уваги в околі точки (радіус окремий від гасіння) — що більше уваги, то довша фіксація
        local = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * rd ** 2))
        masses.append(float((s_late * local).sum() / max(s_late.sum(), 1e-12)))
        pts.append((x, y))
        # гасіння повернення: точка на певний час «вимикається»
        inhib *= 1.0 - 0.95 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * r ** 2))
    # тривалість: типова людська (t_base_ms, медіана UEyes ≈ 280 мс), масштабована відносною масою уваги
    med = float(np.median(masses)) or 1e-12
    path, t_acc = [], 0.0
    for k, ((x, y), m) in enumerate(zip(pts, masses)):
        dur = float(np.clip(t_base_ms * (m / med) ** t_beta, t_min, t_max))
        el = _element_at(layout, x * f, y * f)
        path.append({"k": k + 1, "x": int(x * f), "y": int(y * f), "t_ms": round(dur),
                     "onset_ms": round(t_acc), "element": el.id if el else None,
                     "role": el.role if el else None})
        t_acc += dur + 30  # ~30 мс на саму саккаду
    return path


def scanpath_summary(path: list[dict]) -> dict:
    """Чи доходить погляд до CTA і що його відволікає до того."""
    cta_idx = next((p["k"] for p in path if p["role"] == "cta"), None)
    before = path[: (cta_idx - 1) if cta_idx else len(path)]
    traps = sorted({p["element"] for p in before if p["role"] == "decor"})
    return {
        "cta_fixation_index": cta_idx,
        "cta_time_to_first_fixation_ms": next((p["onset_ms"] for p in path if p["role"] == "cta"), None),
        "attention_traps_before_cta": traps,
        "order": [p["element"] for p in path],
    }
