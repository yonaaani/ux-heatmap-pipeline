"""Крок 8: пошук змін, що піднімають увагу до кнопки дії (CTA) .

Модель уваги кроку 3 майже не реагує на КОЛІР
кнопки (темна / помаранчева / червона заливка: 0,32 → 0,25–0,34 %), але сильно реагує на МІСЦЕ й РОЗМІР
(більша кнопка під заголовком помітно піднімає увагу). Тому шукаються місце й розмір, а не колір.

Метод (пошук з перевіркою):
1. Кандидати — та сама кнопка (пікселі її рамки) у масштабі 1,0 / 1,15 / 1,3 / 1,5 на різних місцях:
   - «scale»: на тому самому місці, лише більша;
   - «local»: у межах свого блоку (найменша картка, що містить кнопку);
   - «global»: будь-де на екрані — це вже зміна структури, вирішує дизайнер.
   Місце допустиме, якщо кнопка не перекриває інших елементів (відступ від тексту — пів висоти кнопки, але
   ≥ 16 px; від решти — 8 px), не виходить за екран, не перетинає край картки (або повністю всередині, або
   повністю поза нею), стоїть щонайменше за 16 px від країв екрана і не ламає порядок читання: у своєму блоці кнопка лишається після текстів, що були над нею,
   а будь-де на екрані — після головного заголовка (кнопка над заголовком читається раніше за сам зміст).
2. Попередній відбір — за картою першої секунди поточного макета: сума уваги в новій рамці (інтегральне
   зображення, тисячі позицій за мілісекунди). Для кожного виду беремо 3 найкращі різні місця.
3. Перевірка — кнопку справді переносимо в пікселях (старе місце зафарбовуємо фоном навколо), і модель кроку 3
   рахує карти заново. Прогноз «після» — це частка уваги з нової карти, а не з попереднього відбору.
4. Незалежне підтвердження — резервний метод кроку 3 (Itti-Koch + спектральний залишок + структура макета, інша
   модель): чи росте увага до кнопки й за ним. Так видно, що зміна не «підігнана» під одну нейромережу.

Обмеження: прогноз — моделі, а не людей (перевірити на людях можна лише новим айтрекінгом); перенесення в пікселях
не перекомпоновує сусідні елементи (кнопка стає на вільне місце).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .saliency import attention_share, temporal_maps
from .ui_parser import _ring

SCALES = (1.0, 1.15, 1.3, 1.5)
MARGIN = 8           # px між кнопкою й нетекстовими сусідами
TEXT_MARGIN = 16     # px між кнопкою й текстом (не менше пів висоти кнопки)
EDGE = 16            # px від краю екрана
MIN_GAP = 16         # px: найвужчий проміжок між групами, куди можна вставити рядок з кнопкою
SHADOW = 10          # px навколо рамки, що теж зафарбовуються (тінь кнопки)
TEXT_TYPES = ("heading", "text", "price")
PER_KIND = 3         # перевірених місць на кожен вид зміни
KINDS = ("scale", "local", "insert", "global")
KIND_UA = {"scale": "збільшити на місці", "local": "перемістити в межах блоку",
           "insert": "звільнити рядок під заголовком і поставити кнопку туди",
           "global": "перемістити на вільне місце нижче головного заголовка"}


def _integral(m: np.ndarray) -> np.ndarray:
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.float64)
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    return ii


def _box_sums(ii: np.ndarray, xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    """Сума в рамках [x, x+w) × [y, y+h) для сітки лівих верхніх кутів (xs, ys) — одним векторним кроком."""
    X, Y = np.meshgrid(xs, ys)
    return ii[Y + h, X + w] - ii[Y, X + w] - ii[Y + h, X] + ii[Y, X]


def _contains(outer, inner, frac=0.9) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    inter = max(0, min(ox + ow, ix + iw) - max(ox, ix)) * max(0, min(oy + oh, iy + ih) - max(oy, iy))
    return inter >= frac * iw * ih


def move_cta_pixels(rgb: np.ndarray, box, new_box) -> np.ndarray:  # new_box=None — лише прибрати
    """Кнопку переносимо / масштабуємо в пікселях: старе місце — колір фону навколо, нове — перемасштабована рамка."""
    H, W = rgb.shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    crop = rgb[y:y + h, x:x + w].copy()
    out = rgb.copy()
    X0, Y0, X1, Y1 = max(0, x - SHADOW), max(0, y - SHADOW), min(W, x + w + SHADOW), min(H, y + h + SHADOW)
    out[Y0:Y1, X0:X1] = _ring(rgb, (X0, Y0, X1, Y1)).astype(rgb.dtype)
    return out if new_box is None else paste_cta_pixels(out, rgb, box, new_box)


def paste_cta_pixels(img: np.ndarray, src: np.ndarray, box, new_box) -> np.ndarray:
    """Пікселі кнопки з src (рамка box) → на img у рамку new_box (з масштабуванням)."""
    H, W = img.shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    crop = src[max(0, y):y + h, max(0, x):x + w]
    nx, ny, nw, nh = (int(round(v)) for v in new_box)
    if crop.shape[:2] != (nh, nw):
        crop = np.asarray(Image.fromarray(np.ascontiguousarray(crop)).resize((nw, nh), Image.LANCZOS))
    out = img.copy()
    out[ny:ny + nh, nx:nx + nw] = crop[:max(0, min(nh, H - ny)), :max(0, min(nw, W - nx))]
    return out


def insert_row_pixels(rgb: np.ndarray, x0: int, x1: int, y: int, dy: int) -> np.ndarray:
    """Вставити порожній рядок висотою dy на висоті y у стовпці [x0, x1) (блок): вміст нижче зсувається вниз,
    розмір кадру не змінюється (нижній край блоку виходить за кадр, як на екрані з прокруткою)."""
    H = rgb.shape[0]
    col = rgb[:, x0:x1]
    out = rgb.copy()
    out[:, x0:x1] = np.concatenate([col[:y], np.repeat(col[y:y + 1], dy, 0), col[y:]], 0)[:H]
    return out


def apply_option(rgb: np.ndarray, layout, cta, opt: dict):
    """Варіант → (нові пікселі, новий макет). Те саме використовує перевірка «що буде» (auditor.apply_fixes)."""
    lay = layout.copy()
    ins = opt.get("insert")
    if ins:
        img = move_cta_pixels(rgb, cta.box, None)  # спершу прибрати кнопку зі старого місця
        img = insert_row_pixels(img, ins["x0"], ins["x1"], ins["y"], ins["dy"])
        for e in lay.elements:
            if e.id != cta.id and e.box[1] >= ins["y"] and e.box[0] < ins["x1"] and e.box[0] + e.box[2] > ins["x0"] \
                    and e.box[2] <= (ins["x1"] - ins["x0"]) + 1:
                e.box = [e.box[0], e.box[1] + ins["dy"], e.box[2], e.box[3]]
        img = paste_cta_pixels(img, rgb, cta.box, opt["box"])
    else:
        img = move_cta_pixels(rgb, cta.box, opt["box"])
    el = lay.get(cta.id)
    el.box = [float(v) for v in opt["box"]]
    el.font_size = opt["font_size"]
    return img, lay


def _insert_candidates(layout, cta, texts, cards, W, H):
    """Місця для варіанта «звільнити рядок»: перші вільні проміжки під заголовком блоку (картки з кнопкою, а
    якщо в картці заголовка немає — під головним заголовком екрана), що не перетинають жодного елемента."""
    home = min((c for c in cards if _contains(c, cta.box)), key=lambda c: c[2] * c[3], default=None)
    # заголовок — помітно більший за підпис кнопки (пункти меню того самого кегля заголовком блоку не є)
    big = [e for e in texts if e.type == "heading" and e.font_size >= 1.25 * cta.font_size]
    heads = [e for e in big if home is not None and _contains(home, e.box)]
    if heads:
        x0, y0, bw, bh = (int(round(v)) for v in home)
        x1, y1 = x0 + bw, y0 + bh
    else:
        heads = big
        x0, y0, x1, y1 = 0, 0, W, H
    head = max(heads, key=lambda e: (e.font_size, -e.box[1]), default=None)
    if head is None:
        return []
    rows = np.zeros(H + 1, bool)
    for e in layout.elements:
        if e.id == cta.id or _contains(cta.box, e.box) or e.type == "card" or _contains(e.box, cta.box) \
                or e.box[2] * e.box[3] > 0.5 * W * H or e.box[0] >= x1 or e.box[0] + e.box[2] <= x0:
            continue
        rows[max(0, int(e.box[1]) - 2):min(H, int(e.box[1] + e.box[3]) + 2)] = True
    out, y = [], int(head.box[1] + head.box[3])
    while y < min(y1, H) - 4 and len(out) < 2:
        if not rows[y]:
            g = y
            while g < H and not rows[g]:
                g += 1
            if g - y >= MIN_GAP:  # вузький проміжок (заголовок → його підпис) — частина однієї групи, не ділимо
                out.append(((y + g) // 2, x0, x1, head))
            y = g
        y += 1
    return out


def cta_options(report: dict, model=None, per_kind: int | None = None) -> dict | None:
    """Варіанти змін кнопки дії з перевіреним прогнозом уваги. None — якщо кнопки дії немає."""
    layout, rgb = report["_layout"], report["_rgb"].astype(np.uint8)
    cta = next((e for e in layout.elements if e.role == "cta"), None)
    if cta is None:
        return None
    if model is None:
        from .saliency_nn import get_neural_model
        model = get_neural_model()
    H, W = rgb.shape[:2]
    base1 = report["_maps"]["1s"]
    before = attention_share(base1, cta.box)
    fb_before = attention_share(temporal_maps(rgb, layout, None)["1s"], cta.box)
    x0, y0, w0, h0 = (int(round(v)) for v in cta.box)

    # перешкоди: усі елементи, крім самої кнопки, її вмісту, карток / контейнерів і фону на пів екрана
    occ = np.zeros((H, W), np.float32)
    texts = []
    cards = []
    for e in layout.elements:
        if e.id == cta.id or _contains(cta.box, e.box):
            continue
        if e.type in ("card",) or _contains(e.box, cta.box) or e.box[2] * e.box[3] > 0.5 * W * H:
            if e.type != "divider" and e.box[2] * e.box[3] < 0.6 * W * H:
                cards.append(e.box)
            continue
        if e.type == "divider":
            continue
        if e.type in TEXT_TYPES:
            texts.append(e)
            continue
        ex, ey, ew, eh = (int(round(v)) for v in e.box)
        occ[max(0, ey - MARGIN):max(0, ey + eh + MARGIN), max(0, ex - MARGIN):max(0, ex + ew + MARGIN)] = 1
    ii_att = _integral(base1.astype(np.float64))
    # порядок читання: головний заголовок (найбільший кегль) і тексти блоку, що стояли над кнопкою
    heads = [e for e in texts if e.type == "heading"]
    main_head = max(heads, key=lambda e: (e.font_size, -e.box[1]), default=None)
    ii_cards = [(_clipped(H, W, c), c) for c in cards]  # рамки карток (як _mask), площа перетину — аналітично
    home = min((c for c in cards if _contains(c, cta.box)), key=lambda c: c[2] * c[3], default=None)
    local_top = max((e.box[1] + e.box[3] for e in texts if home is not None and _contains(home, e.box)
                     and e.box[1] + e.box[3] <= cta.box[1] + 1), default=0)
    step = max(4, round(min(W, H) / 80))
    total = float(base1.sum())

    cands = []
    for s in SCALES:
        w, h = round(w0 * s), round(h0 * s)
        if s > 1 and (w > max(w0, 0.45 * W) or h > max(h0, 0.12 * H, 64)):
            continue
        occ_s, tm = occ.copy(), max(TEXT_MARGIN, round(h / 2))
        for e in texts:
            ex, ey, ew, eh = (int(round(v)) for v in e.box)
            occ_s[max(0, ey - tm):max(0, ey + eh + tm), max(0, ex - tm):max(0, ex + ew + tm)] = 1
        ii_occ = _integral(occ_s)
        xs, ys = np.arange(EDGE, W - w - EDGE + 1, step), np.arange(EDGE, H - h - EDGE + 1, step)
        if not len(xs) or not len(ys):
            continue
        ok = _box_sums(ii_occ, xs, ys, w, h) == 0
        for cb, _ in ii_cards:  # не перетинати край картки
            inside = _inter_grid(cb, xs, ys, w, h)
            ok &= (inside == 0) | (inside >= w * h - 0.5)
        prior = _box_sums(ii_att, xs, ys, w, h) / total * 100
        X, Y = np.meshgrid(xs, ys)
        cx, cy = x0 + w0 / 2, y0 + h0 / 2
        for kind in KINDS:
            if kind == "scale":
                if s == 1.0:
                    continue
                bx, by = round(cx - w / 2), round(cy - h / 2)
                if 0 <= bx <= W - w and 0 <= by <= H - h and occ_s[by:by + h, bx:bx + w].sum() == 0 and all(not 0 < _inter(c, (bx, by, w, h)) < w * h - 0.5 for c in cards):
                    pr = attention_share(base1, (bx, by, w, h))
                    cands.append((kind, s, (bx, by, w, h), pr))
                continue
            m = ok.copy()
            if kind == "local":
                if home is None:
                    continue
                hx, hy, hw, hh = home
                m &= (X >= hx) & (Y >= hy) & (X + w <= hx + hw) & (Y + h <= hy + hh) & (Y >= local_top + tm)
            elif main_head is not None:
                m &= Y >= main_head.box[1] + main_head.box[3] + tm
            for j in np.argsort(-np.where(m, prior, -1), axis=None)[:200]:
                r, c = np.unravel_index(j, m.shape)
                if not m[r, c]:
                    break
                cands.append((kind, s, (int(X[r, c]), int(Y[r, c]), w, h), float(prior[r, c])))

    # 3 найкращі різні місця на кожен вид (рамки, що перекриваються на ≥ 30 %, вважаємо одним місцем)
    chosen = []
    for kind in ("scale", "local", "global"):
        picked = []
        for c in sorted((c for c in cands if c[0] == kind), key=lambda c: -c[3]):
            if all(_iou(c[2], p[2]) < 0.3 for p in picked):
                picked.append(c)
            if len(picked) == (per_kind or PER_KIND):
                break
        chosen += picked

    opts = [{"kind": kind, "scale": s, "box": [float(v) for v in nb], "font_size": round(cta.font_size * s, 1),
             "prior_1s": round(pr, 2)} for kind, s, nb, pr in chosen]
    # «звільнити рядок»: кнопка (1,0 і 1,3) по центру блоку або від лівого краю заголовка — як вирівняно заголовок
    for y_ins, bx0, bx1, head in _insert_candidates(layout, cta, texts, cards, W, H):
        for s in (1.0, 1.3):
            w, h = round(w0 * s), round(h0 * s)
            tm = max(TEXT_MARGIN, round(h / 2))
            if w > (bx1 - bx0) - 2 * tm:
                continue
            nx = round((bx0 + bx1) / 2 - w / 2) if head.align == "center" else round(head.box[0])
            opts.append({"kind": "insert", "scale": s, "box": [float(nx), float(y_ins + tm), float(w), float(h)],
                         "font_size": round(cta.font_size * s, 1), "prior_1s": None,
                         "insert": {"x0": int(bx0), "x1": int(bx1), "y": int(y_ins), "dy": int(h + 2 * tm)}})

    options = []
    for o in opts:
        img, lay2 = apply_option(rgb, layout, cta, o)
        nb = o["box"]
        maps = temporal_maps(img, lay2, model)
        fb = attention_share(temporal_maps(img, lay2, None)["1s"], nb)
        options.append({**o, "kind_ua": KIND_UA[o["kind"]],
                        "after": {k: round(attention_share(maps[k], nb), 2) for k in ("1s", "3s", "7s")},
                        "fallback_before": round(fb_before, 2), "fallback_after": round(fb, 2),
                        "confirmed": fb > fb_before})
    for o in options:
        o["gain_1s"] = round(o["after"]["1s"] - before, 2)
    ranked = _ranked(options)
    return {"cta": cta.id, "before_1s": round(before, 2), "options": options, "ranked": ranked,
            "best": ranked[0] if ranked else None}


def _ranked(options: list[dict]) -> list[dict]:
    """Порядок, у якому план пробує варіанти: спершу найменш руйнівні, що дають поріг 8 % («на місці» → «у блоці» →
    «новий рядок» → «будь-де»), далі решта за приростом. Лише варіанти з приростом, підтвердженим незалежною моделлю."""
    from .evaluate import CTA_MIN_SHARE
    good = [o for o in options if o["gain_1s"] > 0 and o["confirmed"]]
    hit = sorted((o for o in good if o["after"]["1s"] >= CTA_MIN_SHARE),
                 key=lambda o: (KINDS.index(o["kind"]), -o["after"]["1s"]))
    rest = sorted((o for o in good if o["after"]["1s"] < CTA_MIN_SHARE), key=lambda o: -o["gain_1s"])
    return hit + rest


def _clipped(H, W, box) -> tuple[int, int, int, int]:
    """Рамка (x0, y0, x1, y1) у межах зображення — та сама, що заповнює _mask."""
    x, y, w, h = (int(round(v)) for v in box)
    return min(W, max(0, x)), min(H, max(0, y)), min(W, max(0, x + w)), min(H, max(0, y + h))


def _inter_grid(cb, xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    """Площа перетину рамок [x, x+w) × [y, y+h) (сітка xs × ys) з рамкою cb — те саме, що _box_sums(_integral(_mask))."""
    X, Y = np.meshgrid(xs, ys)
    iw = np.clip(np.minimum(X + w, cb[2]) - np.maximum(X, cb[0]), 0, None)
    ih = np.clip(np.minimum(Y + h, cb[3]) - np.maximum(Y, cb[1]), 0, None)
    return (iw * ih).astype(np.float64)


def _mask(H, W, box) -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    x, y, w, h = (int(round(v)) for v in box)
    m[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 1
    return m


def _inter(a, b) -> float:
    return max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)
