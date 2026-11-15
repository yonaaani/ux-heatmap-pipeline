"""Крок 6. Моторно-когнітивна UX-ергономіка.

- Закон Фіттса (формулювання Shannon, як у довідці):  T_click = a + b · log2(1 + D / W),
  D — від поточної точки погляду (крок 4) до центру кнопки, W — ширина кнопки ВЗДОВЖ напрямку руху
  (модель W′: MacKenzie & Buxton, «Extending Fitts' law to two-dimensional tasks», CHI 1992 — для прямокутних
  цілей вона описує час краще, ніж «ширина» чи «висота» окремо).
  Константи — через пропускну здатність (throughput, TP = ID / MT) за MacKenzie, «Fitts' Law», Wiley Handbook
  of HCI, 2018: миша — типово 4,9 біт/с, дотик пальцем — близько 7 біт/с; у валідній моделі вільний член a ≈ 0
  (Soukoreff & MacKenzie, IJHCS 2004), тож b = 1 / TP: миша 0,204 с/біт, палець 0,143 с/біт.
- Закон Хіка-Хаймана:  T_choice = Ic · log2(n + 1), Ic = 150 мс/біт (Card, Moran, Newell, «The Psychology of
  Human-Computer Interaction», 1983, Model Human Processor).
- Карта досяжності великого пальця (Thumb Zone, за Hoober 2013) для правші, що тримає телефон однією рукою.
"""
from __future__ import annotations

import functools
import math

import numpy as np

FITTS_A = 0.0                 # с — вільний член ≈ 0 (Soukoreff & MacKenzie 2004)
TP_MOUSE, TP_TOUCH = 4.9, 7.0  # біт/с — MacKenzie 2018
FITTS_B_MOUSE, FITTS_B_TOUCH = 1 / TP_MOUSE, 1 / TP_TOUCH
FITTS_B = FITTS_B_TOUCH       # для сумісності зі старими викликами fitts_time(d, w)
HICK_B = 0.150                # с/біт — Card, Moran, Newell 1983
MIN_TAP_PX = 44  # Apple HIG: 44 pt; Material: 48 dp
INTERACTIVE = {"button", "chip", "icon"}


def fitts_time(d: float, w: float, a: float = FITTS_A, b: float = FITTS_B) -> float:
    return a + b * math.log2(1 + d / max(w, 1e-6))


def width_along(bw: float, bh: float, dx: float, dy: float) -> float:
    """W′ — довжина відрізка через центр прямокутника bw×bh уздовж напрямку руху (dx, dy).
    Рух по горизонталі → ширина кнопки, по вертикалі → висота, по діагоналі — довша хорда, але ≤ діагоналі."""
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return min(bw, bh)
    ux, uy = abs(dx) / n, abs(dy) / n
    return min(bw / ux if ux > 1e-9 else math.inf, bh / uy if uy > 1e-9 else math.inf)


def hick_time(n: int, b: float = HICK_B) -> float:
    return b * math.log2(n + 1)


# Модель функціональної області великого пальця: Bergstrom-Lehtovirta & Oulasvirta, «Modeling the Functional
# Area of the Thumb on Mobile Touchscreen Surfaces», CHI 2014, рівняння 5:
#     f(x) = a_y·(x + h)² + s − d·(1 + a_k) − b_k      (мм, система координат руки)
# a_y = −0,013, h = −15,996 мм, a_k = −0,257, b_k = 64,655 мм — коефіцієнти авторів (20 учасників);
# s — розмах «вказівний–великий палець» (середнє 189,7 мм), d — відстань кінчика вказівного пальця на задній
# кришці від початку координат. Початок — на боковій грані, де її перетинає лінія вказівного пальця (вісь y),
# вісь x — уздовж грані. Тримання однією рукою — як у прикладі авторів для смартфона: початок за 50 мм від
# нижнього кута, кут руки до грані α = 110°, d = 80 мм. Похибка моделі (перехресна перевірка) — RMSE 16,0 мм.
THUMB_AY, THUMB_H, THUMB_AK, THUMB_BK = -0.013, -15.996, -0.257, 64.655
THUMB_S, THUMB_D, THUMB_ALPHA, THUMB_ORIGIN_MM, THUMB_RMSE = 189.7, 80.0, 110.0, 50.0, 16.0
# фізичний розмір екрана скриншота невідомий — беремо типову ширину екрана смартфона 6,1–6,7″ (≈ 65–71 мм)
SCREEN_W_MM = 68.0


def thumb_reach(xm: np.ndarray, s: float = THUMB_S, d: float = THUMB_D) -> np.ndarray:
    """Межа досяжності f(x), мм — рівняння 5 Bergstrom-Lehtovirta & Oulasvirta (2014)."""
    return THUMB_AY * (xm + THUMB_H) ** 2 + s - d * (1 + THUMB_AK) - THUMB_BK


@functools.lru_cache(maxsize=32)
def _thumb_zone_cached(h: int, w: int, left_handed: bool, screen_w_mm: float) -> np.ndarray:
    z = _thumb_zone_map(h, w, left_handed, screen_w_mm)
    z.setflags(write=False)  # спільний для всіх аналізів екрана цього розміру — лише для читання
    return z


def thumb_zone_map(h: int, w: int, left_handed: bool = False, screen_w_mm: float = SCREEN_W_MM) -> np.ndarray:
    """Карта зон пальця; для того самого розміру екрана рахується один раз (кеш, масив лише для читання)."""
    return _thumb_zone_cached(int(h), int(w), bool(left_handed), float(screen_w_mm))


def _thumb_zone_map(h: int, w: int, left_handed: bool = False, screen_w_mm: float = SCREEN_W_MM) -> np.ndarray:
    """0 = легка (у функціональній області пальця), 1 = середня (за межею, але в межах похибки моделі 16 мм —
    невелике потягування), 2 = важка (треба перехопити телефон). Правша тримає телефон правою рукою за праву грань."""
    mm = screen_w_mm / w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = (xx + 0.5) * mm, (yy + 0.5) * mm  # мм від лівого верхнього кута
    ox, oy = (0.0 if left_handed else w * mm), h * mm - THUMB_ORIGIN_MM  # початок на боковій грані
    ux, uy = 0.0, -1.0                            # уздовж грані вгору
    nx, ny = (1.0 if left_handed else -1.0), 0.0  # всередину екрана
    a = math.radians(THUMB_ALPHA)
    yax, yay = math.cos(a) * ux + math.sin(a) * nx, math.cos(a) * uy + math.sin(a) * ny  # вісь y (лінія пальця)
    xax, xay = math.sin(a) * ux - math.cos(a) * nx, math.sin(a) * uy - math.cos(a) * ny  # вісь x ⟂ y
    lx = (px - ox) * xax + (py - oy) * xay
    ly = (px - ox) * yax + (py - oy) * yay
    f = thumb_reach(lx)
    zone = np.full((h, w), 2, dtype=np.uint8)
    zone[ly <= f + THUMB_RMSE] = 1
    zone[ly <= f] = 0
    return zone


ZONE_NAMES = {0: "легка", 1: "середня", 2: "важка"}

# Групи вибору для закону Хіка: n — кількість пунктів, з яких обирають ОДИН (меню, вкладки, панель кнопок,
# список, ряд чипів), а не всі кнопки екрана. Контейнери — класи Rico з кроку 1 (meta["containers"]).
MENU_CONTAINERS = {"Multi-Tab": "вкладки", "Button Bar": "панель кнопок", "Bottom Navigation": "нижня навігація",
                   "Drawer": "бокове меню", "Toolbar": "панель інструментів"}
ITEM_TYPES = {"button", "chip", "icon", "text"}
MAX_LINK_WORDS = 3  # короткий текст у ряду — пункт меню/посилання; довший — звичайний текст


def _inside(box, cont, tol=0.0):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    x, y, w, h = cont
    return x - tol <= cx <= x + w + tol and y - tol <= cy <= y + h + tol


def _words(text: str) -> int:
    """Слова з ≥ 2 літер: однолітерне «сміття» OCR від іконок поруч (напр. «д Flight + Hotel») не рахуємо."""
    return sum(1 for t in (text or "").split() if sum(ch.isalpha() for ch in t) >= 2)


def _is_item(el) -> bool:
    if el.type not in ITEM_TYPES or el.role in ("heading", "decor"):
        return False
    return el.type != "text" or 0 < _words(el.text) <= MAX_LINK_WORDS


def choice_groups(layout, containers: list[dict] | None = None) -> list[dict]:
    """Групи альтернатив: [{kind, n, box, ids, hick_s}], від найбільшої."""
    els = [e for e in layout.elements if _is_item(e)]
    used, groups = set(), []

    def add(kind, members, box=None):
        if len(members) < 2:
            return
        xs = [m.box[0] for m in members] + [m.box[0] + m.box[2] for m in members]
        ys = [m.box[1] for m in members] + [m.box[1] + m.box[3] for m in members]
        box = box or [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        groups.append({"kind": kind, "n": len(members), "box": [round(v) for v in box],
                       "ids": [m.id for m in members], "hick_s": round(hick_time(len(members)), 3)})
        used.update(m.id for m in members)

    # 1) контейнери-меню з кроку 1
    for c in containers or []:
        if c.get("label") in MENU_CONTAINERS:
            mem = [e for e in els if e.id not in used and _inside(e.box, c["box"], tol=2)]
            add(MENU_CONTAINERS[c["label"]], mem, c["box"])
    # 2) списки: пункти List Item один під одним (схожа ліва межа й ширина)
    items = sorted((c["box"] for c in containers or [] if c.get("label") == "List Item"), key=lambda b: b[1])
    run = []
    for b in items:
        if run and (abs(b[0] - run[-1][0]) > 0.05 * layout.width or abs(b[2] - run[-1][2]) > 0.15 * max(b[2], 1)):
            if len(run) >= 3:
                groups.append({"kind": "список", "n": len(run), "box": [round(run[0][0]), round(run[0][1]),
                               round(run[0][2]), round(run[-1][1] + run[-1][3] - run[0][1])], "ids": [],
                               "hick_s": round(hick_time(len(run)), 3)})
            run = []
        run.append(b)
    if len(run) >= 3:
        groups.append({"kind": "список", "n": len(run), "box": [round(run[0][0]), round(run[0][1]), round(run[0][2]),
                       round(run[-1][1] + run[-1][3] - run[0][1])], "ids": [], "hick_s": round(hick_time(len(run)), 3)})
    # 3) ряди однотипних пунктів (чипи, кнопки, короткі посилання): одна лінія, схожа висота, малі проміжки
    rest = sorted((e for e in els if e.id not in used), key=lambda e: (e.box[1] + e.box[3] / 2, e.box[0]))
    rows = []
    for e in rest:
        cy, hh = e.box[1] + e.box[3] / 2, e.box[3]
        for r in rows:
            r0 = r[0]
            # одна лінія за центром; висота може відрізнятись до 3 разів (вкладка-чип поруч із текстовими пунктами)
            if abs(cy - (r0.box[1] + r0.box[3] / 2)) <= 0.6 * min(hh, r0.box[3]) \
                    and 1 / 3 <= hh / max(r0.box[3], 1) <= 3:
                r.append(e)
                break
        else:
            rows.append([e])
    for r in rows:
        r.sort(key=lambda e: e.box[0])
        seg = [r[0]]
        for a, b in zip(r, r[1:]):
            if b.box[0] - (a.box[0] + a.box[2]) <= 3 * max(a.box[3], b.box[3]):
                seg.append(b)
            else:
                if len(seg) >= 3:
                    add("ряд", seg)
                seg = [b]
        if len(seg) >= 3:
            add("ряд", seg)
    return sorted(groups, key=lambda g: -g["n"])


def ergonomics(layout, scanpath: list[dict] | None = None, mobile: bool | None = None,
               containers: list[dict] | None = None) -> dict:
    h, w = layout.height, layout.width
    # пристрій: відомий із макета (плагін Figma) або за розміром — вузький макет чи портретний скриншот телефона
    # (висота 1,3–2,4 × ширини); довша сторінка — десктопна, телефони мають пропорції до ~2,2 (20:9)
    device = getattr(layout, "device", None)
    if device not in ("mobile", "tablet", "desktop"):
        device = "mobile" if (w <= 480 or 1.3 * w <= h <= 2.4 * w) else "desktop"
    if mobile is not None:
        device = "mobile" if mobile else "desktop"
    mobile = device in ("mobile", "tablet")  # дотик пальцем: закон Фіттса для пальця і мінімальна ціль 44 px
    interactive = [e for e in layout.elements if e.type in INTERACTIVE or e.role == "cta"]
    b = FITTS_B_TOUCH if mobile else FITTS_B_MOUSE
    groups = choice_groups(layout, containers)
    n_max = groups[0]["n"] if groups else 0
    out = {"mobile": mobile, "device": device, "input": "палець" if mobile else "миша", "fitts_b_s_per_bit": round(b, 4),
           # закон Хіка — для найбільшої групи вибору (меню, вкладки, список, ряд); усі групи — у choice_groups
           "n_choices": n_max, "hick_choice_time_s": round(hick_time(n_max), 3) if n_max else 0.0,
           "choice_groups": groups, "n_interactive": len(interactive), "targets": []}
    # зона великого пальця — лише для телефона (планшет тримають інакше, на десктопі — миша)
    zones = thumb_zone_map(h, w) if device == "mobile" else None

    # стартова точка руху — остання фіксація погляду перед CTA (або центр екрана)
    start = (w / 2, h / 2)
    if scanpath:
        cta_k = next((p["k"] for p in scanpath if p["role"] == "cta"), None)
        prev = [p for p in scanpath if cta_k is None or p["k"] < cta_k]
        if prev:
            start = (prev[-1]["x"], prev[-1]["y"])

    for el in interactive:
        x, y, bw, bh = el.box
        cx, cy = x + bw / 2, y + bh / 2
        d = math.hypot(cx - start[0], cy - start[1])
        eff_w = width_along(bw, bh, cx - start[0], cy - start[1])  # W′ уздовж напрямку руху
        rec = {"id": el.id, "role": el.role, "distance_px": round(d), "width_px": round(eff_w),
               "fitts_time_s": round(fitts_time(d, eff_w, b=b), 3),
               "index_of_difficulty_bits": round(math.log2(1 + d / max(eff_w, 1e-6)), 2),
               # мінімум 44×44 px — правило для пальця (Apple HIG / Material); для миші (десктоп) не перевіряється
               "tap_target_checked": mobile,
               "tap_target_ok": (bw >= MIN_TAP_PX and bh >= MIN_TAP_PX) if mobile else True}
        if zones is not None:
            ix, iy = int(min(max(cx, 0), w - 1)), int(min(max(cy, 0), h - 1))
            rec["thumb_zone"] = ZONE_NAMES[int(zones[iy, ix])]
        out["targets"].append(rec)
    out["_zones"] = zones
    return out
