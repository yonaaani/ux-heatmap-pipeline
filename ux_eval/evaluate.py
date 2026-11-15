"""Пайплайн кроків 1–7: один звіт із проблемами та виправленнями.

Тут же реалізовано крок 7 (частка уваги CTA) і поелементні перевірки кроку 2.
Крок 8 (перетворення звіту на план дій) — у auditor.py.
"""
from __future__ import annotations

import os
import time

import functools

import numpy as np

from .clutter import clutter_analysis
from .ui_parser import _ring
from .color import (apca_lc_many, CVD_MATRICES, adjust_lightness_for_apca, apca_lc, apca_min_lc, delta_e2000,
                    delta_e_ok, delta_e_rgb, hex_to_rgb, lab_to_rgb, rgb_to_hex, rgb_to_lab, simulate_cvd)
from .ergonomics import MIN_TAP_PX, ergonomics
from .layout import TEXT_TYPES, Layout, render
from .saliency import attention_share, temporal_maps
from .scanpath import predict_scanpath, scanpath_summary
from .segmentation import segment, whitespace_ratio

# Пороги. CTA_MIN_SHARE і CTA_MIN_DELTA_E — пороги кроку 7,
# CLEAN_MIN — робочий поріг, що потребує калібрування.
CTA_MIN_SHARE = 8.0
CTA_MIN_DELTA_E = 15.0
CLEAN_MIN = 25.0  # бал чистоти = перцентиль серед інтерфейсів UEyes: < 25 — перевантаженіший за 75 % реальних екранів
PENALTY = {"critical": 20, "major": 10, "minor": 4}


def _issue(severity, code, element, message, metrics=None, fix=None) -> dict:
    return {"severity": severity, "code": code, "element": element, "message": message,
            "metrics": metrics or {}, "fix": fix}


def _cvd_delta_e(rgb1, rgb2) -> dict[str, float]:
    return dict(_cvd_delta_e_cached(tuple(float(v) for v in rgb1), tuple(float(v) for v in rgb2)))


@functools.lru_cache(maxsize=4096)  # ті самі пари кольорів повторюються в повторних аналізах (крок 8, плагін)
def _cvd_delta_e_cached(rgb1: tuple, rgb2: tuple) -> tuple:
    return tuple((k, round(delta_e_rgb(simulate_cvd(np.array(rgb1), k), simulate_cvd(np.array(rgb2), k)), 1))
                 for k in CVD_MATRICES)


@functools.lru_cache(maxsize=4096)
def _lc_cvd(text_hex: str, bg_hex: str) -> tuple:
    """APCA Lc тексту очима людини з кожним типом дальтонізму (матриці Machado); запам'ятовується за парою кольорів."""
    return tuple((k, round(apca_lc(simulate_cvd(np.array(hex_to_rgb(text_hex), float), k),
                                   simulate_cvd(np.array(hex_to_rgb(bg_hex), float), k)), 1)) for k in CVD_MATRICES)


@functools.lru_cache(maxsize=4096)
def _delta_e_ok_hex(a_hex: str, b_hex: str) -> float:
    return round(delta_e_ok(hex_to_rgb(a_hex), hex_to_rgb(b_hex)), 1)


def suggest_cta_color(fill_hex: str, text_hex: str, bg_hex: str, font_size: float, bold: bool) -> dict | None:
    """Результат запам'ятовується за кольорами й шрифтом (копія словника); підбір — _suggest_cta_color."""
    r = _suggest_cta_color(fill_hex, text_hex, bg_hex, float(font_size), bool(bold))
    return dict(r) if r is not None else None


@functools.lru_cache(maxsize=1024)
def _suggest_cta_color(fill_hex: str, text_hex: str, bg_hex: str, font_size: float, bold: bool) -> dict | None:
    """Шукає найближчий до фірмового колір кнопки, що помітний на фоні (ΔE00 ≥ 25),
    лишається помітним за всіх типів дальтонізму (ΔE00 ≥ 15) і дає читабельний текст."""
    orig = np.array(hex_to_rgb(fill_hex), dtype=np.float64)
    bg = np.array(hex_to_rgb(bg_hex), dtype=np.float64)
    _, a0, b0 = rgb_to_lab(orig)
    h0 = np.degrees(np.arctan2(b0, a0))
    need_lc = apca_min_lc(font_size, bold)
    text_options = [hex_to_rgb(text_hex), (255, 255, 255), (20, 20, 28)]

    # уся сітка кандидатів (відтінок × світлота × насиченість) — векторно
    hue, L, C = np.meshgrid(np.radians(h0 + np.arange(-180, 181, 10)),
                            np.arange(25, 76, 5.0), np.array([35, 50, 65, 80.0]), indexing="ij")
    lab = np.stack([L.ravel(), (C * np.cos(hue)).ravel(), (C * np.sin(hue)).ravel()], axis=-1)
    rgb = lab_to_rgb(lab)
    ok = np.all((rgb > -0.5) & (rgb < 255.5), axis=1)
    rgb, lab = np.clip(rgb[ok], 0, 255), lab[ok]
    ok = delta_e2000(lab, rgb_to_lab(bg)) >= 25
    for k in CVD_MATRICES:
        ok &= delta_e2000(rgb_to_lab(simulate_cvd(rgb, k)), rgb_to_lab(simulate_cvd(bg, k))) >= CTA_MIN_DELTA_E
    rgb, lab = rgb[ok], lab[ok]
    cost = delta_e2000(lab, rgb_to_lab(orig))
    for i in np.argsort(cost):  # від найближчого до фірмового кольору
        text = next((t for t in text_options if abs(apca_lc(t, rgb[i])) >= need_lc), None)
        if text is not None:
            return {"fill": rgb_to_hex(rgb[i]), "color": rgb_to_hex(text)}
    return None


def suggest_text_fix(text_hex: str, bg_hex: str, need: float, has_fill: bool) -> dict | None:
    """Спочатку змінюємо світлоту тексту; якщо не вдається — пробуємо білий/темний
    текст; для кнопок і чипів — затемнюємо/висвітлюємо заливку. Результат запам'ятовується (копія словника)."""
    r = _suggest_text_fix(text_hex, bg_hex, float(need), bool(has_fill))
    return dict(r) if r is not None else None


@functools.lru_cache(maxsize=4096)
def _suggest_text_fix(text_hex: str, bg_hex: str, need: float, has_fill: bool) -> dict | None:
    text, bg = hex_to_rgb(text_hex), hex_to_rgb(bg_hex)
    fixed = adjust_lightness_for_apca(text, bg, need)
    if fixed is not None:
        return {"color": rgb_to_hex(fixed)}
    for alt in ((255, 255, 255), (30, 27, 46)):
        if abs(apca_lc(alt, bg)) >= need:
            return {"color": rgb_to_hex(alt)}
    if has_fill:
        # кандидати в тому самому порядку, що й раніше (L−1, L+1, L−2, L+2, …), усі одним масивом
        L, a, b = rgb_to_lab(np.array(bg, dtype=np.float64))
        steps = np.repeat(np.arange(1, 101, dtype=np.float64), 2)
        newL = L + steps * np.tile([-1.0, 1.0], 100)
        newL = newL[(newL >= 0) & (newL <= 100)]
        if newL.size:
            cand = np.clip(lab_to_rgb(np.stack([newL, np.full_like(newL, a), np.full_like(newL, b)], axis=-1)), 0, 255)
            ok = np.abs(apca_lc_many(np.asarray(text, dtype=np.float64), cand)) >= need
            if ok.any():
                return {"fill": rgb_to_hex(cand[int(np.argmax(ok))])}
    return None


def _desaturate(hex_color: str, factor: float = 0.3) -> str:
    L, a, b = rgb_to_lab(np.array(hex_to_rgb(hex_color), dtype=np.float64))
    L = L + (88 - L) * 0.5  # світліше й пастельніше
    return rgb_to_hex(np.clip(lab_to_rgb(np.array([L, a * factor, b * factor])), 0, 255))


_PARSER = None


def get_parser():
    """Моделі кроку 1 вантажаться один раз; якщо їх немає — None (тоді OpenCV).

    Без відеокарти — теж None: на CPU працює математичний Fallback (OpenCV, Graceful Degradation), як і в кроці 3,
    бо нейромережі кроку 1 на CPU дуже повільні. Примусово ввімкнути їх на CPU: UX_STEP1_CPU_MODELS=1."""
    global _PARSER
    if _PARSER is None:
        import torch
        if not torch.cuda.is_available() and os.environ.get("UX_STEP1_CPU_MODELS") != "1":
            _PARSER = False
            return None
        try:
            from .ui_parser import UIParser
            _PARSER = UIParser()
        except Exception:  # noqa: BLE001 — немає ultralytics / paddleocr
            _PARSER = False
    return _PARSER or None


def evaluate(layout: Layout | None = None, rgb: np.ndarray | None = None, saliency_model=None,
             use_models: bool = True, seg_meta: dict | None = None, fast_clutter: bool | None = None) -> dict:
    """Повний пайплайн. Можна передати JSON-макет (структура відома) або лише скріншот.

    seg_meta — результат кроку 1 для скриншота, уже знайдений раніше (разом із layout і rgb): тоді крок 1 не
    повторюється, але решта аналізу йде шляхом скриншота (фон із пікселів, спливні вікна) — для «було → стало»
    у кроці 8.

    fast_clutter — швидкий наближений режим кроку 5 (320 px, float32; ux_eval.clutter.clutter_analysis(fast=True)).
    None (типово) — швидкий лише без відеокарти: на CPU точний режим займає понад 1 с,
    а згода з оцінками людей та сама."""
    t0 = time.perf_counter()
    timings: dict[str, int] = {}

    def tick(step: str, start: float) -> float:
        now = time.perf_counter()
        timings[step] = round((now - start) * 1000)
        return now

    # Крок 1 — структура
    t = time.perf_counter()
    source = "layout JSON" if seg_meta is None else "screenshot (структура з попереднього аналізу)"
    if layout is None:
        parser = get_parser() if use_models else None
        usable = parser and (parser.omni or parser.yolo_ui or parser.ocr)
        if usable:
            layout, seg_meta = parser.parse(rgb)
            source = "screenshot (" + ", ".join(k for k, v in seg_meta["models"].items() if v.startswith("OK")) + ")"
        else:
            layout, seg_meta = segment(rgb)
            source = "screenshot (OpenCV fallback)"
    if rgb is None:
        rgb = render(layout)
    seg = {"source": source,
           "whitespace_ratio": seg_meta["whitespace_ratio"] if seg_meta else whitespace_ratio(layout),
           "n_elements": len([e for e in layout.elements if e.type not in ("card", "divider")])}
    if seg_meta and "timings_ms" in seg_meta:
        seg["model_timings_ms"] = seg_meta["timings_ms"]
    t = tick("1_segmentation", t)

    # Крок 3 — теплові карти; крок 5 — шум (крок 2 рахується поелементно нижче)
    # нейромережі кроку 3 (донавчені TempSAL + TranSalNet) — типово на GPU; на CPU get_neural_model() дає None,
    # і працює резервний метод (Itti-Koch + спектральний залишок + структура макета)
    if saliency_model is None and use_models:
        from .saliency_nn import get_neural_model
        saliency_model = get_neural_model()
    maps = temporal_maps(rgb, layout, saliency_model)
    seg["saliency"] = type(saliency_model).__name__ if hasattr(saliency_model, "predict_maps") else "fallback"
    t = tick("3_saliency", t)
    if fast_clutter is None:
        import torch
        fast_clutter = not torch.cuda.is_available()
    clutter = clutter_analysis(rgb, fast=fast_clutter)
    t = tick("5_clutter", t)

    # Крок 4 — траєкторія погляду
    # зони, куди люди майже не дивляться (системні панелі Android, сторінка за спливним вікном), приглушуються ×0,5
    # (прибирає фіксації на значках рядка стану). Лише для скріншотів — у JSON-макеті таких зон немає.
    wgt = None
    if seg_meta is not None:
        from .scanpath import ZONE_W, structure_weights
        from .ui_parser import _system_bar_top
        sp_modals = [m["box"] for m in seg_meta.get("containers", []) if m["label"] == "Modal"]
        wgt = structure_weights(layout, rgb.shape[0], rgb.shape[1], sp_modals, _system_bar_top(rgb), zone_w=ZONE_W)
    path = predict_scanpath(maps["1s"], maps["7s"], layout, weights=wgt)
    path_sum = scanpath_summary(path)
    t = tick("4_scanpath", t)

    # Крок 6 — ергономіка
    ergo = ergonomics(layout, path, containers=(seg_meta or {}).get("containers"))
    t = tick("6_ergonomics", t)

    issues: list[dict] = []
    elements: list[dict] = []
    # відкрите спливне вікно затемнює решту сторінки: контраст позаду нього — не помилка дизайну,
    # тому елементи поза вікном лише позначаємо (behind_modal) і не перевіряємо за APCA
    modals = [m["box"] for m in (seg_meta or {}).get("containers", []) if m["label"] == "Modal"]

    def behind_modal(box):
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        return bool(modals) and not any(mx <= cx <= mx + mw and my <= cy <= my + mh for mx, my, mw, mh in modals)

    under_cache: dict = {}

    def under_of(o):
        """Фон під елементом: у JSON-макеті — з ієрархії; на скріншоті — справжні пікселі навколо рамки
        (у списку можуть бути дублікати однієї кнопки, і «батьком» стала б вона сама)."""
        key = (id(o), tuple(o.box), o.fill)  # той самий елемент питають кілька разів (порівняння кнопок) — рахуємо раз
        if key not in under_cache:
            if seg_meta is None:
                under_cache[key] = layout.underlying_color(o)
            else:
                x0, y0, bw, bh = o.box
                under_cache[key] = rgb_to_hex(_ring(rgb, (x0, y0, x0 + bw, y0 + bh)))
        return under_cache[key]

    map_totals = {k: m.sum() for k, m in maps.items()}  # той самий тип і порядок підсумовування, що й у attention_share
    for el in layout.elements:
        if el.type in ("card", "divider"):
            continue
        hidden = behind_modal(el.box)
        under = under_of(el)
        rec = {"id": el.id, "type": el.type, "role": el.role,
               "attention": {k: round(attention_share(m, el.box, map_totals[k]), 1) for k, m in maps.items()}}
        x, y, w, h = (int(v) for v in el.box)
        rec["clutter"] = round(float(clutter["map"][y:y + h, x:x + w].mean()), 3)

        # ---- крок 2: колірна помітність блока відносно фону (ΔE00 у CIELAB) + Oklab для порівняння
        main = el.fill if (el.fill and el.type not in ("text", "heading", "price")) else el.color
        if main:
            rec["delta_e_bg"] = round(delta_e_rgb(hex_to_rgb(main), hex_to_rgb(under)), 1)
            rec["delta_e_ok"] = _delta_e_ok_hex(main, under)
            if el.type in ("button", "chip", "input", "icon"):
                rec["delta_e_cvd"] = _cvd_delta_e(hex_to_rgb(main), hex_to_rgb(under))

        # ---- читабельність тексту (APCA)
        if hidden:
            rec["behind_modal"] = True
        if el.type in TEXT_TYPES and el.text and not hidden:
            text_bg = el.fill if el.type in ("button", "chip") and el.fill else (el.bg or under)
            lc = apca_lc(hex_to_rgb(el.color), hex_to_rgb(text_bg))
            # два рівні APCA: Bronze (75/60/45) — межа «погано читається»; повна таблиця fontLookupAPCA
            # (кегль × товщина) — суворіша рекомендація для тексту, який читають, а не лише помічають
            need = apca_min_lc(el.font_size, el.bold, mode="bronze")
            need_full = apca_min_lc(el.font_size, el.bold, mode="full")
            rec["apca_lc"], rec["apca_min"], rec["apca_min_full"] = round(lc, 1), need, need_full
            # той самий текст очима людини з дальтонізмом (матриці Machado): контраст може впасти
            lc_cvd = dict(_lc_cvd(el.color, text_bg))
            rec["apca_lc_cvd"] = lc_cvd
            worst = min(lc_cvd, key=lambda k: abs(lc_cvd[k]))
            if abs(lc) >= need and abs(lc_cvd[worst]) < need - 15:
                issues.append(_issue(
                    "minor", "text_cvd_low_contrast", el.id,
                    f"Для людей з {worst} текст «{el.text[:30]}» читається гірше: |Lc| = {abs(lc_cvd[worst]):.0f} "
                    f"(звичайний зір {abs(lc):.0f}, потрібно ≥ {need:.0f}).",
                    {"apca_lc": round(lc, 1), "apca_lc_cvd": lc_cvd, "required": need}))
            if need <= abs(lc) < need_full:
                issues.append(_issue(
                    "minor", "text_contrast_below_apca_full", el.id,
                    f"Текст «{el.text[:30]}» читається, але нижче повної таблиці APCA: |Lc| = {abs(lc):.0f}, "
                    f"для {el.font_size:.0f}px {'жирного' if el.bold else 'звичайного'} рекомендовано ≥ {need_full:.0f} "
                    f"(або збільшити кегль).",
                    {"apca_lc": round(lc, 1), "required_full": need_full, "required_bronze": need}))
            if abs(lc) < need:
                fixed = suggest_text_fix(el.color, text_bg, need, has_fill=text_bg == el.fill)
                sev = "critical" if abs(lc) < need - 30 else "major"
                issues.append(_issue(
                    sev, "low_text_contrast", el.id,
                    f"Текст «{el.text[:30]}» погано читається: |Lc| = {abs(lc):.0f}, "
                    f"для {el.font_size:.0f}px потрібно ≥ {need:.0f} (APCA).",
                    {"apca_lc": round(lc, 1), "required": need},
                    {"id": el.id, "set": fixed} if fixed is not None else None))

        # ---- помітність кнопки дії (ΔE00, CVD, частка уваги)
        if el.role == "cta":
            de = delta_e_rgb(hex_to_rgb(el.fill or el.color), hex_to_rgb(under))
            cvd = _cvd_delta_e(hex_to_rgb(el.fill or el.color), hex_to_rgb(under))
            rec["delta_e_bg"], rec["delta_e_cvd"] = round(de, 1), cvd
            # чи CTA найпомітніша за кольором серед інтерактивних елементів (інакше акцент «розмитий»)
            rivals = [o for o in layout.elements if o is not el and o.type in ("button", "chip") and o.fill]
            stronger = [o.id for o in rivals
                        if delta_e_rgb(hex_to_rgb(o.fill), hex_to_rgb(under_of(o))) > de + 5]
            rec["color_rank_stronger"] = stronger
            if stronger and de >= CTA_MIN_DELTA_E:
                issues.append(_issue(
                    "minor", "cta_not_most_salient_color", el.id,
                    f"Кнопка «{el.text}» не найяскравіша за кольором: {len(stronger)} інш. кнопок/чипів "
                    f"контрастують із фоном сильніше (ΔE00 кнопки {de:.1f}).",
                    {"delta_e00": round(de, 1), "stronger": stronger}))
            worst_cvd = min(cvd, key=cvd.get)
            color_fix = None
            if de < CTA_MIN_DELTA_E or cvd[worst_cvd] < CTA_MIN_DELTA_E:
                sug = suggest_cta_color(el.fill or el.color, el.color, under, el.font_size, el.bold)
                color_fix = {"id": el.id, "set": sug} if sug else None
                if sug:  # нова пара «заливка + текст» уже вирішує й читабельність напису
                    for i in issues:
                        if i["element"] == el.id and i["code"] == "low_text_contrast":
                            i["fix"] = color_fix
            if de < CTA_MIN_DELTA_E:
                issues.append(_issue(
                    "critical", "cta_low_color_salience", el.id,
                    f"Кнопка «{el.text}» майже зливається з фоном: ΔE00 = {de:.1f} (потрібно ≥ {CTA_MIN_DELTA_E:.0f}).",
                    {"delta_e00": round(de, 1)}, color_fix))
            elif cvd[worst_cvd] < CTA_MIN_DELTA_E:
                issues.append(_issue(
                    "major", "cta_cvd_conflict", el.id,
                    f"Для людей з {worst_cvd} кнопка «{el.text}» зливається з фоном: ΔE00 = {cvd[worst_cvd]:.1f}.",
                    {"delta_e00_cvd": cvd}, color_fix))

            share = rec["attention"]["1s"]
            if share < CTA_MIN_SHARE:
                # більша й жирніша кнопка; якщо й колір слабкий — разом із новою парою кольорів (інакше
                # виправлення кольору не дає уваги, а збільшення без кольору — контрасту)
                grow = {"box": [x - 8, y - 8, w + 16, h + 16], "font_size": el.font_size + 2, "bold": True}
                fix = {"id": el.id, "set": {**(color_fix["set"] if color_fix else {}), **grow}}
                issues.append(_issue(
                    "critical", "cta_low_attention", el.id,
                    f"Кнопка «{el.text}» отримує лише {share:.1f}% уваги в першу секунду "
                    f"(поріг {CTA_MIN_SHARE:.0f}%). Ризик втрати конверсії.",
                    {"attention_1s": share, "threshold": CTA_MIN_SHARE}, fix))
        elements.append(rec)

    # ---- «пастки уваги»: декор, що забирає більше уваги, ніж CTA, і притягує погляд
    # щільніше (частка уваги на одиницю площі) — велика спокійна картинка пасткою не є
    cta = next((e for e in elements if e["role"] == "cta"), None)

    def density(e):
        _, _, bw, bh = layout.get(e["id"]).box
        return e["attention"]["1s"] / max(bw * bh / (layout.width * layout.height) * 100, 1e-6)

    if cta:
        for e in elements:
            if (e["role"] == "decor" and e["attention"]["1s"] > cta["attention"]["1s"]
                    and density(e) > density(cta)):
                el = layout.get(e["id"])
                set_ = {}
                if el.palette:
                    set_["palette"] = [_desaturate(c) for c in el.palette]
                if el.fill:
                    set_["fill"] = _desaturate(el.fill)
                issues.append(_issue(
                    "major", "attention_trap", e["id"],
                    f"Декоративний елемент «{e['id']}» забирає {e['attention']['1s']:.1f}% уваги — "
                    f"більше, ніж кнопка дії ({cta['attention']['1s']:.1f}%).",
                    {"attention_1s": e["attention"]["1s"]},
                    {"id": e["id"], "set": set_} if set_ else None))

    # ---- візуальне перевантаження
    if clutter["cleanliness_score"] < CLEAN_MIN:
        decor = sorted((e for e in elements if e["role"] == "decor"), key=lambda e: -e["clutter"])
        issues.append(_issue(
            "major", "visual_clutter", None,
            f"Макет перевантажений: чистота {clutter['cleanliness_score']:.0f}/100 (поріг {CLEAN_MIN:.0f}).",
            {"cleanliness": clutter["cleanliness_score"]},
            {"remove": [e["id"] for e in decor[:2]]} if decor else None))

    # ---- траєкторія погляду (крок 4)
    if cta and path_sum["cta_fixation_index"] is None:
        issues.append(_issue(
            "major", "cta_not_fixated", cta["id"],
            f"За прогнозом перших {len(path)} фіксацій погляд взагалі не доходить до кнопки дії.",
            {"order": path_sum["order"]}))
    known_traps = {i["element"] for i in issues if i["code"] == "attention_trap"}
    for trap in path_sum["attention_traps_before_cta"]:
        if trap not in known_traps:
            issues.append(_issue(
                "minor", "scanpath_detour", trap,
                f"Погляд зупиняється на декоративному «{trap}» раніше, ніж на кнопці дії.",
                {"cta_fixation_index": path_sum["cta_fixation_index"]}))

    # ---- ергономіка (крок 6)
    for tgt in ergo["targets"]:
        el = layout.get(tgt["id"])
        x, y, bw, bh = el.box
        if not tgt["tap_target_ok"]:
            issues.append(_issue(
                "major" if el.role == "cta" else "minor", "small_tap_target", el.id,
                f"Ціль «{el.text or el.id}» замала для пальця: {bw:.0f}×{bh:.0f}px (мінімум {MIN_TAP_PX}×{MIN_TAP_PX}).",
                {"size": [bw, bh]},
                {"id": el.id, "set": {"box": [x, y - max(0, MIN_TAP_PX - bh) / 2,
                                              max(bw, MIN_TAP_PX), max(bh, MIN_TAP_PX)]}}))
        if el.role == "cta" and tgt.get("thumb_zone") == "важка":
            new_y = layout.height - bh - 24
            issues.append(_issue(
                "major", "cta_hard_thumb_zone", el.id,
                f"Кнопка «{el.text}» у важкодоступній для великого пальця зоні (верх екрана).",
                {"thumb_zone": tgt["thumb_zone"]},
                {"id": el.id, "set": {"box": [x, new_y, bw, bh]}}))
    if ergo["n_choices"] > 7:
        issues.append(_issue(
            "minor", "too_many_choices", None,
            f"У групі вибору «{ergo['choice_groups'][0]['kind']}» {ergo['n_choices']} пунктів: час вибору за законом "
            f"Хіка ≈ {ergo['hick_choice_time_s']:.2f} с. Варто згрупувати або лишити 5–7.",
            {"n": ergo["n_choices"]}))

    timings["2_7_checks"] = round((time.perf_counter() - t) * 1000)
    penalty = sum(PENALTY[i["severity"]] for i in issues)
    # 70 % — штрафи за проблеми (плавно спадає), 30 % — чистота макета
    overall = 0.7 * 100 * float(np.exp(-penalty / 80)) + 0.3 * clutter["cleanliness_score"]
    return {
        "scores": {
            "overall": round(overall, 1),
            "cleanliness": clutter["cleanliness_score"],
            "cta_attention_1s": cta["attention"]["1s"] if cta else None,
            "issues": {s: sum(i["severity"] == s for i in issues) for s in PENALTY},
        },
        "segmentation": seg,
        "modals": modals,
        "elements": elements,
        "clutter": {k: v for k, v in clutter.items() if k != "map"},
        "scanpath": path,
        "scanpath_summary": path_sum,
        "ergonomics": {k: v for k, v in ergo.items() if not k.startswith("_")},
        "issues": issues,
        "timings_ms": timings,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        "_layout": layout,
        "_zones": ergo["_zones"],
        "_maps": maps,
        "_clutter_map": clutter["map"],
        "_rgb": rgb,
        "_seg_meta": seg_meta,
    }


def public_report(report: dict) -> dict:
    """Звіт без масивів — для JSON і для передачі мовній моделі (крок 8)."""
    return {k: v for k, v in report.items() if not k.startswith("_")}
