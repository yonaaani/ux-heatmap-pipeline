"""Крок 8. Мультимодальний AI-UX аудитор.

Збирає всі числові матриці з кроків 1–7 і переводить їх у
зрозумілий дизайнеру план дій (на який HEX змінити колір, на скільки пікселів
збільшити шрифт чи відступ тощо).

- improve_cta()      — пошук місця й розміру кнопки дії, що піднімає увагу до неї (ux_eval.cta_boost);
- action_plan()      — Rule-Based Fallback (працює без GPU, реалізовано);
- what_if()          — застосування всіх змін і повторний аналіз кроків 2–7 («було → стало»);
- build_vlm_prompt() — запит до Qwen2.5-VL-7B (картинка макета + звіт); модель
                       запускається окремо на GPU.
"""
from __future__ import annotations

import json
import re

from .evaluate import public_report

_SEV_ORDER = {"critical": 0, "major": 1, "minor": 2}
_SEV_UA = {"critical": "КРИТИЧНО", "major": "ВАЖЛИВО", "minor": "ПОРАДА"}
_FIELD_UA = {"color": "колір тексту", "fill": "колір заливки", "box": "розмір/позицію [x, y, w, h]",
             "font_size": "розмір шрифту", "bold": "жирність", "palette": "палітру"}


def _fix_text(fix: dict | None, layout) -> str:
    if not fix:
        return ""
    if "remove" in fix:
        return "Прибрати декоративні елементи: " + ", ".join(fix["remove"]) + "."
    el = layout.get(fix["id"])
    parts = []
    for field, val in fix["set"].items():
        old = getattr(el, field, None)
        if old == val:
            continue
        if field == "box":
            parts.append(f"{_FIELD_UA[field]} {[round(v) for v in old]} → {[round(v) for v in val]}")
        elif field == "palette":
            parts.append(f"палітру → {', '.join(val)}")
        else:
            parts.append(f"{_FIELD_UA.get(field, field)} {old} → {val}")
    ins = (fix.get("cta_option") or {}).get("insert")
    if ins:
        parts.append(f"звільнити рядок {ins['dy']} px під заголовком на висоті y = {ins['y']} "
                     f"(вміст нижче зсунути вниз на {ins['dy']} px)")
    return ("Змінити " + "; ".join(parts) + ".") if parts else ""


def improve_cta(report: dict, model=None, tries: int = 3) -> dict | None:
    """Крок 8: якщо кнопка дії отримує мало уваги — шукає місце й розмір, що піднімають увагу (ux_eval.cta_boost),
    і замінює виправлення «трохи більша кнопка» найкращим перевіреним варіантом. Решта варіантів — у
    report["cta_options"] (план показує їх з прогнозом).

    Варіант іде в план, лише якщо повний повторний аналіз (кроки 2–7) після цієї однієї зміни не гіршає: загальна
    оцінка не падає більш ніж на 0,5, важливих і критичних проблем не більше, увага до кнопки за 1 с росте. Інакше
    (наприклад, кнопка вище забирає увагу, але маршрут погляду до неї більше не доходить) пробуємо наступний."""
    issue = next((i for i in report["issues"] if i["code"] == "cta_low_attention"), None)
    if issue is None:
        return None
    from .cta_boost import cta_options
    from .evaluate import evaluate
    res = cta_options(report, model)
    report["cta_options"] = res
    if not res:
        return None
    old_fix = issue.get("fix")
    colors = {k: v for k, v in ((old_fix or {}).get("set") or {}).items() if k in ("fill", "color")}
    screenshot = str(report["segmentation"]["source"]).startswith("screenshot")
    meta = report.get("_seg_meta") if screenshot else None
    base = None
    res["best"] = None
    for opt in res["ranked"][:tries]:
        issue["fix"] = {"id": res["cta"], "set": {**colors, "box": opt["box"], "font_size": opt["font_size"]},
                        "cta_option": opt}
        if base is None:
            base = _recheck_base(report)
        lay2, rgb2, _ = apply_fixes(report, select=lambda i: i is issue)
        after = evaluate(layout=lay2, rgb=rgb2, seg_meta=meta, fast_clutter=_recheck_fast())
        sev = lambda r: r["scores"]["issues"]["critical"] + r["scores"]["issues"]["major"]  # noqa: E731
        opt["check"] = {"overall": [base["scores"]["overall"], after["scores"]["overall"]],
                        "critical_major": [sev(base), sev(after)],
                        "cta_1s": [base["scores"]["cta_attention_1s"], after["scores"]["cta_attention_1s"]]}
        if after["scores"]["overall"] >= base["scores"]["overall"] - 0.5 and sev(after) <= sev(base) and                 (after["scores"]["cta_attention_1s"] or 0) > (base["scores"]["cta_attention_1s"] or 0):
            res["best"] = opt
            break
    if res["best"] is None:
        issue["fix"] = old_fix
    return res


def _cta_option_lines(report: dict) -> list[str]:
    res = report.get("cta_options")
    if not res or not res["options"]:
        return []
    lines = ["   Варіанти для кнопки (прогноз уваги за 1 с моделлю кроку 3; ✓ — приріст підтверджує й резервна модель):"]
    from .cta_boost import KINDS
    for k in KINDS:
        best = max((o for o in res["options"] if o["kind"] == k), key=lambda o: o["after"]["1s"], default=None)
        if best:
            mark = "✓" if best["confirmed"] and best["gain_1s"] > 0 else "✗"
            star = " ← у плані" if res["best"] is not None and best is res["best"] else ""
            lines.append(f"   • {best['kind_ua']} (кнопка ×{best['scale']}): {res['before_1s']} → "
                         f"{best['after']['1s']} % {mark}{star}")
    return lines


def _recolor(rgb, box, old_hex, new_hex, tol=12.0, protect=(), max_frac=1.0):
    """Скриншот: пікселі в рамці, близькі до старого кольору (ΔE00 < tol), фарбуються новим (зі збереженням
    відтінків згладжування — зсув на різницю кольорів). protect — рамки вкладених елементів, які не чіпаємо
    (інакше, напр., зміна кольору тексту великого блоку перефарбовувала фон кнопки всередині нього).
    max_frac: якщо під старий колір потрапляє більша частка рамки, це не текст, а фон (колір тексту визначено
    невдало, |Lc| ≈ 0) — пікселі не змінюються, щоб не перефарбувати фон разом з усім, що на ньому. Повертає True,
    якщо перефарбовано."""
    import numpy as np

    from .color import delta_e2000, hex_to_rgb, rgb_to_lab
    x, y, w, h = (int(round(v)) for v in box)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return False
    patch = rgb[y0:y1, x0:x1].astype(np.float64)
    old, new = np.array(hex_to_rgb(old_hex), float), np.array(hex_to_rgb(new_hex), float)
    de = delta_e2000(rgb_to_lab(patch), rgb_to_lab(old[None, None, :]))
    m = de < tol
    if m.mean() > max_frac:
        return False
    for px, py, pw, ph in protect:
        a0, b0 = int(max(px, x0)) - x0, int(max(py, y0)) - y0
        a1, b1 = int(min(px + pw, x1)) - x0, int(min(py + ph, y1)) - y0
        if a1 > a0 and b1 > b0:
            m[b0:b1, a0:a1] = False
    patch[m] = np.clip(patch[m] + (new - old), 0, 255)
    rgb[y0:y1, x0:x1] = patch.astype(rgb.dtype)
    return True


def _nested_boxes(layout, el, frac: float = 0.8) -> list:
    """Рамки менших елементів, що щонайменше на frac лежать усередині рамки el."""
    x, y, w, h = el.box
    out = []
    for o in layout.elements:
        if o is el or o.id == el.id:
            continue
        ox, oy, ow, oh = o.box
        if ow * oh >= w * h or ow <= 0 or oh <= 0:
            continue
        iw = max(0, min(x + w, ox + ow) - max(x, ox))
        ih = max(0, min(y + h, oy + oh) - max(y, oy))
        if iw * ih >= frac * ow * oh:
            out.append(o.box)
    return out


def apply_fixes(report: dict, select=None):
    """Застосовує всі запропоновані виправлення: новий макет (і для скриншота — підфарбоване зображення).

    Макет JSON — змінюються поля елементів (колір, заливка, розмір, шрифт, палітра), прибираються елементи.
    Скриншот — змінюються лише кольори (заливка й текст перефарбовуються в пікселях), бо перемалювати справжній
    скриншот з іншими розмірами неможливо; зміни розмірів для скриншота лишаються лише порадою."""
    layout = report["_layout"].copy()
    screenshot = str(report["segmentation"]["source"]).startswith("screenshot")
    rgb = report["_rgb"].copy() if screenshot else None
    applied = []
    for i in report["issues"]:
        fx = i.get("fix")
        if not fx or (select is not None and not select(i)):  # select — лише вибрані виправлення (плагін Figma)
            continue
        if "cta_option" in fx:  # місце / розмір кнопки (і новий рядок) — так само, як їх перевіряв пошук
            from .cta_boost import apply_option
            el = layout.get(fx["id"])
            if screenshot:
                rgb, layout = apply_option(rgb, layout, el, fx["cta_option"])
            else:
                _, layout = apply_option(report["_rgb"].astype("uint8"), layout, el, fx["cta_option"])
            applied.append((i["code"], f"{el.id}.{fx['cta_option']['kind']}"))
            fx = {"id": fx["id"], "set": {k: v for k, v in fx["set"].items() if k not in ("box", "font_size")}}
        if "remove" in fx:
            if not screenshot:
                layout.elements = [e for e in layout.elements if e.id not in fx["remove"]]
                applied.append((i["code"], "remove"))
            continue
        el = next((e for e in layout.elements if e.id == fx["id"]), None)
        if el is None:
            continue
        for field, val in fx["set"].items():
            if screenshot and field in ("fill", "color") and getattr(el, field):
                if not _recolor(rgb, el.box, getattr(el, field), val, protect=_nested_boxes(layout, el),
                                max_frac=1.0 if field == "fill" else 0.5):
                    continue  # колір тексту збігається з фоном — на скриншоті це не застосувати

            elif screenshot and field not in ("fill", "color"):
                continue
            setattr(el, field, val)
            applied.append((i["code"], f"{el.id}.{field}"))
    return layout, rgb, applied


def _metrics(rep: dict) -> dict:
    cta = next((e for e in rep["elements"] if e["role"] == "cta"), None)
    return {"overall": rep["scores"]["overall"], "cleanliness": rep["scores"]["cleanliness"],
            "cta_attention_1s": cta["attention"]["1s"] if cta else None,
            "cta_attention_7s": cta["attention"]["7s"] if cta else None,
            "cta_delta_e00": cta.get("delta_e_bg") if cta else None,
            "cta_fixation": rep["scanpath_summary"]["cta_fixation_index"],
            "issues": dict(rep["scores"]["issues"])}


# Повторні перевірки «було → стало» (improve_cta, what_if) рахують візуальний шум кроку 5 швидким наближеним методом,
# якщо RECHECK_FAST_CLUTTER = True (так робить сервер плагіна Figma: точний метод тричі за аналіз — до ~30 с).
# «Було» й «стало» тоді рахуються одним методом, тож різниця чесна; основна оцінка звіту лишається точною.
RECHECK_FAST_CLUTTER = False


def _recheck_fast():
    return True if RECHECK_FAST_CLUTTER else None


def _recheck_base(report: dict) -> dict:
    """«Було» для повторних перевірок (improve_cta, what_if): той самий макет, рахується один раз на звіт.
    Макет звіту виправлення не змінюють (apply_fixes працює з копією), тож результат однаковий для обох."""
    if "_recheck_base" not in report:
        from .evaluate import evaluate
        screenshot = str(report["segmentation"]["source"]).startswith("screenshot")
        meta = report.get("_seg_meta") if screenshot else None
        report["_recheck_base"] = evaluate(layout=report["_layout"].copy(),
                                           rgb=report["_rgb"].copy() if screenshot else None, seg_meta=meta,
                                           fast_clutter=_recheck_fast())
    return report["_recheck_base"]


def what_if(report: dict, select=None) -> dict:
    """Перевірка ефекту плану: застосувати виправлення і заново прогнати кроки 2–7 тим самим шляхом.

    «Було» теж перераховується з тим самим (уже знайденим) макетом, щоб порівняння було чесним: різниця —
    лише від виправлень."""
    from .evaluate import evaluate
    screenshot = str(report["segmentation"]["source"]).startswith("screenshot")
    meta = report.get("_seg_meta") if screenshot else None
    base = _recheck_base(report)
    layout2, rgb2, applied = apply_fixes(report, select)
    after = evaluate(layout=layout2, rgb=rgb2, seg_meta=meta, fast_clutter=_recheck_fast())
    return {"applied": applied, "before": _metrics(base), "after": _metrics(after), "_after_report": after}


def _effect_text(wi: dict) -> list[str]:
    b, a = wi["before"], wi["after"]

    def arrow(k, unit="", nd=1):
        if b[k] is None or a[k] is None:
            return None
        return f"{b[k]:.{nd}f}{unit} → {a[k]:.{nd}f}{unit}"

    out = ["Очікуваний ефект, якщо застосувати всі зміни (повторний аналіз зміненого макета):"]
    for k, name, unit in (("cta_attention_1s", "увага до кнопки дії за 1 с", "%"),
                          ("cta_attention_7s", "увага до кнопки дії за 7 с", "%"),
                          ("cta_delta_e00", "контраст кнопки з фоном ΔE00", ""),
                          ("cleanliness", "чистота макета", "/100"), ("overall", "загальна оцінка", "/100")):
        t = arrow(k, unit)
        if t:
            out.append(f"   • {name}: {t}")
    if b["cta_fixation"] != a["cta_fixation"]:
        out.append(f"   • погляд доходить до кнопки: на {b['cta_fixation'] or '—'}-й → на {a['cta_fixation'] or '—'}-й фіксації")
    out.append(f"   • проблем (критичних / важливих / порад): {b['issues']['critical']}/{b['issues']['major']}/"
               f"{b['issues']['minor']} → {a['issues']['critical']}/{a['issues']['major']}/{a['issues']['minor']}")
    return out


def action_plan(report: dict, check_effect: bool = False) -> str:
    """План дій для дизайнера (Rule-Based Fallback кроку 8). check_effect — додати «було → стало»."""
    layout = report["_layout"]
    s = report["scores"]
    lines = [f"Загальна оцінка: {s['overall']}/100 · чистота: {s['cleanliness']}/100 · "
             f"увага до CTA (1 с): {s['cta_attention_1s']}%"]
    sp = report["scanpath_summary"]
    if sp["cta_fixation_index"]:
        lines.append(f"Погляд доходить до кнопки дії на {sp['cta_fixation_index']}-й фіксації "
                     f"(≈{sp['cta_time_to_first_fixation_ms']} мс).")
    issues = sorted(report["issues"], key=lambda i: _SEV_ORDER[i["severity"]])
    if not issues:
        lines.append("Проблем не знайдено.")
    for n, i in enumerate(issues, 1):
        lines.append(f"{n}. [{_SEV_UA[i['severity']]}] {i['message']}")
        fx = _fix_text(i["fix"], layout)
        if fx:
            lines.append(f"   → {fx}")
        if i["code"] == "cta_low_attention":
            lines.extend(_cta_option_lines(report))
    if check_effect and any(i.get("fix") for i in issues):
        lines.extend(_effect_text(what_if(report)))
    return "\n".join(lines)


def vlm_facts(report: dict, effect: dict | None = None, max_issues: int = 12) -> dict:
    """Стислий звіт кроків 1–7 для мовної моделі: оцінки, кнопка дії, траєкторія, ергономіка, проблеми з
    конкретними виправленнями (HEX, пікселі) і — якщо є — перевірений ефект «було → стало» (what_if)."""
    layout = report["_layout"]
    cta = next((e for e in report["elements"] if e["role"] == "cta"), None)
    g = report["ergonomics"]
    issues = sorted(report["issues"], key=lambda i: _SEV_ORDER[i["severity"]])[:max_issues]
    facts = {
        "scores": report["scores"],
        "cta": None if not cta else {"text": layout.get(cta["id"]).text, "attention_pct": cta["attention"],
                                    "delta_e00_to_background": cta.get("delta_e_bg")},
        "scanpath": {k: report["scanpath_summary"][k] for k in ("cta_fixation_index", "cta_time_to_first_fixation_ms",
                                                               "attention_traps_before_cta")},
        "ergonomics": {"input": g.get("input"), "largest_choice_group": g.get("n_choices"),
                       "hick_s": g.get("hick_choice_time_s")},
        "issues": [{"severity": i["severity"], "message": i["message"],
                    **({"fix": _fix_text(i["fix"], layout)} if i.get("fix") and _fix_text(i["fix"], layout) else {})}
                   for i in issues],
    }
    if effect:
        facts["effect_if_all_fixes_applied"] = {"before": effect["before"], "after": effect["after"]}
    return facts


HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")
NUM_RE = re.compile(r"(?<![\w#])(\d+(?:[.,]\d+)?)\s*(%|px|пікс|мс|ms|с\b|s\b|/100|бал)", re.I)
_ANY_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_POINT_RE = re.compile(r"(?m)^\s*(?:#+\s*)?\d+\.\s")
_EFFECT_RE = re.compile(r"(?mi)^[\s#*]*(очікуван|ожидан|ефект|результат)")
_RU_FIX = [(re.compile(r"(?i)ожиданн?(ий|ые|ый)"), "Очікуваний"), (re.compile(r"(?i)ожидаем\w*"), "очікуваний")]
MAX_POINTS = 6


def fact_values(facts: dict, rules_plan: str = "") -> tuple[set, set]:
    """HEX-коди й числа (округлені до 0,1), що є у звіті, — з ними звіряється текст моделі."""
    src = json.dumps(facts, ensure_ascii=False) + "\n" + rules_plan
    # abs: «-50%» у назві елемента — це те саме число, що й «50%» у тексті моделі
    return {h.upper() for h in HEX_RE.findall(src)}, {abs(round(float(x), 1)) for x in _ANY_NUM.findall(src)}


def unsupported(text: str, src_hex: set, src_num: set) -> list[str]:
    """HEX-коди й числа з одиницею в тексті, яких немає у звіті."""
    bad = [h.upper() for h in HEX_RE.findall(text) if h.upper() not in src_hex]
    bad += [f"{v}{u}" for v, u in NUM_RE.findall(text) if round(float(v.replace(",", ".")), 1) not in src_num]
    return bad


def _critical_groups(facts: dict) -> list[dict]:
    """Критичні проблеми з виправленням; проблеми з однаковим виправленням (напр. контраст, ΔE і увага до однієї
    кнопки) об'єднані в одну — з найповнішим текстом виправлення."""
    groups: list[list] = []  # [найповніше виправлення, [повідомлення]]
    for i in (i for i in facts["issues"] if i["severity"] == "critical" and i.get("fix")):
        f = i["fix"].rstrip(". ")
        g = next((g for g in groups if f in g[0] or g[0].rstrip(". ") in i["fix"]), None)
        if g is None:
            groups.append([i["fix"], [i["message"]]])
        else:
            g[0] = max(g[0], i["fix"], key=len)
            g[1].append(i["message"])
    return [{"message": " ".join(m), "fix": fix} for fix, m in groups]


def _fix_keys(fix: str) -> set[str]:
    """HEX-коди й числа (2–4 цифри) з виправлення — за ними видно, чи план його містить."""
    return {h.upper() for h in HEX_RE.findall(fix)} | set(re.findall(r"\b\d{2,4}\b", fix))


def clean_vlm_plan(plan: str, facts: dict, rules_plan: str = "", add_missing_critical: bool = True) -> dict:
    """Обробка плану Qwen після генерації (модель іноді вигадує ефект і порушує формат).

    1) Рядок про очікуваний ефект від моделі відкидається і замінюється на «було → стало» з повторного аналізу
       (facts["effect_if_all_fixes_applied"], what_if) — числам моделі про ефект довіряти не можна.
    2) Формат: русизми виправляються; однакові пункти лишаються один раз; пунктів не більше 6; пункт, де є HEX або
       число, якого немає у звіті, позначається «⚠ не підтверджено звітом: …» (не видаляється — рішення за людиною).
    3) add_missing_critical: критична проблема, якої немає в плані моделі, додається на початок пунктом з правил
       (Qwen-7B у 4 бітах іноді ігнорує список обов'язкових проблем із запиту v2)."""
    for rx, rep in _RU_FIX:
        plan = rx.sub(rep, plan)
    starts = [m.start() for m in _POINT_RE.finditer(plan)]
    head = plan[:starts[0]] if starts else plan
    chunks = [plan[a:b] for a, b in zip(starts, starts[1:] + [len(plan)])]
    # ефект моделі: усе від заголовка «Очікуваний ефект / Результат» після останнього пункту
    dropped_effect = ""
    if chunks:
        m = _EFFECT_RE.search(chunks[-1], 1)
        if m:
            dropped_effect, chunks[-1] = chunks[-1][m.start():], chunks[-1][:m.start()]
    else:
        m = _EFFECT_RE.search(head)
        if m:
            dropped_effect, head = head[m.start():], head[:m.start()]
    src_hex, src_num = fact_values(facts, rules_plan)
    seen, points, removed_dup, flagged = set(), [], 0, 0
    for c in chunks:
        body = _POINT_RE.sub("", c, count=1).strip()
        key = re.sub(r"[\W_]+", " ", body.lower()).strip()
        if key in seen:
            removed_dup += 1
            continue
        seen.add(key)
        bad = unsupported(body, src_hex, src_num)
        if bad:
            body += f"\n   ⚠ не підтверджено звітом: {', '.join(sorted(set(bad)))}"
            flagged += 1
        points.append(body)
    # критичні проблеми, яких модель не згадала (жодного HEX / розміру з виправлення), — пункт з правил на початок
    added, n_points = [], len(points)
    if add_missing_critical:
        groups = _critical_groups(facts)
        kept = points[:MAX_POINTS]
        for _ in range(MAX_POINTS + 1):  # перевірка після обрізання: доданий пункт може витіснити пункт моделі
            text = "\n".join(kept).upper()
            missing = [g for g in groups if not any(k in text for k in _fix_keys(g["fix"]))]
            added = [f"(з правил, модель пропустила) {g['message']} → {g['fix']}" for g in missing]
            if len(added) > MAX_POINTS:  # критичних більше, ніж пунктів: решту — одним пунктом
                added = added[:MAX_POINTS - 1] + ["(з правил, модель пропустила) Також критичне: "
                                                  + "; ".join(g["fix"].rstrip(". ") for g in missing[MAX_POINTS - 1:])]
            new_kept = points[:max(0, MAX_POINTS - len(added))]
            if new_kept == kept:
                break
            kept = new_kept
        points = added + kept
    cut = max(0, n_points + len(added) - MAX_POINTS)
    points = points[:MAX_POINTS]
    out = head.rstrip() + ("\n\n" if head.strip() else "") + "\n\n".join(f"{n}. {p}" for n, p in enumerate(points, 1))
    eff = facts.get("effect_if_all_fixes_applied")
    if eff:
        lines = _effect_text(eff)
        lines[0] = "### Очікуваний ефект, якщо застосувати всі зміни (повторний аналіз зміненого макета, не модель):"
        out += "\n\n" + "\n".join(lines)
    return {"plan": out.strip(), "removed_duplicates": removed_dup, "removed_over_limit": cut,
            "flagged_points": flagged, "added_critical": len(added),
            "dropped_model_effect": dropped_effect.strip()}


def build_vlm_prompt(report: dict, effect: dict | None = None, version: int = 2, max_issues: int = 12) -> str:
    """Текстова частина запиту до Qwen2.5-VL-7B (зображення макета передається окремо).

    version=1 — базовий варіант. version=2 — явний список критичних проблем, кожна з яких має стати
    окремим пунктом з HEX/px з виправлення, і без рядка про ефект (його додає clean_vlm_plan з повторного аналізу).
    """
    if version == 2:
        return _vlm_prompt_v2(report, effect, max_issues)
    return (
        "Ти — UX-аудитор. Тобі надано зображення інтерфейсу та результати автоматичного аналізу (кроки 1–7): "
        "теплові карти уваги, траєкторію погляду, колірну доступність (APCA, ΔE2000, дальтонізм), візуальний шум "
        "і ергономіку (закони Фіттса й Хіка, зона великого пальця).\n"
        "Склади для дизайнера план дій українською мовою: не більше 6 пунктів, від найважливішого. У кожному пункті — "
        "що саме змінити (HEX-коди, пікселі, розміри шрифтів) і навіщо. Наприкінці — один рядок про очікуваний "
        "ефект для частки уваги до головної кнопки та загальної оцінки.\n"
        "ВАЖЛИВО: використовуй лише числа, HEX-коди й назви елементів, що є в результатах аналізу; не вигадуй нових. "
        "Зображення використовуй, щоб пояснити, де саме на екрані елемент.\n\n"
        f"РЕЗУЛЬТАТИ АНАЛІЗУ:\n{json.dumps(vlm_facts(report, effect), ensure_ascii=False)}"
    )


def _vlm_prompt_v2(report: dict, effect: dict | None = None, max_issues: int = 12) -> str:
    facts = vlm_facts(report, effect, max_issues)
    crit = _critical_groups(facts)[:MAX_POINTS]
    must = ""
    if crit:
        must = ("ОБОВ'ЯЗКОВО зроби окремим пунктом кожну критичну проблему нижче, у цьому порядку, і перенеси в пункт "
                "HEX-коди та пікселі з її виправлення без змін:\n"
                + "\n".join(f"- {i['message']} → {i['fix']}" for i in crit) + "\n"
                "Решту пунктів (до 6 разом) візьми з інших проблем звіту.\n")
    return (
        "Ти — UX-аудитор. Тобі надано зображення інтерфейсу та результати автоматичного аналізу (кроки 1–7): "
        "теплові карти уваги, траєкторію погляду, колірну доступність (APCA, ΔE2000, дальтонізм), візуальний шум "
        "і ергономіку (закони Фіттса й Хіка, зона великого пальця).\n"
        "Склади для дизайнера план дій: не більше 6 пронумерованих пунктів (1. … 6.), від найважливішого, без "
        "повторів. У кожному пункті — що саме змінити (HEX-коди, пікселі, розміри шрифтів), де це на екрані "
        "(за зображенням) і навіщо.\n" + must +
        "Пиши лише українською мовою, без російських слів. Не пиши підсумку чи рядка про очікуваний ефект — "
        "його система додасть сама з повторного аналізу.\n"
        "ВАЖЛИВО: використовуй лише числа, HEX-коди й назви елементів, що є в результатах аналізу; не вигадуй нових.\n\n"
        f"РЕЗУЛЬТАТИ АНАЛІЗУ:\n{json.dumps(facts, ensure_ascii=False)}"
    )
