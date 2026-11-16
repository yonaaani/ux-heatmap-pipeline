"""Демонстрація пайплайну по кроках 1–8 і метрик валідації.

Запуск:  python demo.py [скріншот_сайту.png]
Результати — у папці out/.
"""
import json
import os
import sys

import cv2
import numpy as np

from ux_eval.auditor import _effect_text, action_plan, build_vlm_prompt, improve_cta, what_if
from ux_eval.evaluate import evaluate, public_report
from ux_eval.layout import Layout, render
from ux_eval.metrics import evaluate_saliency
from ux_eval.visualize import report_panel

sys.stdout.reconfigure(encoding="utf-8")
OUT = "out"
os.makedirs(OUT, exist_ok=True)


def header(text):
    print("\n" + "─" * 78 + f"\n{text}\n" + "─" * 78)


def show_steps(name: str, r: dict):
    els = {e["id"]: e for e in r["elements"]}

    header(f"[{name}] КРОК 1. Структурна UI-сегментація")
    seg = r["segmentation"]
    print(f"Джерело структури: {seg['source']} · елементів: {seg['n_elements']} · "
          f"частка «повітря» (Whitespace Ratio): {seg['whitespace_ratio']:.0%}")

    header(f"[{name}] КРОК 2. Колориметрія, контраст (APCA), дальтонізм")
    print(f"  {'елемент':<14} {'ΔE00 до фону':>12} {'ΔEok':>6} {'|Lc|':>6} {'мін.':>5}  {'найгірше |Lc| при CVD':<28}")
    for e in r["elements"]:
        lc = f"{abs(e['apca_lc']):6.1f}" if "apca_lc" in e else "     -"
        mn = f"{e['apca_min']:5.0f}" if "apca_min" in e else "    -"
        cvd = ""
        if "apca_lc_cvd" in e:
            k = min(e["apca_lc_cvd"], key=lambda k: abs(e["apca_lc_cvd"][k]))
            cvd = f"{abs(e['apca_lc_cvd'][k]):.0f} ({k})"
        flag = "  ПОГАНО" if "apca_lc" in e and abs(e["apca_lc"]) < e["apca_min"] else ""
        print(f"  {e['id']:<14} {e.get('delta_e_bg', '-'):>12} {e.get('delta_e_ok', '-'):>6} {lc} {mn}  {cvd:<28}{flag}")
    for e in r["elements"]:
        if e.get("role") == "cta":
            print(f"  CTA {e['id']}: ΔE00 за дальтонізму {e['delta_e_cvd']}; "
                  f"кнопки/чипи, яскравіші за кольором: {e.get('color_rank_stronger') or 'немає'}")

    header(f"[{name}] КРОК 3. Теплові карти уваги в часі (частка уваги, %)")
    print(f"  {'елемент':<12} {'0.5–1.5 с':>10} {'3–5 с':>8} {'7+ с':>8}")
    for e in r["elements"]:
        a = e["attention"]
        print(f"  {e['id']:<12} {a['1s']:>10.1f} {a['3s']:>8.1f} {a['7s']:>8.1f}")

    header(f"[{name}] КРОК 4. Траєкторія погляду (Scanpath)")
    for p in r["scanpath"]:
        print(f"  {p['k']}. ({p['x']:>3}, {p['y']:>3})  {p['t_ms']:>3} мс  → {p['element']}")
    s = r["scanpath_summary"]
    print(f"  Кнопка дії: фіксація №{s['cta_fixation_index']} "
          f"(≈{s['cta_time_to_first_fixation_ms']} мс); "
          f"«пастки уваги» до неї: {s['attention_traps_before_cta'] or 'немає'}")

    header(f"[{name}] КРОК 5. Візуальний шум (Feature Congestion + Subband Entropy)")
    c = r["clutter"]
    print(f"  Feature Congestion = {c['feature_congestion']} (колір {c['components']['color']}, "
          f"контраст {c['components']['contrast']}, орієнтації {c['components']['orientation']})")
    print(f"  Subband Entropy = {c['subband_entropy']} · бал чистоти = {c['cleanliness_score']}/100")

    header(f"[{name}] КРОК 6. Ергономіка: закони Фіттса, Хіка, Thumb Zone")
    g = r["ergonomics"]
    print(f"  Варіантів вибору n = {g['n_choices']} → T_choice (Хік) = {g['hick_choice_time_s']} с")
    for t in g["targets"]:
        print(f"  {t['id']:<12} D={t['distance_px']:>3}px W={t['width_px']:>3}px → "
              f"T_click (Фіттс) = {t['fitts_time_s']} с · зона пальця: {t.get('thumb_zone', '-')} · "
              f"розмір цілі {('OK' if t['tap_target_ok'] else 'замалий') if t.get('tap_target_checked', True) else 'не перевіряється (миша)'}")

    header(f"[{name}] КРОК 7. Конверсійна помітність (CTA Attention Share)")
    cta = next(e for e in r["elements"] if e["role"] == "cta")
    risk = cta["attention"]["1s"] < 8 or cta["delta_e_bg"] < 15
    print(f"  AoI_CTA = {cta['attention']['1s']}% (поріг 8%) · ΔE00 = {cta['delta_e_bg']} (поріг 15)")
    print("  → КРИТИЧНИЙ РИЗИК втрати конверсії" if risk else "  → кнопка помітна, ризику немає")

    header(f"[{name}] КРОК 8. AI-UX аудитор (Rule-Based Fallback) — план дій і перевірка ефекту")
    improve_cta(r)  # місце й розмір кнопки дії, що піднімають увагу до неї
    wi = what_if(r)
    plan = action_plan(r) + "\n" + "\n".join(_effect_text(wi))
    print(plan)

    # файли
    report_panel(r, f"{name}", after=wi).save(f"{OUT}/{name}_panel.png")
    with open(f"{OUT}/{name}_report.json", "w", encoding="utf-8") as f:
        json.dump(public_report(r), f, ensure_ascii=False, indent=2, default=str)
    with open(f"{OUT}/{name}_plan.txt", "w", encoding="utf-8") as f:
        f.write(plan)
    with open(f"{OUT}/{name}_vlm_prompt.txt", "w", encoding="utf-8") as f:
        f.write(build_vlm_prompt(r, {"before": wi["before"], "after": wi["after"]}))


def main():
    bad = Layout.from_json("examples/product_bad.json")
    good = Layout.from_json("examples/product_good.json")

    evaluate(good)  # «прогрів» (перше виконання включає завантаження бібліотек)
    r_bad = evaluate(bad)
    r_good = evaluate(good)
    show_steps("product_bad", r_bad)
    show_steps("product_good", r_good)

    header("КРОК 1 на скріншоті (без JSON: YOLO11x-UI + OmniParser + PP-OCRv5): що знайшов алгоритм")
    shot = render(bad)
    cv2.imwrite(f"{OUT}/product_bad_screenshot.png", shot[..., ::-1])
    r_shot = evaluate(rgb=shot)
    for e in r_shot["_layout"].elements:
        print(f"  {e.id:<12} {e.role:<9} box={e.box}")
    print(f"  Частка «повітря»: {r_shot['segmentation']['whitespace_ratio']:.0%}")
    report_panel(r_shot, "Скріншот → автоматична сегментація", after=what_if(r_shot)).save(
        f"{OUT}/screenshot_panel.png")

    header("УСІ КРОКИ на скріншоті сайту (шлях до PNG — першим аргументом)")
    gh = sys.argv[1] if len(sys.argv) > 1 else ""
    if os.path.exists(gh):
        from PIL import Image
        r_gh = evaluate(rgb=np.asarray(Image.open(gh).convert("RGB")))
        improve_cta(r_gh)
        print(action_plan(r_gh, check_effect=True))
        report_panel(r_gh, "Скріншот сайту", after=what_if(r_gh)).save(f"{OUT}/site_panel.png")

    header("РОЗДІЛ 5. Метрики валідації (KL, CC, NSS, AUC-Judd) — перевірка на штучних даних")
    print("  Справжня валідація потребує айтрекінгу (датасет UEyes). Тут — перевірка, що")
    print("  метрики рахуються правильно: ідеальний прогноз має пройти, випадковий — ні.")
    h, w = 180, 100
    yy, xx = np.mgrid[0:h, 0:w]
    gt = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 7 ** 2)) for cx, cy in [(25, 30), (70, 90), (50, 150)])
    rng = np.random.default_rng(0)
    fix = np.zeros((h, w), bool)
    fix.ravel()[rng.choice(gt.size, 40, replace=False, p=(gt / gt.sum()).ravel())] = True
    print("  ідеальний прогноз: ", evaluate_saliency(gt, gt, fix))
    print("  випадковий прогноз:", evaluate_saliency(rng.random((h, w)), gt, fix))

    header("ШВИДКОДІЯ ")
    for name, r in (("product_bad", r_bad), ("product_good", r_good)):
        print(f"  {name}: {r['elapsed_ms']} мс · за кроками: {r['timings_ms']}")
    r_fast = evaluate(bad, fast_clutter=True)
    print(f"  product_bad, швидкий режим кроку 5 (320 px, float32): {r_fast['elapsed_ms']} мс · "
          f"за кроками: {r_fast['timings_ms']} · чистота {r_fast['scores']['cleanliness']} "
          f"(точний режим {r_bad['scores']['cleanliness']})")

    print(f"\nУсі результати збережено в папці {OUT}/")


if __name__ == "__main__":
    main()
