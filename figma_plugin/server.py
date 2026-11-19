"""Локальний сервер для плагіна Figma: кроки 2–8 пайплайну для фрейму з Figma.

Плагін передає точну структуру фрейму (тип, рамка, кольори, шрифти — з шарів Figma, тож крок 1 не потрібен) і
картинку фрейму (експорт PNG 1×). Сервер повертає оцінки, проблеми з виправленнями (id шарів Figma), план дій,
перевірку «було → стало» і картинки для шарів: дальтонізм, увага 1 / 3 / 7 с, маршрут погляду, шум, зона пальця.
Поради мовної моделі — лише якщо локально запущено Ollama (безкоштовно, без інтернету); інакше — лише правила.

Запуск:  python figma_plugin/server.py   (адреса http://127.0.0.1:8765, лише цей комп'ютер)
Без сторонніх бібліотек для сервера (http.server зі стандартної бібліотеки Python).
"""
from __future__ import annotations

import base64
from pathlib import Path
import io
import json
import os
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
from ux_eval.auditor import (_cta_option_lines, _effect_text, _fix_text, action_plan, build_vlm_prompt,  # noqa: E402
                             clean_vlm_plan, improve_cta, vlm_facts, what_if)
import ux_eval.auditor as _auditor  # noqa: E402

# прогнози «було → стало» — швидким методом візуального шуму (обидва однаково): аналіз у 2–2,5 раза швидший,
# основна оцінка й поради ті самі, приріст відрізняється ≤ 0,5 бала
_auditor.RECHECK_FAST_CLUTTER = True
from ux_eval.color import hex_to_rgb, simulate_cvd  # noqa: E402
from ux_eval.evaluate import evaluate, public_report  # noqa: E402
from ux_eval.layout import Element, Layout  # noqa: E402
from ux_eval.visualize import _boxes, _heat, _scanpath, _zones  # noqa: E402

HOST, PORT = "127.0.0.1", int(os.environ.get("UX_PLUGIN_PORT", "8765"))
OLLAMA = os.environ.get("UX_OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("UX_LLM_MODEL", "qwen2.5vl:3b")
DEV_DIR = os.path.join(ROOT, "figma_plugin")  # для перевірки плагіна в браузері (dev/harness.html)
_ELEMENT_FIELDS = {f.name for f in fields(Element)}
_CACHE: dict[str, dict] = {}  # analysis_id -> звіт (для порад LLM), лише останні кілька
_LOCK = threading.Lock()      # моделі на GPU — по одному аналізу за раз
_READY = {"models": False, "error": None}


def layout_from_plugin(d: dict) -> Layout:
    """Структура від плагіна → Layout (зайві поля відкидаються, рамки обрізаються до фрейму)."""
    W, H = int(round(d["width"])), int(round(d["height"]))
    els = []
    for e in d["elements"]:
        e = {k: v for k, v in e.items() if k in _ELEMENT_FIELDS}
        x, y, w, h = e["box"]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        e["box"] = [x0, y0, x1 - x0, y1 - y0]
        els.append(Element(**e))
    # розмір фрейму Figma — у логічних px: до 480 — телефон, до 1024 — планшет, ширше — десктоп
    device = "mobile" if W <= 480 else "tablet" if W <= 1024 else "desktop"
    return Layout(W, H, d.get("background") or "#FFFFFF", els, device)


def rgb_from_png(b64: str, layout: Layout) -> np.ndarray:
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, tuple(hex_to_rgb(layout.background)) + (255,))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if im.size != (layout.width, layout.height):  # експорт не 1× — приводимо до розміру фрейму
        im = im.resize((layout.width, layout.height), Image.LANCZOS)
    return np.asarray(im)


def colors_from_pixels(layout: Layout, rgb: np.ndarray) -> None:
    """Кольори, яких Figma не дає суцільною заливкою: фото (середній колір — фон для тексту поверх нього) і
    кнопки/картки з градієнтом (медіанний колір рамки). Беруться з картинки фрейму."""
    from ux_eval.color import rgb_to_hex
    for e in layout.elements:
        if e.type in ("image", "face") or (e.fill is None and e.type in ("button", "chip", "input", "card")):
            x, y, w, h = (int(round(v)) for v in e.box)
            patch = rgb[max(0, y):y + h, max(0, x):x + w].reshape(-1, 3)
            if len(patch):
                c = patch.mean(0) if e.type in ("image", "face") else np.median(patch, 0)
                e.fill = rgb_to_hex(c)


MAX_IMAGE_SIDE = 4096


def _jpeg(img) -> str:
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img).astype(np.uint8))
    if max(im.size) > MAX_IMAGE_SIDE:  # Figma не приймає картинки, більші за 4096 px; шар однаково розтягує на розмір фрейму
        k = MAX_IMAGE_SIDE / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def overlays(r: dict) -> list[dict]:
    """Картинки для шарів у Figma (той самий розмір, що й фрейм) — як плитки підсумкової панелі."""
    rgb, maps = r["_rgb"], r["_maps"]
    out = []
    out.append(("1 · Елементи зі структури Figma (тип кожного)", typed_boxes(rgb, r["_layout"].elements, "Структура Figma")))
    cvd = np.clip(simulate_cvd(rgb.astype(np.float64), "deuteranopia"), 0, 255).astype(np.uint8)
    out.append(("2 · Дейтеранопія", Image.fromarray(cvd)))
    t = Image.fromarray(_heat(rgb, maps["1s"]))
    _boxes(t, r, "1s")
    out.append(("3 · Увага 0.5–1.5 с (7: частка уваги, %)", t))
    out.append(("3 · Увага 3–5 с", Image.fromarray(_heat(rgb, maps["3s"]))))
    t = Image.fromarray(_heat(rgb, maps["7s"]))
    _scanpath(t, r["scanpath"])
    out.append(("3–4 · Увага 7+ с і маршрут погляду", t))
    out.append(("5 · Візуальний шум", Image.fromarray(_heat(rgb, r["_clutter_map"], 0.6))))
    if r.get("_zones") is not None:
        out.append(("6 · Зона великого пальця", Image.fromarray(_zones(rgb, r["_zones"]))))
    return [{"name": n, "jpeg_b64": _jpeg(im)} for n, im in out]


TYPE_COLORS = {"button": (230, 30, 60), "heading": (30, 110, 230), "text": (60, 160, 230), "price": (0, 150, 120),
               "icon": (240, 150, 0), "image": (150, 70, 200), "face": (200, 60, 160), "input": (0, 170, 170),
               "chip": (120, 160, 0), "card": (120, 120, 120), "divider": (160, 160, 160)}
TEXT_FAMILY = {"heading", "text", "price"}


def typed_boxes(rgb: np.ndarray, elements: list, title: str) -> Image.Image:
    """Рамки елементів з підписом типу (колір — за типом): так видно, ЩО розпізнано, а не лише де."""
    from ux_eval.visualize import _font
    img = Image.fromarray(rgb.astype(np.uint8).copy())
    d, f = ImageDraw.Draw(img), _font(12, True)
    H, W = rgb.shape[:2]
    for e in elements:
        x, y, w, h = e.box
        if w * h > 0.6 * W * H:
            continue  # фон / велика ілюстрація — рамка на весь кадр нічого не показує
        col = TYPE_COLORS.get(e.type, (255, 255, 255))
        d.rectangle([x, y, x + w, y + h], outline=col, width=3 if e.role == "cta" else 2)
        label = e.type + (" · CTA" if e.role == "cta" else "")
        tw = d.textlength(label, font=f)
        ly = y - 16 if y >= 16 else y
        d.rectangle([x, ly, x + tw + 6, ly + 16], fill=col)
        d.text((x + 3, ly + 1), label, font=f, fill=(255, 255, 255))
    d.rectangle([0, 0, d.textlength(title, font=f) + 10, 18], fill=(0, 0, 0))
    d.text((5, 2), title, font=f, fill=(255, 255, 255))
    return img


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, a[2] * a[3] + b[2] * b[3] - inter)


# Моделі кроку 1 (YOLO, OCR) працюють швидко лише в тому потоці, де вже працювали: у новому потоці HTTP-запиту
# розбір займав 11–13 с замість 1,3 с. Тому крок 1 завжди виконується в одному постійному потоці.
_S1 = ThreadPoolExecutor(1)


def _step1_parse(rgb: np.ndarray):
    """Розпізнавання кроку 1 у потоці _S1: (макет, мс) або None, якщо моделей немає."""
    from ux_eval.evaluate import get_parser
    parser = get_parser()
    if not parser:
        return None
    t = time.perf_counter()
    lay1, _ = parser.parse(rgb)
    return lay1, round((time.perf_counter() - t) * 1000)


def step1_compare(rgb: np.ndarray, figma_layout: Layout, parsed=None) -> dict:
    """Крок 1 пайплайну на картинці фрейму (YOLO11x-UI + OmniParser + PaddleOCR) і порівняння зі структурою Figma,
    яку дизайнер задав сам (еталон). Аналіз кроків 2–8 і далі йде за структурою Figma — вона точна.
    parsed — Future з _S1.submit(_step1_parse, rgb): розпізнавання йде паралельно з кроками 2–8."""
    res = (parsed or _S1.submit(_step1_parse, rgb)).result()
    if not res:
        return {"ok": False, "message": "Моделі кроку 1 не завантажились"}
    lay1, ms = res
    H, W = rgb.shape[:2]
    ref = [e for e in figma_layout.elements if e.box[2] * e.box[3] <= 0.6 * W * H and e.type not in ("divider", "card")]
    found = same = 0
    missed = []
    for e in ref:
        best = max(lay1.elements, key=lambda p: _iou(e.box, p.box), default=None)
        if best is not None and _iou(e.box, best.box) >= 0.5:
            found += 1
            if best.type == e.type or (best.type in TEXT_FAMILY and e.type in TEXT_FAMILY):
                same += 1
        else:
            missed.append(e.text or e.type)
    cta1 = next((p for p in lay1.elements if p.role == "cta"), None)
    ctaF = next((e for e in figma_layout.elements if e.role == "cta"), None)
    return {"ok": True, "ms": ms, "n_step1": len(lay1.elements), "n_figma": len(ref), "found": found, "same_type": same,
            "missed": missed[:8], "cta_same": bool(cta1 and ctaF and _iou(cta1.box, ctaF.box) >= 0.5),
            "cta_step1": (cta1.text or cta1.type) if cta1 else None,
            "_overlay": typed_boxes(rgb, lay1.elements, "Крок 1: розпізнано з картинки (YOLO + OmniParser + OCR)")}


def cta_overlays(r: dict) -> list[dict]:
    """Найкращий варіант кожного виду для кнопки дії — картинкою (пікселі фрейму з перенесеною кнопкою)."""
    res = r.get("cta_options")
    if not res or not res["options"]:
        return []
    from ux_eval.cta_boost import KINDS, apply_option
    cta = r["_layout"].get(res["cta"])
    out = []
    for k in KINDS:
        best = max((o for o in res["options"] if o["kind"] == k and o["gain_1s"] > 0), key=lambda o: o["after"]["1s"], default=None)
        if best:
            img, _ = apply_option(r["_rgb"].astype(np.uint8), r["_layout"], cta, best)
            title = f"8 · Кнопка: {best['kind_ua']} — увага за 1 с {res['before_1s']} → {best['after']['1s']} %"
            out.append({"name": title, "jpeg_b64": _jpeg(typed_boxes(img, [], title))})
    return out


RUNS_DIR = Path(__file__).resolve().parent.parent / "out" / "figma_runs"


def _save_run(aid: str, req: dict, issues: list, pub: dict) -> None:
    """Кожен аналіз з Figma → out/figma_runs/<час>_<id>/: що прочитав плагін (layout.json), PNG фрейму, звіт."""
    d = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{aid}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "layout.json").write_text(json.dumps(req["layout"], ensure_ascii=False, indent=1), encoding="utf-8")
    if req.get("png_b64"):
        (d / "frame.png").write_bytes(base64.b64decode(req["png_b64"]))
    (d / "report.json").write_text(json.dumps({"scores": pub["scores"], "issues": issues}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")


# Дбайливий режим: плагін не застосовує зміни, що стирають стиль макета, — вони лишаються порадою дизайнеру
STYLE_CODES = {"attention_trap", "visual_clutter", "scanpath_detour"}  # приглушити / прибрати декор
MIN_CTA_GAIN = 1.0  # п.п. уваги до кнопки за 1 с (поріг проблеми 8 %); менше — збільшення кнопки лише порада


def _advice_reason(issue: dict, keep_ids: set) -> str | None:
    """Чому виправлення лише порада (кнопки «Застосувати» немає); None — можна застосувати."""
    fx = issue.get("fix")
    if not fx:
        return None
    ids = set(fx.get("remove") or []) | ({fx["id"]} if fx.get("id") else set())
    if ids & keep_ids:
        return "шар захищено (keep / бренд у назві)"
    if issue["code"] in STYLE_CODES or "remove" in fx:
        return "змінює стиль макета (декор, акцентні кольори) — вирішує дизайнер"
    return None


def analyze(req: dict) -> dict:
    layout = layout_from_plugin(req["layout"])
    rgb = rgb_from_png(req["png_b64"], layout) if req.get("png_b64") else None
    if rgb is not None:
        colors_from_pixels(layout, rgb)
    keep_ids = {e["id"] for e in req["layout"]["elements"] if e.get("keep")}
    with _LOCK:
        t = time.perf_counter()
        # крок 1 (розпізнавання з картинки, лише для показу) — паралельно з аналізом за структурою Figma
        s1_fut = _S1.submit(_step1_parse, rgb) if (rgb is not None and req.get("step1", True)) else None
        r = evaluate(layout=layout, rgb=rgb, fast_clutter=bool(req.get("fast_clutter")))
        cta_res = improve_cta(r)  # місце й розмір кнопки дії, що піднімають увагу (ux_eval.cta_boost)
        advice = {id(i): _advice_reason(i, keep_ids) for i in r["issues"]}
        for i in r["issues"]:
            if i["code"] != "cta_low_attention" or not i.get("fix") or advice[id(i)]:
                continue
            opt = i["fix"].get("cta_option")
            if opt is None:
                advice[id(i)] = "пошук не знайшов місця чи розміру, що помітно піднімає увагу до кнопки — вирішує дизайнер"
            elif opt["kind"] in ("insert", "global"):
                advice[id(i)] = (f"{opt['kind_ua']} — це зміна структури макета, вирішує дизайнер "
                                 f"(прогноз уваги за 1 с: {cta_res['before_1s']} → {opt['after']['1s']} %, див. «Шари результатів»)")
            elif opt["gain_1s"] < MIN_CTA_GAIN:
                advice[id(i)] = (f"за прогнозом увага до кнопки майже не зміниться ({cta_res['before_1s']} → "
                                 f"{opt['after']['1s']} %) — тут потрібне рішення дизайнера")
        wi = what_if(r, select=lambda i: not advice[id(i)]) if req.get("check_effect", True) else None
        ms = round((time.perf_counter() - t) * 1000)
        s1 = step1_compare(rgb, layout, s1_fut) if s1_fut is not None else None
    names = {e["id"]: e.get("name") for e in req["layout"]["elements"] if e.get("name")}

    def named(text: str) -> str:  # «1:23» (id шару Figma) → «назва шару»
        for k, v in names.items():
            text = text.replace(f"«{k}»", f"«{v}»").replace(f" {k},", f" {v},").replace(f" {k}.", f" {v}.")
        return text

    plan = named(action_plan(r) + ("\n" + "\n".join(_effect_text(wi)) if wi else ""))
    issues = []
    for i in r["issues"]:
        issues.append({"severity": i["severity"], "code": i["code"], "message": named(i["message"]),
                       "fix": i.get("fix"), "fix_text": named(_fix_text(i.get("fix"), r["_layout"])),
                       "advice_only": advice[id(i)] if i.get("fix") else None,
                       "options": [named(x.strip()) for x in _cta_option_lines(r)] if i["code"] == "cta_low_attention" else []})
    aid = uuid.uuid4().hex[:12]
    _CACHE[aid] = {"report": r, "effect": wi}
    while len(_CACHE) > 5:
        _CACHE.pop(next(iter(_CACHE)))
    pub = public_report(r)
    try:
        _save_run(aid, req, issues, pub)
    except Exception as e:  # збереження — лише для перевірки, аналіз не ламає
        print("Не вдалося зберегти запуск:", e)
    return {"analysis_id": aid, "elapsed_ms": ms, "scores": pub["scores"], "segmentation": pub["segmentation"],
            "timings_ms": pub["timings_ms"], "elements": pub["elements"], "scanpath": pub["scanpath"],
            "scanpath_summary": pub["scanpath_summary"], "ergonomics": pub["ergonomics"],
            "clutter": pub["clutter"], "issues": issues, "plan": plan,
            "effect": None if not wi else {"before": wi["before"], "after": wi["after"],
                                          "note": "оцінка за спрощеним рендером макета після змін"},
            "step1": {k: v for k, v in s1.items() if not k.startswith("_")} if s1 else None,
            "overlays": ([{"name": "0 · Крок 1: розпізнано з картинки фрейму", "jpeg_b64": _jpeg(s1["_overlay"])}]
                         if s1 and s1.get("ok") else []) + overlays(r) + cta_overlays(r)}


LLM_IMAGE_SIDE = 768
LLM_MAX_ISSUES = int(os.environ.get("UX_LLM_MAX_ISSUES", "6"))


def _free_gpu_cache():
    """Повертає драйверу зарезервовану, але порожню пам'ять PyTorch (моделі аналізу лишаються завантаженими)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def ollama_models() -> list[str] | None:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2) as f:
            return [m["name"] for m in json.load(f).get("models", [])]
    except Exception:  # noqa: BLE001
        return None


def advice(req: dict) -> dict:
    """Поради мовної моделі через локальний Ollama; числа — лише зі звіту (clean_vlm_plan)."""
    item = _CACHE.get(req.get("analysis_id", ""))
    if not item:
        return {"ok": False, "message": "Спершу зробіть аналіз (результат на сервері вже не зберігся)."}
    models = ollama_models()
    if models is None:
        return {"ok": False, "message": "Ollama не запущено. Поради мовної моделі необов'язкові: план правил уже є."}
    model = req.get("model") or LLM_MODEL
    if model not in models and f"{model}:latest" not in models:
        return {"ok": False, "message": f"У Ollama немає моделі {model}. Є: {', '.join(models) or 'жодної'}."}
    r, wi = item["report"], item["effect"]
    eff = {"before": wi["before"], "after": wi["after"]} if wi else None
    prompt = build_vlm_prompt(r, eff, max_issues=LLM_MAX_ISSUES)  # коротший запит: 3B на 4 ГБ читає його довго
    buf = io.BytesIO()
    img = Image.fromarray(r["_rgb"].astype(np.uint8))
    img.thumbnail((LLM_IMAGE_SIDE, LLM_IMAGE_SIDE))  # довгі сторінки інакше дають тисячі токенів картинки
    img.save(buf, "PNG")
    _free_gpu_cache()  # більше шарів мовної моделі вміщається у 4 ГБ відеокарти
    # keep_alive=0: модель звільняє відеопам'ять одразу після відповіді, щоб не гальмувати наступний аналіз
    body = json.dumps({"model": model, "prompt": prompt, "images": [base64.b64encode(buf.getvalue()).decode()],
                       "stream": False, "keep_alive": 0,
                       "options": {"temperature": 0, "num_predict": 700}}).encode()
    t = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(OLLAMA + "/api/generate", body,
                                                       {"Content-Type": "application/json"}), timeout=600) as f:
        text = json.load(f).get("response", "")
    c = clean_vlm_plan(text, vlm_facts(r, eff), action_plan(r))
    return {"ok": True, "model": model, "seconds": round(time.perf_counter() - t, 1), "plan": c["plan"],
            "added_critical": c["added_critical"], "flagged_points": c["flagged_points"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload, ctype="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")  # UI плагіна Figma має origin "null"
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # noqa: N802
        self._send(204, b"")

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"ok": True, "models_ready": _READY["models"], "error": _READY["error"],
                                    "ollama_models": ollama_models(), "llm_model": LLM_MODEL})
        if self.path.startswith("/dev/") or self.path.startswith("/plugin/") or self.path.startswith("/examples/"):
            # лише для перевірки плагіна в браузері (імітація Figma) — файли з папки figma_plugin і examples
            rel = self.path.split("?")[0].lstrip("/")
            base = ROOT if rel.startswith("examples/") else DEV_DIR
            p = os.path.normpath(os.path.join(base, rel))
            if p.startswith(base) and os.path.isfile(p):
                ctype = {".html": "text/html", ".js": "text/javascript", ".json": "application/json"}.get(
                    os.path.splitext(p)[1], "application/octet-stream")
                return self._send(200, open(p, "rb").read(), ctype + "; charset=utf-8")
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/analyze":
                return self._send(200, analyze(req))
            if self.path == "/advice":
                return self._send(200, advice(req))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        print(time.strftime("%H:%M:%S"), self.address_string(), fmt % args, flush=True)


def warm_up():
    """Завантаження моделей кроку 3 одразу після старту (перший аналіз інакше чекав би ~25 с)."""
    try:
        demo = Layout.from_json(os.path.join(ROOT, "examples", "product_good.json"))
        with _LOCK:
            evaluate(layout=demo)
            from ux_eval.evaluate import get_parser
            parser = _S1.submit(get_parser).result()  # моделі кроку 1 — для показу «як розпізнає крок 1»
            if parser:  # перший розбір ~8 с (ініціалізація OCR на GPU) — робимо його тут, а не в першому аналізі
                from PIL import Image as _I
                _S1.submit(parser.parse, np.asarray(_I.open(os.path.join(ROOT, "out", "product_good_panel.png")).convert("RGB"))[:900, :1400]
                             if os.path.exists(os.path.join(ROOT, "out", "product_good_panel.png")) else
                             np.full((600, 900, 3), 255, np.uint8)).result()
        _READY["models"] = True
        print("Моделі готові.", flush=True)
    except Exception as e:  # noqa: BLE001
        _READY["error"] = repr(e)
        traceback.print_exc()


def main():
    threading.Thread(target=warm_up, daemon=True).start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"UX-аналіз для Figma: http://{HOST}:{PORT}  (Ctrl+C — зупинити)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
