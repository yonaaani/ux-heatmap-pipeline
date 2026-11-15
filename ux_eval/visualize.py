"""Візуалізація звіту: макет, дальтонізм, теплові карти, траєкторія погляду, шум, Thumb Zone, макет після плану."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .layout import _font


def _heat(rgb: np.ndarray, m: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    colored = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (rgb * (1 - alpha) + colored * alpha).astype(np.uint8)


def _zones(rgb: np.ndarray, zones: np.ndarray | None) -> np.ndarray:
    if zones is None:
        return rgb.copy()
    pal = np.array([[60, 200, 90], [250, 200, 40], [235, 70, 60]], dtype=np.float64)
    return (rgb * 0.55 + pal[zones] * 0.45).astype(np.uint8)


def _boxes(img: Image.Image, report: dict, key: str | None):
    d = ImageDraw.Draw(img)
    f = _font(12, True)
    for e in report["elements"]:
        el = report["_layout"].get(e["id"])
        x, y, w, h = el.box
        col = (230, 30, 60) if e["role"] == "cta" else (255, 255, 255)
        d.rectangle([x, y, x + w, y + h], outline=col, width=2 if e["role"] == "cta" else 1)
        if key:
            label = f"{e['attention'][key]:.0f}%"
            tw = d.textlength(label, font=f)
            d.rectangle([x, y, x + tw + 6, y + 16], fill=(0, 0, 0))
            d.text((x + 3, y + 1), label, font=f, fill=col)


def _scanpath(img: Image.Image, path: list[dict]):
    d = ImageDraw.Draw(img)
    f = _font(13, True)
    for a, b in zip(path, path[1:]):
        d.line([a["x"], a["y"], b["x"], b["y"]], fill=(255, 255, 255), width=3)
    for p in path:
        r = 6 + p["t_ms"] / 40
        col = (230, 30, 60) if p["role"] == "cta" else (40, 40, 40)
        d.ellipse([p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r], fill=col, outline=(255, 255, 255), width=2)
        label = str(p["k"])
        d.text((p["x"] - d.textlength(label, font=f) / 2, p["y"] - 8), label, font=f, fill=(255, 255, 255))


def report_panel(report: dict, title: str = "", after: dict | None = None) -> Image.Image:
    """Панель усіх кроків. after — результат ux_eval.auditor.what_if (крок 8: макет після плану дій)."""
    from .color import simulate_cvd
    rgb = report["_rgb"]
    maps = report["_maps"]
    h, w = rgb.shape[:2]
    cvd = np.clip(simulate_cvd(rgb.astype(np.float64), "deuteranopia"), 0, 255).astype(np.uint8)
    tiles = [
        ("1 · Макет і елементи", Image.fromarray(rgb.copy()), None),
        ("2 · Дейтеранопія", Image.fromarray(cvd), None),
        ("3 · Увага 0.5–1.5 с (7: AoI)", Image.fromarray(_heat(rgb, maps["1s"])), "1s"),
        ("3–4 · Увага 7+ с + маршрут", Image.fromarray(_heat(rgb, maps["7s"])), None),
        ("5 · Візуальний шум", Image.fromarray(_heat(rgb, report["_clutter_map"], 0.6)), None),
        ("6 · Thumb Zone", Image.fromarray(_zones(rgb, report["_zones"])), None),
    ]
    if after is not None:
        a = after["_after_report"]["_rgb"]
        if a.shape[:2] != (h, w):
            a = cv2.resize(a.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
        tiles.append(("8 · Після плану дій", Image.fromarray(a.astype(np.uint8)), None))
    _boxes(tiles[0][1], report, None)
    _boxes(tiles[2][1], report, "1s")
    _scanpath(tiles[3][1], report["scanpath"])

    # вузькі (мобільні) екрани — в один ряд; широкі (сайти) — сіткою по 3, плитки не ширші за 900 px
    sc = min(1.0, 900 / w)
    tw, th = max(1, round(w * sc)), max(1, round(h * sc))
    if sc < 1:
        tiles = [(n, t.resize((tw, th), Image.LANCZOS), k) for n, t, k in tiles]
    ncol = len(tiles) if h >= w else 3
    nrow = -(-len(tiles) // ncol)
    gap, head, cap, foot = 12, 34, 24, 86
    panel = Image.new("RGB", (ncol * (tw + gap) + gap, head + nrow * (th + cap + gap) + foot), (250, 250, 252))
    d = ImageDraw.Draw(panel)
    d.text((gap, 6), title, font=_font(18, True), fill=(20, 20, 30))
    for i, (name, tile, _) in enumerate(tiles):
        x = gap + (i % ncol) * (tw + gap)
        y = head + (i // ncol) * (th + cap + gap)
        d.text((x, y + 2), name, font=_font(14, True), fill=(60, 60, 70))
        panel.paste(tile, (x, y + cap))
    h = nrow * (th + cap + gap) - 24  # низ сітки — для підписів нижче
    s = report["scores"]
    iss = s["issues"]
    d.text((gap, h + head + 34),
           f"Оцінка {s['overall']}/100   ·   чистота {s['cleanliness']}/100   ·   "
           f"увага до CTA {s['cta_attention_1s']}%   ·   проблеми: {iss['critical']} критичних, "
           f"{iss['major']} важливих, {iss['minor']} порад   ·   {report['elapsed_ms']} мс на CPU",
           font=_font(15, False), fill=(30, 30, 40))
    if after is not None:
        b, a = after["before"], after["after"]
        d.text((gap, h + head + 56),
               f"Крок 8, якщо застосувати план: оцінка {b['overall']} → {a['overall']}   ·   "
               f"проблем критичних {b['issues']['critical']} → {a['issues']['critical']}   ·   "
               f"увага до CTA {b['cta_attention_1s']}% → {a['cta_attention_1s']}%",
               font=_font(15, False), fill=(30, 30, 40))
    return panel
