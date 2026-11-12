"""Крок 1. Структурна UI-сегментація: «плоска картинка» -> список елементів.

Повна версія — ui_parser.py (OmniParser v2 + YOLO11x-UI + PaddleOCR, GPU).
Тут реалізовано Fallback: морфологія OpenCV + Canny.
Якщо структура макета вже відома (JSON, експорт із Figma), крок пропускається.

Евристики класифікації:
- button: суцільна заливка, відмінна від фону, з текстом усередині, помірний розмір;
- text: висока щільність дрібних контурів, низька висота рядка;
- image: висока колірна дисперсія, велика площа;
- icon: маленький майже квадратний блок.
OCR у Fallback немає — текст елементів лишається порожнім.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import delta_e_rgb, hex_to_rgb, rgb_to_hex
from .layout import Element, Layout


def _dominant_bg(rgb: np.ndarray) -> np.ndarray:
    """Найчастіший колір (квантований) — фон екрана."""
    q = (rgb[..., :3] >> 3).astype(np.int32).reshape(-1, 3)
    code = np.bincount((q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2], minlength=1 << 15).argmax()
    return np.array([(code >> 10) & 31, (code >> 5) & 31, code & 31]) * 8 + 4


def _text_and_fill(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-means у RGB: більший кластер — заливка, менший — текст/пиктограма."""
    data = patch.reshape(-1, 3).astype(np.float32)
    if len(data) < 4:
        m = data.mean(0)
        return m, m
    _, labels, centers = cv2.kmeans(data, 2, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=2)
    return centers[counts.argmin()], centers[counts.argmax()]


def _estimate_font_px(mask: np.ndarray) -> float:
    """Медіанна висота зв'язних компонент (літер) / 0.72 ≈ кегль шрифту."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    hs = hs[(hs > 4) & (hs < mask.shape[0])]
    return float(np.median(hs) / 0.72) if len(hs) else 14.0


def segment(rgb: np.ndarray, min_area_frac: float = 0.0015) -> tuple[Layout, dict]:
    h, w = rgb.shape[:2]
    bg = _dominant_bg(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1) усе, що відрізняється від фону, + краї Canny
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 18
    edges = cv2.Canny(gray, 40, 120) > 0
    fg = (diff | edges).astype(np.uint8) * 255

    # 2) морфологічне закриття зливає літери в рядки, рядки — в блоки
    k = max(3, w // 60)
    blocks = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k * 2, k)))
    contours, _ = cv2.findContours(blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements: list[Element] = []
    for i, cnt in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[1])):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < min_area_frac * w * h or bw * bh > 0.9 * w * h:
            continue
        patch = rgb[y:y + bh, x:x + bw]
        txt, fill = _text_and_fill(patch)
        fill_vs_bg = delta_e_rgb(fill, bg)
        color_std = float(patch.reshape(-1, 3).std(0).mean())
        edge_density = float(edges[y:y + bh, x:x + bw].mean())
        aspect = bw / max(bh, 1)

        if color_std > 45 and bw * bh > 0.03 * w * h:
            kind = "image"
        elif fill_vs_bg > 8 and 1.8 < aspect < 14 and bh < 0.12 * h:
            kind = "button"
        elif 0.7 < aspect < 1.4 and bw < 0.12 * w:
            kind = "icon"
        else:
            kind = "text"

        text_mask = np.abs(patch.astype(np.int16) - fill.astype(np.int16)).max(axis=2) > 30
        font_px = _estimate_font_px(text_mask)
        if kind == "text" and bh > 2 * font_px:
            font_px = min(font_px, bh / 2)
        if kind == "text" and font_px >= 22:
            kind = "heading"

        elements.append(Element(
            id=f"{kind}_{i}", type=kind, box=[x, y, bw, bh],
            role="secondary",
            fill=rgb_to_hex(fill) if kind in ("button", "image", "icon") else None,
            color=rgb_to_hex(txt), font_size=round(font_px, 1),
            text="?" if kind in ("text", "heading", "button") else "",  # OCR немає у Fallback
        ))

    # кандидат у CTA: широка помітна кнопка; нижчі на екрані мають перевагу
    # (у мобільних і веб-макетах головна дія зазвичай унизу картки чи екрана)
    buttons = [e for e in elements if e.type == "button"]
    if buttons:
        def cta_score(e):
            contrast = delta_e_rgb(np.array(hex_to_rgb(e.fill)), bg)
            return contrast * e.box[2] * (0.3 + (e.box[1] + e.box[3] / 2) / h)
        max(buttons, key=cta_score).role = "cta"

    covered = np.zeros((h, w), bool)
    for e in elements:
        x, y, bw, bh = e.box
        covered[y:y + bh, x:x + bw] = True
    meta = {"whitespace_ratio": round(1.0 - float(covered.mean()), 3),
            "background": rgb_to_hex(bg), "n_elements": len(elements)}
    return Layout(w, h, rgb_to_hex(bg), elements), meta


def whitespace_ratio(layout: Layout) -> float:
    """Частка «повітря» для макета з відомою структурою (без карток-підкладок)."""
    covered = np.zeros((layout.height, layout.width), bool)
    for e in layout.elements:
        if e.type in ("card", "divider"):
            continue
        x, y, bw, bh = (int(v) for v in e.box)
        covered[max(0, y):y + bh, max(0, x):x + bw] = True
    return round(1.0 - float(covered.mean()), 3)
