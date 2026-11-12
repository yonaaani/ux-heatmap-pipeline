"""Опис макета у вигляді JSON і його рендер у картинку.

Пайплайн приймає або скріншот (тоді структуру знаходить крок 1), або
готову структуру макета — наприклад, експорт із Figma. Формат:

{
  "width": 390, "height": 720, "background": "#F4F1EC",
  "elements": [
    {"id": "cta", "type": "button", "role": "cta", "box": [28, 590, 334, 56],
     "text": "Купити", "fill": "#1F6FEB", "color": "#FFFFFF",
     "font_size": 18, "bold": true, "radius": 14}
  ]
}

type: card | heading | text | price | button | chip | input | image | icon | face | divider
role: cta (головна дія) | primary | secondary | decor (декор)
"""
from __future__ import annotations

import copy
import json
import os
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import hex_to_rgb

TEXT_TYPES = {"heading", "text", "price", "button", "chip"}


@dataclass
class Element:
    id: str
    type: str
    box: list[float]
    role: str = "secondary"
    text: str = ""
    fill: str | None = None
    color: str = "#222222"
    font_size: float = 16
    bold: bool = False
    radius: float = 0
    align: str = "left"
    palette: list[str] = field(default_factory=list)  # для зображень-заглушок
    bg: str | None = None       # виміряний колір під текстом (для макетів зі скріншота)
    confidence: float = 1.0     # впевненість детектора (1.0 для JSON-макетів)
    source: str = "json"        # хто знайшов елемент: json | yolo_ui | omniparser | ocr | opencv


@dataclass
class Layout:
    width: int
    height: int
    background: str
    elements: list[Element]
    # пристрій, якщо відомий (плагін Figma: ширина фрейму в логічних px): mobile | tablet | desktop;
    # None — визначається за розміром у кроці 6 (ergonomics)
    device: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls(d["width"], d["height"], d.get("background", "#FFFFFF"),
                   [Element(**e) for e in d["elements"]], d.get("device"))

    @classmethod
    def from_json(cls, path: str) -> "Layout":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Layout":
        return copy.deepcopy(self)

    def get(self, el_id: str) -> Element:
        return next(e for e in self.elements if e.id == el_id)

    def underlying_color(self, el: Element) -> str:
        """Колір, на якому лежить елемент: найближчий попередній фон, що його містить."""
        idx = self.elements.index(el)
        x, y, w, h = el.box
        for other in reversed(self.elements[:idx]):
            if other.fill and other.type in ("card", "image", "button", "chip"):
                ox, oy, ow, oh = other.box
                if ox <= x and oy <= y and x + w <= ox + ow and y + h <= oy + oh:
                    return other.fill
        return self.background


# ---------------------------------------------------------------- рендер

_FONT_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
_FONT_FILES = {False: ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
               True: ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]}


def _font(size: float, bold: bool) -> ImageFont.FreeTypeFont:
    for name in _FONT_FILES[bold]:
        for path in (os.path.join(_FONT_DIR, name), name):
            try:
                return ImageFont.truetype(path, int(round(size)))
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_image_placeholder(img: Image.Image, el: Element):
    """Ілюстрація-заглушка: кола з палітри (чим строкатіша палітра — тим більше шуму)."""
    x, y, w, h = (int(v) for v in el.box)
    layer = Image.new("RGB", (w, h), el.fill or "#DDDDDD")
    d = ImageDraw.Draw(layer)
    rng = np.random.default_rng(zlib.crc32(el.id.encode()))
    palette = el.palette or ["#CCCCCC"]
    n = 3 if len(palette) <= 2 else 18
    for i in range(n):
        r = rng.uniform(0.12, 0.35) * min(w, h) if n == 3 else rng.uniform(0.05, 0.2) * min(w, h)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=palette[i % len(palette)])
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=el.radius, fill=255)
    img.paste(layer, (x, y), mask)


def render(layout: Layout) -> np.ndarray:
    img = Image.new("RGB", (layout.width, layout.height), layout.background)
    draw = ImageDraw.Draw(img)
    for el in layout.elements:
        x, y, w, h = el.box
        if el.type == "image":
            _draw_image_placeholder(img, el)
            continue
        if el.type == "divider":
            draw.rectangle([x, y, x + w, y + h], fill=el.fill or el.color)
            continue
        if el.type == "icon":
            draw.ellipse([x, y, x + w, y + h], fill=el.fill or el.color)
            continue
        if el.fill and el.type in ("card", "button", "chip"):
            draw.rounded_rectangle([x, y, x + w - 1, y + h - 1], radius=el.radius, fill=el.fill)
        if el.text:
            font = _font(el.font_size, el.bold)
            pad = 14 if el.type in ("button", "chip") else 0
            lines = _wrap(draw, el.text, font, w - 2 * pad)
            line_h = el.font_size * 1.3
            total_h = line_h * len(lines)
            ty = y + (h - total_h) / 2 if el.type in ("button", "chip") else y
            for line in lines:
                lw = draw.textlength(line, font=font)
                if el.align == "center" or el.type in ("button", "chip"):
                    tx = x + (w - lw) / 2
                else:
                    tx = x + pad
                draw.text((tx, ty), line, font=font, fill=hex_to_rgb(el.color))
                ty += line_h
    return np.asarray(img)
