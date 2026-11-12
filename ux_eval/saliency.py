"""Мультичасові карти уваги (крок 3 пайплайну) — базова CPU-версія.

Це НЕ нейромережа, а класичний «Fallback»:
- bottom-up: спектральний залишок (Hou & Zhang, CVPR 2007) + центр-оточення
  за каналами L, a, b у дусі Itti-Koch-Niebur (1998);
- top-down: семантичний пріор за типами елементів макета (знаємо їх із
  JSON-структури макета або з кроку 1) + пріор читання зверху вниз
  зліва направо (F-pattern);
- центральний пріор екрана.

Часові карти (0.5–1.5 с, 3–5 с, 7+ с) — зважені суміші цих складових,
що імітують ідею TempSAL: спочатку домінують яскраві плями, потім структура
й текст. Інтерфейс `SaliencyModel` дозволяє пізніше підставити TranSalNet,
DeepGaze чи TempSAL без зміни решти коду.
"""
from __future__ import annotations

import warnings
from typing import Protocol

import cv2
import numpy as np

from .color import rgb_to_lab

# ваги семантичного пріору (за висновками UEyes, CHI 2023: текст, обличчя
# та кнопки притягують погляд в інтерфейсах сильніше, ніж на природних фото)
SEMANTIC_WEIGHTS = {
    "face": 1.0,
    "heading": 0.85,
    "price": 0.8,   # ціни — один із перших об'єктів погляду в e-commerce
    "button": 0.8,
    "input": 0.5,
    "image": 0.55,
    "text": 0.5,
    "chip": 0.45,
    "icon": 0.4,
    "card": 0.0,
    "divider": 0.0,
}

# суміші складових для трьох часових вікон: (bottom-up, семантика, читання, центр)
TIME_MIX = {
    "1s": (0.60, 0.15, 0.05, 0.20),
    "3s": (0.35, 0.40, 0.15, 0.10),
    "7s": (0.20, 0.45, 0.30, 0.05),
}


class SaliencyModel(Protocol):
    """Будь-яка модель, що з RGB-картинки (H, W, 3) робить карту (H, W)."""

    def predict(self, rgb: np.ndarray) -> np.ndarray: ...


def _norm01(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float64)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def _itti_n(m: np.ndarray) -> np.ndarray:
    """Оператор нормалізації N(.) Itti-Koch: підсилює карти з одним яскравим піком."""
    m = _norm01(m)
    local_max = cv2.dilate(m, np.ones((7, 7), np.uint8))
    peaks = m[(m == local_max) & (m > 0.1)]
    mean_peak = peaks[peaks < 1.0].mean() if (peaks < 1.0).any() else 0.0
    return m * (1.0 - mean_peak) ** 2


def spectral_residual(rgb: np.ndarray, width: int = 64) -> np.ndarray:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    small = cv2.resize(gray, (width, max(1, round(width * h / w))), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-9)
    phase = np.angle(f)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    return cv2.resize(_norm01(sal), (w, h), interpolation=cv2.INTER_LINEAR)


class BottomUpSaliency:
    """Спектральний залишок + центр-оточення за кольором та яскравістю."""

    def __init__(self, work_width: int = 256):
        self.work_width = work_width

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        ww = self.work_width
        small = cv2.resize(rgb, (ww, max(1, round(ww * h / w))), interpolation=cv2.INTER_AREA)
        # CIELAB через OpenCV у float32 (розбіжність із rgb_to_lab ≤ 0,3 — для карти уваги неважливо, але в 20 разів швидше)
        lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        maps = []
        for ch in range(3):
            c = np.ascontiguousarray(lab[..., ch])
            blur = {s: cv2.GaussianBlur(c, (0, 0), s) for s in (1, 2, 3, 4, 8, 16)}  # кожен масштаб — один раз
            cs = sum(np.abs(blur[a] - blur[b]) for a, b in ((1, 4), (2, 8), (3, 16)))
            maps.append(_itti_n(cs))
        # колірні канали a, b ділять одну «колірну» вагу, як у Itti-Koch
        cs_map = 0.5 * maps[0] + 0.25 * maps[1] + 0.25 * maps[2]
        if (w, h) != (small.shape[1], small.shape[0]):
            cs_map = cv2.resize(cs_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        sr = spectral_residual(rgb)
        return _norm01(0.6 * _norm01(cs_map) + 0.4 * sr)


def center_prior(h: int, w: int, sigma_frac: float = 0.35) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.45, w * 0.5  # погляд у середньому трохи вище центру
    return np.exp(-(((x - cx) / (sigma_frac * w)) ** 2 + ((y - cy) / (sigma_frac * h)) ** 2) / 2)


def reading_prior(h: int, w: int) -> np.ndarray:
    """F-pattern: увага спадає зверху вниз і зліва направо."""
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-1.6 * y / h) * np.exp(-0.9 * x / w)


def semantic_prior(layout, h: int, w: int) -> np.ndarray:
    """h, w — розмір робочої карти; рамки елементів масштабуються під нього."""
    m = np.zeros((h, w), dtype=np.float64)
    sx, sy = w / layout.width, h / layout.height
    for el in layout.elements:
        wgt = SEMANTIC_WEIGHTS.get(el.type, 0.3)
        if el.role == "cta":
            wgt = max(wgt, 0.9)
        elif el.role == "decor":
            wgt *= 0.6
        if wgt <= 0:
            continue
        x, y, bw, bh = el.box[0] * sx, el.box[1] * sy, el.box[2] * sx, el.box[3] * sy
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
        if x1 <= x0 or y1 <= y0:
            continue
        # щільність спадає з площею: великий абзац не має «з'їдати» всю увагу
        area_frac = (bw * bh) / (w * h)
        density = wgt / (1.0 + 8.0 * area_frac)
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], density)
    return cv2.GaussianBlur(m, (0, 0), max(h, w) / 60)


def temporal_maps(rgb: np.ndarray, layout=None, model: SaliencyModel | None = None,
                  work_width: int = 160) -> dict[str, np.ndarray]:
    """Три карти уваги S(x, y) у розмірі макета, кожна нормована так, що сума = 1.

    Модель із predict_maps (нейромережі кроку 3) сама дає всі три карти. Якщо вона
    збоїть, працює резервний метод нижче, щоб пайплайн не зупинявся.
    Рахуємо на зменшеній копії (увага — низькочастотний сигнал), потім розтягуємо.
    """
    if hasattr(model, "predict_maps"):
        try:
            return model.predict_maps(rgb)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Нейромережі кроку 3 не спрацювали, беру резервний метод: {e!r}")
            model = None
    H, W = rgb.shape[:2]
    w = min(W, work_width)
    h = max(1, round(w * H / W))
    step = max(1, min(W // (2 * w), H // (2 * h)))  # великий екран спершу проріджуємо (у 2+ рази більше за ціль),
    src = np.ascontiguousarray(rgb[::step, ::step]) if step > 1 else rgb  # далі INTER_AREA усереднює решту
    small = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    bu = (model or BottomUpSaliency(work_width=w)).predict(small)
    sem = semantic_prior(layout, h, w) if layout is not None else np.zeros((h, w))
    parts = (_norm01(bu), _norm01(sem), reading_prior(h, w), center_prior(h, w))
    out = {}
    for key, mix in TIME_MIX.items():
        s = sum(k * p for k, p in zip(mix, parts))
        s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), max(h, w) / 100)
        # нормуємо ще на малій карті (білінійне збільшення зберігає середнє), тож суми на повному розмірі не треба;
        # до повного розміру — у float32 (у 4 рази менше пам'яті й часу, ніж float64)
        s *= np.float32(1.0 / (float(s.mean(dtype=np.float64)) * W * H))
        out[key] = cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
    return out


def attention_share(s: np.ndarray, box, total: float | None = None) -> float:
    """AoI_i = сума S у рамці елемента / сума S на всьому макеті × 100 %.
    total — уже порахована сума S (на довгій сторінці сума всієї карти для кожного елемента помітно гальмує)."""
    h, w = s.shape
    x, y, bw, bh = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(s[y0:y1, x0:x1].sum() / (s.sum() if total is None else total) * 100.0)
