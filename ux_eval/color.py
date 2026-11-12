"""Перцептивна колориметрія (крок 2 пайплайну).

- sRGB -> CIELAB (D65)
- колірна відстань CIEDE2000 (Sharma, Wu, Dalal, 2005)
- контраст тексту APCA 0.0.98G-4g (основа WCAG 3.0)
- симуляція дальтонізму, матриці Machado, Oliveira, Fernandes (2009), severity = 1.0
"""
from __future__ import annotations

import functools

import numpy as np

# ---------------------------------------------------------------- перетворення

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(np.clip(c, 0, 255)))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """rgb у діапазоні 0..255 -> лінійний RGB 0..1."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """лінійний RGB 0..1 -> sRGB 0..255."""
    lin = np.clip(lin, 0.0, 1.0)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return c * 255.0


_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb (..., 3) у 0..255 -> Lab (..., 3)."""
    xyz = srgb_to_linear(rgb) @ _M_RGB2XYZ.T / _WHITE_D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > eps, f ** 3, (116 * f - 16) / kappa) * _WHITE_D65
    lin = xyz @ np.linalg.inv(_M_RGB2XYZ).T
    return linear_to_srgb(lin)


_M_OK1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
_M_OK2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb (..., 3) у 0..255 -> Oklab (L 0..1, a, b) — Björn Ottosson (2020); рівноконтрастний простір,
    точніше передає відтінок, ніж CIELAB, у синіх і насичених кольорах."""
    lms = srgb_to_linear(rgb) @ _M_OK1.T
    return np.cbrt(lms) @ _M_OK2.T


def delta_e_ok(rgb1, rgb2) -> float:
    """Евклідова відстань в Oklab ×100 (порівнянна за масштабом з ΔE CIELAB)."""
    return float(np.linalg.norm(rgb_to_oklab(np.asarray(rgb1, float)) - rgb_to_oklab(np.asarray(rgb2, float))) * 100)


# ---------------------------------------------------------------- CIEDE2000

def delta_e2000(lab1: np.ndarray, lab2: np.ndarray, kL=1.0, kC=1.0, kH=1.0) -> np.ndarray:
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_bar ** 7 / (C_bar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lp_bar = (L1 + L2) / 2
    Cp_bar = (C1p + C2p) / 2
    hsum = h1p + h2p
    hp_bar = np.where(np.abs(h1p - h2p) > 180,
                      np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
                      hsum / 2)
    hp_bar = np.where(C1p * C2p == 0, hsum, hp_bar)

    T = (1 - 0.17 * np.cos(np.radians(hp_bar - 30))
         + 0.24 * np.cos(np.radians(2 * hp_bar))
         + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
         - 0.20 * np.cos(np.radians(4 * hp_bar - 63)))
    d_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cp_bar ** 7 / (Cp_bar ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lp_bar - 50) ** 2 / np.sqrt(20 + (Lp_bar - 50) ** 2)
    Sc = 1 + 0.045 * Cp_bar
    Sh = 1 + 0.015 * Cp_bar * T
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc

    return np.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
                   + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))


def delta_e_rgb(rgb1, rgb2) -> float:
    a, b = np.asarray(rgb1, dtype=np.float64), np.asarray(rgb2, dtype=np.float64)
    if a.shape == (3,) and b.shape == (3,):  # пара кольорів — з кешу (ті самі пари повторюються між елементами)
        return _delta_e_pair(tuple(a.tolist()), tuple(b.tolist()))
    return float(delta_e2000(rgb_to_lab(a), rgb_to_lab(b)))


@functools.lru_cache(maxsize=8192)
def _delta_e_pair(a: tuple, b: tuple) -> float:
    return float(delta_e2000(rgb_to_lab(np.array(a)), rgb_to_lab(np.array(b))))


# ---------------------------------------------------------------- APCA

def _apca_y(rgb) -> float:
    c = (np.asarray(rgb, dtype=np.float64) / 255.0) ** 2.4
    y = 0.2126729 * c[0] + 0.7151522 * c[1] + 0.0721750 * c[2]
    if y < 0.022:
        y += (0.022 - y) ** 1.414
    return float(y)


def _apca_y_many(rgb: np.ndarray) -> np.ndarray:
    """_apca_y для масиву кольорів (N, 3) — ті самі формули."""
    c = (np.asarray(rgb, dtype=np.float64) / 255.0) ** 2.4
    y = 0.2126729 * c[..., 0] + 0.7151522 * c[..., 1] + 0.0721750 * c[..., 2]
    return np.where(y < 0.022, y + np.abs(0.022 - y) ** 1.414, y)


def apca_lc_many(text_rgb: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """apca_lc для багатьох пар одразу (text_rgb і bg_rgb — (N, 3) або (3,)); формули ті самі, що й в apca_lc."""
    yt, yb = np.broadcast_arrays(_apca_y_many(text_rgb), _apca_y_many(bg_rgb))
    with np.errstate(invalid="ignore"):
        s1 = (yb ** 0.56 - yt ** 0.57) * 1.14
        s2 = (yb ** 0.65 - yt ** 0.62) * 1.14
    o1 = np.where(s1 < 0.1, 0.0, s1 - 0.027)
    o2 = np.where(s2 > -0.1, 0.0, s2 + 0.027)
    out = np.where(yb > yt, o1, o2)
    return np.where(np.abs(yb - yt) < 0.0005, 0.0, out) * 100.0


def apca_lc(text_rgb, bg_rgb) -> float:
    """Контраст Lc за APCA. Знак: + темний текст на світлому, - світлий на темному."""
    yt, yb = _apca_y(text_rgb), _apca_y(bg_rgb)
    if abs(yb - yt) < 0.0005:
        return 0.0
    if yb > yt:
        sapc = (yb ** 0.56 - yt ** 0.57) * 1.14
        out = 0.0 if sapc < 0.1 else sapc - 0.027
    else:
        sapc = (yb ** 0.65 - yt ** 0.62) * 1.14
        out = 0.0 if sapc > -0.1 else sapc + 0.027
    return out * 100.0


# Повна таблиця APCA «контраст × товщина × кегль» — fontLookupAPCA() 0.1.7 (G), 28.05.2022,
# Myndex Research / Andrew Somers, репозиторій apca-w3 (W3C Software and Document License).
# Рядок: Lc; стовпці: товщина 100..900; значення — мінімальний кегль (px, еталонний шрифт Barlow);
# 777 — лише не-текст (іконки, декор), 999 — заборонено.
_APCA_WEIGHTS = (100, 200, 300, 400, 500, 600, 700, 800, 900)
_APCA_FONT = np.array([
    [0, 999, 999, 999, 999, 999, 999, 999, 999, 999], [10, 999, 999, 999, 999, 999, 999, 999, 999, 999],
    [15, 777, 777, 777, 777, 777, 777, 777, 777, 777], [20, 777, 777, 777, 777, 777, 777, 777, 777, 777],
    [25, 777, 777, 777, 120, 120, 108, 96, 96, 96], [30, 777, 777, 120, 108, 108, 96, 72, 72, 72],
    [35, 777, 120, 108, 96, 72, 60, 48, 48, 48], [40, 120, 108, 96, 60, 48, 42, 32, 32, 32],
    [45, 108, 96, 72, 42, 32, 28, 24, 24, 24], [50, 96, 72, 60, 32, 28, 24, 21, 21, 21],
    [55, 80, 60, 48, 28, 24, 21, 18, 18, 18], [60, 72, 48, 42, 24, 21, 18, 16, 16, 18],
    [65, 68, 46, 32, 21.75, 19, 17, 15, 16, 18], [70, 64, 44, 28, 19.5, 18, 16, 14.5, 16, 18],
    [75, 60, 42, 24, 18, 16, 15, 14, 16, 18], [80, 56, 38.25, 23, 17.25, 15.81, 14.81, 14, 16, 18],
    [85, 52, 34.5, 22, 16.5, 15.625, 14.625, 14, 16, 18], [90, 48, 32, 21, 16, 15.5, 14.5, 14, 16, 18],
    [95, 45, 28, 19.5, 15.5, 15, 14, 13.5, 16, 18], [100, 42, 26.5, 18.5, 15, 14.5, 13.5, 13, 16, 18],
    [105, 39, 25, 18, 14.5, 14, 13, 12, 16, 18], [110, 36, 24, 18, 14, 13, 12, 11, 16, 18],
    [115, 34.5, 22.5, 17.25, 12.5, 11.875, 11.25, 10.625, 14.5, 16.5],
    [120, 33, 21, 16.5, 11, 10.75, 10.5, 10.25, 13, 15], [125, 32, 20, 16, 10, 10, 10, 10, 12, 14]])
_APCA_DELTA = np.array([
    [0] * 10, [10] + [0] * 9, [15] + [0] * 9, [20] + [0] * 9,
    [25, 0, 0, 0, 12, 12, 12, 24, 24, 24], [30, 0, 0, 12, 12, 36, 36, 24, 24, 24],
    [35, 0, 12, 12, 36, 24, 18, 16, 16, 16], [40, 12, 12, 24, 18, 16, 14, 8, 8, 8],
    [45, 12, 24, 12, 10, 4, 4, 3, 3, 3], [50, 16, 12, 12, 4, 4, 3, 3, 3, 3],
    [55, 8, 12, 6, 4, 3, 3, 2, 2, 0], [60, 4, 2, 10, 2.25, 2, 1, 1, 0, 0],
    [65, 4, 2, 4, 2.25, 1, 1, 0.5, 0, 0], [70, 4, 2, 4, 1.5, 2, 1, 0.5, 0, 0],
    [75, 4, 3.75, 1, 0.75, 0.188, 0.188, 0, 0, 0], [80, 4, 3.75, 1, 0.75, 0.188, 0.188, 0, 0, 0],
    [85, 4, 2.5, 1, 0.5, 0.125, 0.125, 0, 0, 0], [90, 3, 4, 1.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
    [95, 3, 1.5, 1, 0.5, 0.5, 0.5, 0.5, 0, 0], [100, 3, 1.5, 0.5, 0.5, 0.5, 0.5, 1, 0, 0],
    [105, 3, 1, 0, 0.5, 1, 1, 1, 0, 0], [110, 1.5, 1.5, 0.75, 1.5, 1.125, 0.75, 0.375, 1.5, 1.5],
    [115, 1.5, 1.5, 0.75, 1.5, 1.125, 0.75, 0.375, 1.5, 1.5], [120, 1, 1, 0.5, 1, 0.75, 0.5, 0.25, 1, 1],
    [125] + [0] * 9])


def apca_font_lookup(lc: float) -> dict[int, float]:
    """Порт fontLookupAPCA(): {товщина: мінімальний кегль px} для даного |Lc|, з тією самою інтерполяцією."""
    c = min(abs(float(lc)), 125.0)
    # у JS індекс рядка = int(Lc×0,2) з рядком-заголовком під номером 0; у нас заголовка немає, тому −1
    idx = max(0, (1 if c == 0 else int(c * 0.2)) - 1)
    adj = (c - _APCA_FONT[idx][0]) * 0.2
    out = {}
    for w, wt in enumerate(_APCA_WEIGHTS, start=1):
        f = float(_APCA_FONT[idx][w])
        if f > 400:
            out[wt] = f
        elif c < 14.5:
            out[wt] = 999.0
        elif c < 29.5:
            out[wt] = 777.0
        elif f > 24:
            out[wt] = float(np.floor(f - _APCA_DELTA[idx][w] * adj + 0.5))  # як Math.round у JS (не банківське)
        else:
            out[wt] = f - int(2.0 * _APCA_DELTA[idx][w] * adj) * 0.5
    return out


@functools.lru_cache(maxsize=4096)  # той самий результат для того самого кеглю й товщини — пошук один раз
def apca_min_lc_full(font_size_px: float, weight: int = 400) -> float:
    """Найменший |Lc|, за якого текст цього кеглю й товщини проходить повну таблицю APCA (крок 0,5)."""
    wt = min(_APCA_WEIGHTS, key=lambda v: abs(v - weight))
    for lc in np.arange(0, 125.5, 0.5):
        need = apca_font_lookup(lc)[wt]
        if need < 777 and font_size_px >= need:
            return float(lc)
    return 125.0


def apca_min_lc(font_size_px: float, bold: bool = False, mode: str | None = None) -> float:
    """Мінімальний |Lc| для тексту. mode: "full" — повна таблиця APCA (товщина 400 або 700 для жирного,
    бо крок 1 розрізняє лише ці дві), "bronze" — спрощені рівні 75/60/45. Типово — APCA_MODE."""
    if (mode or APCA_MODE) == "full":
        return apca_min_lc_full(font_size_px, 700 if bold else 400)
    return apca_min_lc_bronze(font_size_px, bold)


APCA_MODE = "full"


def apca_min_lc_bronze(font_size_px: float, bold: bool = False) -> float:
    """Мінімальний |Lc| за APCA Bronze Simple Mode.

    Lc 75 — основний текст; Lc 60 — великий текст (≥ 24px або ≥ 16px жирний);
    Lc 45 — заголовки (≥ 36px або ≥ 24px жирний).
    """
    if font_size_px >= 36 or (bold and font_size_px >= 24):
        return 45.0
    if font_size_px >= 24 or (bold and font_size_px >= 16):
        return 60.0
    return 75.0


# ---------------------------------------------------------------- дальтонізм

CVD_MATRICES = {
    "protanopia": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deuteranopia": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "tritanopia": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ]),
}


def simulate_cvd(rgb: np.ndarray, kind: str) -> np.ndarray:
    """rgb (..., 3) у 0..255 -> як це бачить людина з даним типом дальтонізму."""
    lin = srgb_to_linear(rgb)
    sim = lin @ CVD_MATRICES[kind].T
    return linear_to_srgb(sim)


# ---------------------------------------------------------------- підбір кольорів

def adjust_lightness_for_apca(text_rgb, bg_rgb, target_lc: float):
    """Змінює світлоту L* тексту (зберігаючи відтінок), доки |Lc| >= target_lc.

    Повертає найближчий до вихідного колір або None, якщо ціль недосяжна. Усі кандидати (крок L* = 1)
    рахуються одним масивом, і береться перший, що проходить, — той самий результат, що й перебір по одному.
    """
    lab = rgb_to_lab(np.asarray(text_rgb, dtype=np.float64))
    bg_y = _apca_y(bg_rgb)
    go_darker = bg_y > 0.36  # на світлому фоні текст темнішаємо, на темному — світлішаємо
    steps = np.arange(0, 101, dtype=np.float64)
    L = lab[0] - steps if go_darker else lab[0] + steps
    L = L[: int(np.argmax(~((L >= 0) & (L <= 100))))] if not np.all((L >= 0) & (L <= 100)) else L
    if L.size == 0:
        return None
    cand = np.clip(lab_to_rgb(np.stack([L, np.full_like(L, lab[1]), np.full_like(L, lab[2])], axis=-1)), 0, 255)
    ok = np.abs(apca_lc_many(cand, np.asarray(bg_rgb, dtype=np.float64))) >= target_lc
    return cand[int(np.argmax(ok))] if ok.any() else None
