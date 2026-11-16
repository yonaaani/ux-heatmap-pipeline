"""Крок 5. Карта візуального шуму та когнітивного навантаження: Feature Congestion + Subband Entropy.

Алгоритм — Rosenholtz, Li, Nakano, «Measuring Visual Clutter», Journal of Vision 7(2), 2007 (MATLAB-код авторів,
MIT DSpace 1721.1/37593), у тому вигляді, як його портовано на Python групою інтерфейсів Університету Аалто
(github.com/kargaranamir/visual-clutter; ті самі метрики m7/m8 в Aalto Interface Metrics). Математика і всі константи —
як в еталоні; тут лише
прибрано попіксельні цикли Python (у 10–100 разів швидше), піраміди — та сама бібліотека pyrtools.

Feature Congestion (FC): у трьох масштабах гаусової піраміди рахується локальна мінливість
  • кольору — об'єм еліпсоїда коваріації (L, a, b) у гаусовому вікні σ = 3;
  • контрасту — стандартне відхилення відгуку центр-оточення (DoG, σ = 1) у вікні σ = 3;
  • орієнтацій — 4 орієнтовані фільтри (другі похідні гауса, 0°/45°/90°/135° — орієнтовані фільтри, що їх у
    довідці названо фільтрами Габора), протилежна енергія (cos 2θ, sin 2θ) і об'єм її еліпса коваріації у вікні σ = 14;
  масштаби зводяться максимумом, карта FC = колір/0,2088 + контраст/0,0660 + орієнтації/0,0269, бал — її середнє.
Subband Entropy (SE): керована піраміда (3 масштаби × 4 орієнтації), ентропія Шеннона кожної смуги,
яскравість + 0,0625 × кожен канал кольору, ділення на 1 + 2 × 0,0625.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np

# ---- константи еталона ----
NUM_LEVELS = 3
CONTRAST_FILT_SIGMA = 1.0
CONTRAST_POOL_SIGMA = 3.0
COLOR_POOL_SIGMA = 3.0
DELTA_L2, DELTA_A2, DELTA_B2 = 0.0007 ** 2, 0.1 ** 2, 0.05 ** 2
OPP_NOISE, OPP_FILTER_SCALE, OPP_POOL_SCALE = 1.0, 16 / 14 * 1.75, 1.75
ORIENT_NOISE, ORIENT_POOL_SCALE = 0.001, 7 / 2
COLOR_COEF, CONTRAST_COEF, ORIENT_COEF = 0.2088, 0.0660, 0.0269
SE_LEVELS, SE_ORIENTS, SE_W_CHROM = 3, 4, 0.0625
_K5 = np.array([0.05, 0.25, 0.4, 0.25, 0.05])

# Робоча ширина: еталонні фільтри задані в пікселях, тож результат залежить від роздільності.
# Скриншот зводиться до сталої ширини, щоб бал не залежав від розміру екрана.
WORK_WIDTH = 640


def rgb2lab_ref(im: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Перетворення RGB → Lab рівно як в еталоні (RGB2Lab у utils.py, разом з його масштабом XYZ: ділення на 95,047
    / 100 / 108,833 при XYZ у [0, 1] — так у коді авторів порту й AIM; від цього залежать константи нормування)."""
    im = np.asarray(im)
    if im.dtype == np.uint8:
        lin = _SRGB_LUT[im]  # ті самі значення, що й формула нижче (таблиця на 256 рівнів), але без піднесення до 2,4
    else:
        lin = np.float32(im) / 255
        mask = lin >= 0.04045
        lin[mask] = ((lin[mask] + 0.055) / 1.055) ** 2.4
        lin[~mask] = lin[~mask] / 12.92
    # cv2.transform і cv2.pow — ті самі операції, що np.dot і c ** (1/3), лише векторизовані (різниця ≤ 1e-15)
    # dtype=float32 — швидкий режим (різниця з float64 ~1e-5 одиниці Lab)
    c = cv2.transform(np.ascontiguousarray(lin, dtype=dtype), _XYZ_M.astype(dtype))
    c /= np.array([95.047, 100.000, 108.833], dtype=dtype)
    c = np.where(c >= 0.008856, cv2.pow(c, 1 / 3), (7.787 * c + 16 / 116).astype(c.dtype))
    lab = np.empty_like(c)
    lab[..., 0] = 116 * c[..., 1] - 16
    lab[..., 1] = 500 * (c[..., 0] - c[..., 1])
    lab[..., 2] = 200 * (c[..., 1] - c[..., 2])
    return lab


_XYZ_M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])


def _srgb_lut() -> np.ndarray:
    v = np.arange(256, dtype=np.float32) / 255
    m = v >= 0.04045
    v[m] = ((v[m] + 0.055) / 1.055) ** 2.4
    v[~m] = v[~m] / 12.92
    return v


_SRGB_LUT = _srgb_lut()


def _gauss1d(halfsupport: int, sigma: float, center: float = 0.0) -> np.ndarray:
    t = np.arange(-halfsupport, halfsupport + 1, dtype=np.float64)
    k = np.exp(-(t - center) ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _overlapconv_sep(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """RRoverlapconv по x, потім по y: згортка з нулями за межами і перенормуванням на частку ядра, що в межах."""
    kd = k.astype(x.dtype, copy=False)  # точний режим — float64, швидкий — float32
    num = cv2.sepFilter2D(x, -1, kd, kd, borderType=cv2.BORDER_CONSTANT)
    # частка ядра всередині — добуток одновимірних часток (ядро сепарабельне); залежить лише від розміру,
    # тож множник k.sum()² / (wx·wy) кешується як готова карта
    num *= _overlap_factor(x.shape, k.tobytes(), x.dtype.str)
    return num


@functools.lru_cache(maxsize=64)
def _overlap_factor(shape: tuple[int, int], kb: bytes, dt: str = "<f8") -> np.ndarray:
    k = np.frombuffer(kb, dtype=np.float64)
    wx = cv2.filter2D(np.ones((1, shape[1])), -1, k.reshape(1, -1), borderType=cv2.BORDER_CONSTANT)
    wy = cv2.filter2D(np.ones((shape[0], 1)), -1, k.reshape(-1, 1), borderType=cv2.BORDER_CONSTANT)
    return (k.sum() ** 2 / (wx * wy)).astype(dt)


def _overlapconv_1d(x: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    kk = k.reshape(1, -1) if axis == 1 else k.reshape(-1, 1)
    num = cv2.filter2D(x, -1, kk, borderType=cv2.BORDER_CONSTANT)
    ones = np.ones((1, x.shape[1])) if axis == 1 else np.ones((x.shape[0], 1))
    w = cv2.filter2D(ones, -1, kk, borderType=cv2.BORDER_CONSTANT)
    return (k.sum() * num / w).astype(x.dtype, copy=False)


def _filt2(kernel: np.ndarray, x: np.ndarray) -> np.ndarray:
    """filt2 еталона: згортка з непарним відбиттям країв (симетричні ядра → згортка = кореляція)."""
    return cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _reduce(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    x = _filt2(k.reshape(1, -1), x)[:, ::2]
    return _filt2(k.reshape(-1, 1), x)[::2, :]


def _expand(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """RRoverlapconvexpand: подвоєння з нулями і згортка (ядро ×2) з перенормуванням."""
    h, w = x.shape
    k2 = k * 2
    tmp = np.zeros((h, 2 * w), x.dtype)
    tmp[:, ::2] = x
    tmp = _overlapconv_1d(tmp, k2, axis=1)
    out = np.zeros((2 * h, 2 * w), x.dtype)
    out[::2, :] = tmp
    return _overlapconv_1d(out, k2, axis=0)


_BINOM5 = np.array([1, 4, 6, 4, 1]) / 16 * np.sqrt(2)  # фільтр 'binom5' pyrtools (без нормування)


def _gauss_pyr(x: np.ndarray, n: int = NUM_LEVELS, dtype=np.float64) -> list[np.ndarray]:
    """GaussianPyramid pyrtools (binom5, краї reflect1, крок 2) — тим самим розкладом через OpenCV, у 2 рази швидше
    (різниця з pyrtools ≤ 1e-14)."""
    out = [np.asarray(x, dtype)]
    for _ in range(n - 1):
        out.append(_reduce(out[-1], _BINOM5))
    return out


def _collapse(levels: list[np.ndarray]) -> np.ndarray:
    """Зведення масштабів максимумом (дрібніші рівні збільшуються upConv з ядром 5×5, як в еталоні)."""
    out = levels[0].copy()
    for s in range(1, len(levels)):
        up = levels[s]
        for _ in range(s):
            # upConv pyrtools (крок 2, краї reflect1) = вставка нулів + згортка з відбиттям BORDER_REFLECT_101
            # (перевірено: різниця ≤ 1e-16); ядро 5×5 сепарабельне — sepFilter2D
            t = np.zeros((2 * up.shape[0], 2 * up.shape[1]), up.dtype)
            t[::2, ::2] = up
            k5 = _K5.astype(up.dtype)
            up = cv2.sepFilter2D(t, -1, k5, k5, borderType=cv2.BORDER_REFLECT_101)
        a, b = min(out.shape[0], up.shape[0]), min(out.shape[1], up.shape[1])
        o, u = out[:a, :b], up[:a, :b]
        out[:a, :b] = np.where(u > o, u, o)  # як max() Python в еталоні (NaN не «перемагає»)
    return out


def _color_clutter(Lp, ap, bp) -> np.ndarray:
    g = _gauss1d(round(2 * COLOR_POOL_SIGMA), COLOR_POOL_SIGMA)
    levels = []
    for L, a, b in zip(Lp, ap, bp):
        mL, ma, mb, eLL, eLa, eLb, eaa, eab, ebb = (_overlapconv_sep(x, g) for x in
                                                    (L, a, b, L * L, L * a, L * b, a * a, a * b, b * b))
        c00 = eLL - mL ** 2 + DELTA_L2
        c01 = eLa - mL * ma
        c02 = eLb - mL * mb
        c11 = eaa - ma ** 2 + DELTA_A2
        c12 = eab - ma * mb
        c22 = ebb - mb ** 2 + DELTA_B2
        det = c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02)
        with np.errstate(invalid="ignore"):
            levels.append(np.sqrt(det) ** (1 / 3))
    return _collapse(levels)


def _contrast_clutter(Lp) -> np.ndarray:
    a = round(CONTRAST_FILT_SIGMA * 3)
    gi = _gauss1d(a, 0.71 * CONTRAST_FILT_SIGMA)
    go = _gauss1d(a, 1.14 * CONTRAST_FILT_SIGMA)
    g = _gauss1d(round(CONTRAST_POOL_SIGMA * 2), CONTRAST_POOL_SIGMA)
    levels = []
    for L in Lp:
        inner = _filt2(gi.reshape(-1, 1), _filt2(gi.reshape(1, -1), L))
        outer = _filt2(go.reshape(-1, 1), _filt2(go.reshape(1, -1), L))
        c = np.abs(inner - outer)
        m, m2 = _overlapconv_sep(c, g), _overlapconv_sep(c * c, g)
        levels.append(np.sqrt(np.abs(m2 - m * m)))
    return _collapse(levels)


@functools.lru_cache(maxsize=4)
def _orient_kernels(sigma: float):
    """orient_filtnew еталона: другі похідні гауса в 4 напрямках (повороти на ±45° — skimage, бікубічно, як в еталоні)."""
    from skimage import transform

    def rot(im, angle):
        lo, rng = im.min(), im.max() - im.min()
        return transform.rotate((im - lo) / rng, angle, order=3, resize=False) * rng + lo

    hs = round(3 * sigma)
    gx = _gauss1d(hs, sigma).reshape(1, -1)
    Ga = gx * _gauss1d(hs, sigma, sigma).reshape(-1, 1)
    Gb = gx * _gauss1d(hs, sigma).reshape(-1, 1)
    Gc = gx * _gauss1d(hs, sigma, -sigma).reshape(-1, 1)
    Ga, Gb, Gc = Ga / Ga.sum(), Gb / Gb.sum(), Gc / Gc.sum()
    H = -Ga + 2 * Gb - Gc
    out = [H, H.T]
    for ang in (-45, 45):  # L (ліва діагональ), R (права) — як в еталоні
        A, B, C = (rot(G, ang) for G in (Ga, Gb, Gc))
        out.append(-A / A.sum() + 2 * B / B.sum() - C / C.sum())
    return tuple(out)  # H, V, L, R


@functools.lru_cache(maxsize=16)
def _expand_reduce_matrix(n: int, kb: bytes, dt: str = "<f8"):
    """_reduce(_expand(x)) — лінійна операція, що діє окремо по рядках і стовпцях; її одновимірна матриця n×n
    (розріджена, ~10 ненульових у рядку) будується один раз: тими самими функціями з одиничної матриці."""
    from scipy import sparse
    k = np.frombuffer(kb, dtype=np.float64)
    tmp = np.zeros((n, 2 * n))
    tmp[:, ::2] = np.eye(n)
    tmp = _overlapconv_1d(tmp, k * 2, axis=1)
    m = _filt2(k.reshape(1, -1), tmp)[:, ::2]
    return sparse.csr_matrix(m.astype(dt)), sparse.csr_matrix(m.T.astype(dt))


def _expand_reduce(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """= _reduce(_expand(x, k), k), але без проміжного зображення вдвічі більшого розміру (різниця ≤ 1e-15)."""
    h, w = x.shape
    mx, _ = _expand_reduce_matrix(w, k.tobytes(), x.dtype.str)
    _, myt = _expand_reduce_matrix(h, k.tobytes(), x.dtype.str)
    return np.asarray(myt @ np.asarray((mx.T @ x.T).T))


def _orientation_clutter(Lp) -> np.ndarray:
    H, V, Lk, R = _orient_kernels(OPP_FILTER_SCALE)
    pk = _gauss1d(round(2 * OPP_POOL_SCALE), OPP_POOL_SCALE)
    g = _gauss1d(round(8 * ORIENT_POOL_SCALE), 4 * ORIENT_POOL_SCALE)
    levels = []
    for L in Lp:
        # еталон фільтрує ядром як згорткою (conv2 'same'); ядра L/R після повороту не центросиметричні,
        # тож для кореляції cv2 перевертаємо ядро на 180°
        h, v, l, r = (_expand_reduce(_filt2(np.rot90(K, 2).astype(L.dtype), L) ** 2, pk) for K in (H, V, Lk, R))
        total = h + v + l + r + OPP_NOISE
        cmx, smx = (h - v) / total, (r - l) / total
        Dc, Ds = _overlapconv_sep(cmx, g), _overlapconv_sep(smx, g)
        c00 = _overlapconv_sep(cmx * cmx, g) - Dc ** 2 + ORIENT_NOISE
        c01 = _overlapconv_sep(cmx * smx, g) - Dc * Ds
        c11 = _overlapconv_sep(smx * smx, g) - Ds ** 2 + ORIENT_NOISE
        with np.errstate(invalid="ignore"):
            levels.append((c00 * c11 - c01 ** 2) ** (1 / 4))
    return _collapse(levels)


def _entropy(x: np.ndarray) -> float:
    """entropy() еталона: рівномірні біни, nbins = ⌈√n⌉, підрахунок як histc у MATLAB."""
    x = x.ravel()
    nb = int(np.ceil(np.sqrt(x.size)))
    if nb == 1:
        return 0.0
    # histc MATLAB: біни [e_k, e_k+1) і окремий останній бін для x == max — це гістограма numpy з тими самими
    # межами, з якої значення, рівні максимуму, перенесено в окремий бін (у 10+ разів швидше за digitize)
    cnt, _ = np.histogram(x, bins=nb - 1)
    cnt = cnt.astype(np.float64)
    n_max = float(np.count_nonzero(x == x.max()))
    cnt[-1] -= n_max
    cnt = np.append(cnt, n_max)
    p = cnt / cnt.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _entropy_fast(x: np.ndarray) -> float:
    """Швидкий режим: та сама ентропія, що й _entropy (рівномірні біни, окремий бін для максимуму), але номер біна
    рахується напряму, а підрахунок — bincount (у кілька разів швидше за np.histogram). Відрізняється від
    _entropy лише для значень, що лежать точно на межі біна (похибка округлення float)."""
    x = x.ravel()
    nb = int(np.ceil(np.sqrt(x.size)))
    if nb == 1:
        return 0.0
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return 0.0
    idx = ((x - mn) * ((nb - 1) / (mx - mn))).astype(np.int64)
    np.minimum(idx, nb - 2, out=idx)
    idx[x == mx] = nb - 1  # значення, рівні максимуму, — окремий останній бін (як histc у MATLAB)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    p = cnt / cnt.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@functools.lru_cache(maxsize=8)
def _sf_masks(shape: tuple[int, int]):
    """Частотні маски керованої піраміди pyrtools (SteerablePyramidFreq, real, order 3) для розміру shape.

    Маски залежать лише від розміру, тож беремо їх у pyrtools один раз (з нульового зображення) і далі робимо
    той самий розклад самі — так само, як у SteerablePyramidFreq.__init__, лише без повторного побудування масок."""
    import warnings

    import pyrtools as pt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # «Reconstruction will not be perfect with odd-sized images» — відновлення не потрібне
        p = pt.pyramids.SteerablePyramidFreq(np.zeros(shape), height=SE_LEVELS, order=SE_ORIENTS - 1)
    crops = []
    dims = np.asarray(shape)
    for _ in range(SE_LEVELS):
        ctr = np.ceil((dims + 0.5) / 2).astype(int)
        lodims = np.ceil((dims - 0.5) / 2).astype(int)
        loctr = np.ceil((lodims + 0.5) / 2).astype(int)
        lostart = ctr - loctr
        crops.append((lostart, lostart + lodims))
        dims = lodims
    return (p._hi0mask.reshape(shape), p._lo0mask.reshape(shape), p._himasks, p._anglemasks, p._lomasks, crops)


def _sf_pyr_bands(m: np.ndarray) -> list[np.ndarray]:
    """Смуги керованої піраміди (3 масштаби × 4 орієнтації + 2 залишки) — як pyr_coeffs у pyrtools."""
    from scipy import fft as sfft
    hi0, lo0, himasks, anglemasks, lomasks, crops = _sf_masks(tuple(m.shape))
    imdft = sfft.fftshift(sfft.fft2(m, workers=-1))
    out = [np.real(sfft.ifft2(sfft.ifftshift(imdft * hi0), workers=-1))]
    lodft = imdft * lo0
    k = (-1j) ** (SE_ORIENTS - 1)
    for i in range(SE_LEVELS):
        for am in anglemasks[i]:
            out.append(np.real(sfft.ifft2(sfft.ifftshift(k * lodft * am * himasks[i]), workers=-1)))
        (a0, b0), (a1, b1) = crops[i]
        lodft = lodft[a0:a1, b0:b1] * lomasks[i]
    out.append(np.real(sfft.ifft2(sfft.ifftshift(lodft), workers=-1)))
    return out


@functools.lru_cache(maxsize=8)
def _sf_masks_shifted(shape: tuple[int, int], dt: str = "<f8"):
    """Маски _sf_masks, уже перемножені (−i)³ · кут · високі частоти і зсунуті ifftshift — щоб для кожної смуги
    лишалось одне множення (ifftshift(A·B) = ifftshift(A)·ifftshift(B))."""
    from scipy import fft as sfft
    hi0, lo0, himasks, anglemasks, lomasks, crops = _sf_masks(shape)
    k = (-1j) ** (SE_ORIENTS - 1)
    bands = [sfft.ifftshift(np.stack([k * am * himasks[i] for am in anglemasks[i]]), axes=(-2, -1))
             for i in range(SE_LEVELS)]
    if dt == "<f4":  # швидкий режим: маски в одинарній точності, щоб FFT лишались complex64
        bands = [b.astype(np.complex64) for b in bands]
        return (sfft.ifftshift(hi0).astype(np.float32), lo0.astype(np.float32), bands,
                [m.astype(np.float32) for m in lomasks], crops)
    return sfft.ifftshift(hi0), lo0, bands, lomasks, crops


_FFT_WORKERS = -1  # потоки FFT для Subband Entropy (змінюється для перевірки швидкодії)


def _sf_pyr_bands_many(ms: np.ndarray) -> list[np.ndarray]:
    """_sf_pyr_bands для кількох каналів одразу (ms: [n, H, W]); смуги — масиви [n, ...] у тому самому порядку.
    Ті самі операції, лише FFT усіх каналів і 4 орієнтацій одним викликом (багатопотоково)."""
    from scipy import fft as sfft
    hi0s, lo0, bands, lomasks, crops = _sf_masks_shifted(tuple(ms.shape[1:]), ms.dtype.str)
    imdft = sfft.fft2(ms, axes=(-2, -1), workers=_FFT_WORKERS)  # незсунутий спектр
    out = [np.real(sfft.ifft2(imdft * hi0s, axes=(-2, -1), workers=_FFT_WORKERS))]
    lodft = sfft.fftshift(imdft, axes=(-2, -1)) * lo0
    for i in range(SE_LEVELS):
        z = sfft.ifftshift(lodft, axes=(-2, -1))[:, None] * bands[i]  # [n, 4, h, w]
        b = np.real(sfft.ifft2(z, axes=(-2, -1), workers=_FFT_WORKERS))
        out.extend(b[:, j] for j in range(SE_ORIENTS))
        (a0, b0), (a1, b1) = crops[i]
        lodft = lodft[:, a0:a1, b0:b1] * lomasks[i]
    out.append(np.real(sfft.ifft2(sfft.ifftshift(lodft, axes=(-2, -1)), axes=(-2, -1), workers=_FFT_WORKERS)))
    return out


def subband_entropy(L: np.ndarray, a: np.ndarray, b: np.ndarray, dtype=np.float64, fast: bool = False) -> float:
    chans = [np.asarray(L, dtype)]
    for ch in (a, b):
        ch = np.asarray(ch, dtype)
        chans.append(np.zeros_like(ch) if ch.max() - ch.min() < 0.008 else ch)
    bands = _sf_pyr_bands_many(np.stack(chans))
    h = _entropy_fast if fast else _entropy
    ent = np.array([[h(bd[c]) for c in range(3)] for bd in bands]).mean(axis=0)  # середнє по смугах для кожного каналу
    return float((ent[0] + SE_W_CHROM * ent[1] + SE_W_CHROM * ent[2]) / (1 + 2 * SE_W_CHROM))


def feature_congestion(lab: np.ndarray, dtype=np.float64, color_dtype=np.float64, pool=None) -> dict:
    """dtype — точність контрасту й орієнтацій, color_dtype — кольору (точний режим: обидві float64)."""
    L, a, b = (lab[..., i].astype(np.float32) for i in range(3))
    Lp, ap, bp = _gauss_pyr(L, dtype=color_dtype), _gauss_pyr(a, dtype=color_dtype), _gauss_pyr(b, dtype=color_dtype)
    fut_col = pool.submit(_color_clutter, Lp, ap, bp) if pool is not None else None
    col = None if fut_col else _color_clutter(Lp, ap, bp)
    if dtype != color_dtype:
        Lp = [x.astype(dtype) for x in Lp]
    con = _contrast_clutter(Lp)
    ori = _orientation_clutter(Lp)
    if fut_col:
        col = fut_col.result()
    fc_map = col / COLOR_COEF + con / CONTRAST_COEF + ori / ORIENT_COEF
    return {"map": fc_map, "fc": float(np.mean(fc_map)), "color": float(np.nanmean(col / COLOR_COEF)),
            "contrast": float(np.nanmean(con / CONTRAST_COEF)), "orientation": float(np.nanmean(ori / ORIENT_COEF))}


def _fast_len(n: int) -> int:
    """Найближче до n парне число виду 2^a·3^b·5^c (швидкі розміри для FFT)."""
    best = None
    for c in range(max(2, n // 2 - n // 40), n // 2 + n // 40 + 2):
        m = 2 * c
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1 and (best is None or abs(m - n) < abs(best - n)):
            best = m
    return best or n + (n % 2)


FAST_WIDTH = 320  # швидкий режим: менша робоча ширина (норми UEyes для неї є в clutter_norms.json)


from concurrent.futures import ThreadPoolExecutor as _TPE  # noqa: E402

_FAST_POOL = _TPE(2, thread_name_prefix="clutter")  # постійні потоки: створення пулу на кожен виклик коштує ~1 мс


def clutter_analysis(rgb: np.ndarray, work_width: int = WORK_WIDTH, fast: bool = False) -> dict:
    """Карта шуму (розмір як у скриншота) + FC, SE і бал чистоти 0–100 (калібрування — у cleanliness_score).

    fast=True — швидкий наближений режим (не за еталоном): робоча ширина FAST_WIDTH, контраст, орієнтації й SE в
    одинарній точності (колір лишається float64 — у float32 коваріація кольору дає NaN), SE рахується паралельно з FC.
    Відхилення FC/SE від точного розрахунку на тій самій ширині ≤ 1e-5; бал — за нормами UEyes для 320 px."""
    h, w = rgb.shape[:2]
    if fast:
        work_width = FAST_WIDTH
    if work_width and w != work_width:
        # висота — найближче парне число з множниками 2, 3, 5: FFT керованої піраміди на таких розмірах у рази
        # швидша (314 = 2·157 чи 1138 = 2·569 — повільні); пропорції змінюються менш ніж на 1 %
        hh = _fast_len(max(16, round(work_width * h / w)))
        small = cv2.resize(rgb, (work_width, hh), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    lab = rgb2lab_ref(small, np.float32 if fast else np.float64)
    chans = [lab[..., i].astype(np.float32) for i in range(3)]
    if fast:
        # швидкий режим: ентропія, колір і контраст+орієнтації — у трьох потоках (OpenCV, FFT і numpy на
        # великих масивах відпускають GIL); результат той самий, що й послідовно
        fut_se = _FAST_POOL.submit(subband_entropy, *chans, dtype=np.float32, fast=True)
        fc = feature_congestion(lab, dtype=np.float32, color_dtype=np.float64, pool=_FAST_POOL)
        se = fut_se.result()
    else:
        fc = feature_congestion(lab)
        se = subband_entropy(*chans)
    return {
        "map": cv2.resize(np.nan_to_num(fc["map"]).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR),
        "feature_congestion": round(fc["fc"], 4),
        "components": {k: round(fc[k], 4) for k in ("color", "contrast", "orientation")},
        "subband_entropy": round(se, 4),
        "cleanliness_score": cleanliness_score(fc["fc"], se, work_width),
        "mode": "fast" if fast else "exact", "work_width": work_width,
    }


@functools.lru_cache(maxsize=1)
def _norms() -> dict | None:
    import json
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clutter_norms.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def cleanliness_score(fc: float, se: float, work_width: int = WORK_WIDTH) -> float:
    """Бал чистоти 0–100 (100 — чисто): наскільки екран чистіший за інтерфейси UEyes.

    Перцентиль FC і перцентиль SE серед 1 485 інтерфейсів UEyes (web, mobile, desktop) на тій самій робочій ширині
    (ux_eval/clutter_norms.json); бал = 100 − середнє двох перцентилів. Тобто 80 означає
    «у середньому за двома мірами Rosenholtz чистіший за 80 % реальних інтерфейсів».
    """
    n = _norms()
    if not n:  # норм ще немає — грубо за FC (лише щоб пайплайн працював)
        return round(float(np.clip(100.0 * (1.0 - fc / 10.0), 0, 100)), 1)
    w = min(n["widths"], key=lambda k: abs(int(k) - (work_width or WORK_WIDTH)))
    q = np.asarray(n["percentiles"])
    p_fc = float(np.interp(fc, n["widths"][w]["FC"], q))
    p_se = float(np.interp(se, n["widths"][w]["SE"], q))
    return round(100.0 - (p_fc + p_se) / 2.0, 1)
