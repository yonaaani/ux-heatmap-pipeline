"""Нейромережі кроку 3: TranSalNet (загальна карта) і TempSAL (карти за часом).

Код авторів і ваги лежать поза проєктом, у папці UX_STEP3_DIR. Авторський код
розрахований на старий PyTorch і лише на GPU, тому завантажувачі нижче адаптують
його без правок у самих файлах: відключають .cuda() і DataParallel на час побудови,
а ваги читають лише як дані (weights_only=True).
"""
from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import os
import sys
import types
import warnings

import cv2
import numpy as np
import torch
from torch import nn

STEP3_DIR = os.environ.get(
    "UX_STEP3_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "step3"))
THIRD_PARTY = os.path.join(STEP3_DIR, "third_party")
WEIGHTS = os.path.join(STEP3_DIR, "weights")
TEMPSAL_SLICES = ("0-1s", "1-2s", "2-3s", "3-4s", "4-5s")


def _device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class _Wrap(nn.Module):
    """Замість DataParallel: ті самі ключі ваг (module.*), але без прив'язки до GPU."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@contextlib.contextmanager
def _author_code(root: str, redirect: dict[str, str] | None = None):
    """Шляхи й заміни лише на час імпорту та побудови авторської моделі."""
    real_load, real_cuda, real_dp = torch.load, nn.Module.cuda, nn.DataParallel
    redirect = redirect or {}

    def safe_load(path, *args, **kwargs):
        return real_load(redirect.get(path, path), map_location="cpu", weights_only=True)

    sys.path.insert(0, root)
    torch.load, nn.Module.cuda, nn.DataParallel = safe_load, lambda self, *a, **k: self, _Wrap
    try:
        yield
    finally:
        torch.load, nn.Module.cuda, nn.DataParallel = real_load, real_cuda, real_dp
        sys.path.remove(root)


def _to_original(pred: np.ndarray, h: int, w: int) -> np.ndarray:
    """Прибирає поля, додані під формат 384×288, і повертає розмір макета."""
    ph, pw = pred.shape
    if h / ph > w / pw:
        new_w = pw * h // ph
        pred = cv2.resize(pred, (new_w, h), interpolation=cv2.INTER_LINEAR)
        left = (new_w - w) // 2
        return pred[:, left:left + w]
    new_h = ph * w // pw
    pred = cv2.resize(pred, (w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (new_h - h) // 2
    return pred[top:top + h, :]


class TranSalNetModel:
    """TranSalNet_Res. Вхід як у авторів: BGR, 384×288 із полями, значення 0..1."""

    def __init__(self, device: str | None = None):
        root = os.path.join(THIRD_PARTY, "transalnet")
        self.device = _device(device)
        # pandas і skimage автори імпортують, але не використовують
        for name in ("skimage", "pandas"):
            if name not in sys.modules and importlib.util.find_spec(name) is None:
                stub = types.ModuleType(name)
                stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
                stub.io, stub.transform = None, None
                sys.modules[name] = stub
        resnet = os.path.join(WEIGHTS, "transalnet", "resnet50.pth")
        with _author_code(root, {"pretrained_models/resnet50-0676ba61.pth": resnet}):
            from TranSalNet_Res import TranSalNet
            model = TranSalNet()
            model.load_state_dict(torch.load(os.path.join(WEIGHTS, "transalnet", "TranSalNet_Res.pth")))
        self.model = model.to(self.device).eval()

    @staticmethod
    def _prepare(rgb: np.ndarray) -> np.ndarray:
        bgr = rgb[..., ::-1].astype(np.uint8)
        h, w = bgr.shape[:2]
        out = np.ones((288, 384, 3), dtype=np.uint8)
        if h / 288 > w / 384:
            new_w = min(384, w * 288 // h)
            left = (384 - new_w) // 2
            out[:, left:left + new_w] = cv2.resize(bgr, (new_w, 288), interpolation=cv2.INTER_LINEAR)
        else:
            new_h = min(288, h * 384 // w)
            top = (288 - new_h) // 2
            out[top:top + new_h, :] = cv2.resize(bgr, (384, new_h), interpolation=cv2.INTER_LINEAR)
        return out

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        x = torch.from_numpy(self._prepare(rgb).transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(self.device)
        pred = self.model(x)
        return _to_original(pred.squeeze().cpu().numpy(), h, w)


class TempSALModel:
    """TempSAL. Вхід як у авторів: RGB, 256×256 без збереження пропорцій, mean/std 0,5."""

    def __init__(self, device: str | None = None):
        root = os.path.join(THIRD_PARTY, "tempsal")
        self.device = _device(device)
        path = os.path.join(WEIGHTS, "tempsal", "multilevel_tempsal.pt")
        with _author_code(root):
            sys.path.insert(0, os.path.join(root, "PNAS"))
            try:
                from model import PNASBoostedModelMultiLevel
                model = PNASBoostedModelMultiLevel(self.device, path, path, 5)
            finally:
                sys.path.remove(os.path.join(root, "PNAS"))
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict_all(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Загальна карта 0-5 с і п'ять карт по секунді, усі в розмірі макета."""
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.transpose(2, 0, 1).copy()).float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(self.device)
        final, volume = self.model(x)
        out = {"0-5s": final.squeeze().cpu().numpy()}
        out.update(zip(TEMPSAL_SLICES, volume.squeeze(0).cpu().numpy()))
        return {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) for k, v in out.items()}

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return self.predict_all(rgb)["0-5s"]


FINETUNED = os.path.join(WEIGHTS, "finetuned")
KEYS3 = ("1s", "3s", "7s")
# який варіант донавчених карт використовувати (mix — середнє двох моделей, найточніше на валідації UEyes)
FINETUNED_MODE = os.environ.get("UX_STEP3_MODE", "mix")


class FineTunedTempSAL:
    """TempSAL, донавчений на UEyes: часова гілка (PNASVolModellast) з трьома виходами 1 / 3 / 7 с."""

    def __init__(self, device: str | None = None, path: str | None = None):
        root = os.path.join(THIRD_PARTY, "tempsal")
        self.device = _device(device)
        with _author_code(root):
            sys.path.insert(0, os.path.join(root, "PNAS"))
            try:
                from model import PNASVolModellast
                inner = PNASVolModellast(time_slices=5, load_weight=0)
            finally:
                sys.path.remove(os.path.join(root, "PNAS"))
        inner.deconv_layer5[4] = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        ckpt = torch.load(path or os.path.join(FINETUNED, "tempsal", "best.pt"), map_location="cpu", weights_only=True)
        inner.load_state_dict({k[len("inner."):]: v for k, v in ckpt["model"].items()})
        # глибше донавчання вписує макет у 256×256 з полями (без сплющування); ознака зберігається у ваговому файлі
        self.keep_aspect = bool(ckpt.get("keep_aspect", False))
        self.model = inner.to(self.device).eval()

    @torch.inference_mode()
    def predict3_small(self, rgb: np.ndarray, size: tuple[int, int]) -> dict[str, np.ndarray]:
        """Три карти у робочому розмірі size=(w, h) з пропорціями макета (float32)."""
        h, w = rgb.shape[:2]
        if self.keep_aspect:
            sc = 256 / max(h, w)
            nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
            x0, y0 = (256 - nw) // 2, (256 - nh) // 2
            small = np.full((256, 256, 3), 128, np.uint8)
            small[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            small = cv2.resize(rgb.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.transpose(2, 0, 1).copy()).float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(self.device)
        out = self.model(x)[0][0].float().cpu().numpy()
        if self.keep_aspect:
            out = out[:, y0:y0 + nh, x0:x0 + nw]
        return {k: cv2.resize(np.maximum(out[i], 0), size, interpolation=cv2.INTER_LINEAR) for i, k in enumerate(KEYS3)}

    def predict3(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w = rgb.shape[:2]
        return {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR) for k, v in self.predict3_small(rgb, (256, 256)).items()}


class FineTunedTranSalNet:
    """TranSalNet_Res, донавчений на UEyes: останній шар декодера дає три карти 1 / 3 / 7 с."""

    def __init__(self, device: str | None = None, path: str | None = None):
        base = TranSalNetModel(device)
        self.device = base.device
        model = base.model
        model.decoder.conv7 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        state = torch.load(path or os.path.join(FINETUNED, "transalnet", "best.pt"), map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict3_small(self, rgb: np.ndarray, size: tuple[int, int]) -> dict[str, np.ndarray]:
        """Три карти без полів 384×288, у робочому розмірі size=(w, h) (float32)."""
        h, w = rgb.shape[:2]
        x = torch.from_numpy(TranSalNetModel._prepare(rgb).transpose(2, 0, 1).copy()).float().div(255)
        out = self.model(x.unsqueeze(0).to(self.device))[0].float().cpu().numpy()
        # поля прибираємо на рідній роздільності моделі (як _to_original, але без збільшення до повного розміру)
        if h / 288 > w / 384:
            nw = min(384, w * 288 // h); left = (384 - nw) // 2; crop = out[:, :, left:left + nw]
        else:
            nh = min(288, h * 384 // w); top = (288 - nh) // 2; crop = out[:, top:top + nh, :]
        return {k: cv2.resize(np.maximum(crop[i], 0), size, interpolation=cv2.INTER_LINEAR) for i, k in enumerate(KEYS3)}

    @torch.inference_mode()
    def predict3(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        h, w = rgb.shape[:2]
        x = torch.from_numpy(TranSalNetModel._prepare(rgb).transpose(2, 0, 1).copy()).float().div(255)
        out = self.model(x.unsqueeze(0).to(self.device))[0].float().cpu().numpy()
        return {k: _to_original(np.maximum(out[i], 0), h, w) for i, k in enumerate(KEYS3)}


ELEMENT_GROUPS = {"text": ("text",), "heading": ("heading",), "price": ("price",), "button": ("button", "chip"),
                  "input": ("input",), "image": ("image",), "icon": ("icon",), "face": ("face",)}


def element_maps(layout, size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Карти елементів кроку 1 за групами типів + CTA + центр екрана, на сітці size=(w, h).

    Щільність елемента спадає з площею (великий абзац не «з'їдає» всю увагу), карта розмита на ~2 % екрана.
    """
    w, h = size
    out = {g: np.zeros((h, w), np.float32) for g in (*ELEMENT_GROUPS, "cta")}
    if layout is not None:
        sx, sy = w / layout.width, h / layout.height
        for el in layout.elements:
            x0, y0 = int(el.box[0] * sx), int(el.box[1] * sy)
            x1, y1 = max(x0 + 1, int(round((el.box[0] + el.box[2]) * sx))), max(y0 + 1, int(round((el.box[1] + el.box[3]) * sy)))
            dens = 1.0 / (1.0 + 8.0 * (el.box[2] * el.box[3]) / (layout.width * layout.height))
            groups = [g for g, types in ELEMENT_GROUPS.items() if el.type in types]
            if el.role == "cta":
                groups.append("cta")
            for g in groups:
                out[g][y0:y1, x0:x1] = np.maximum(out[g][y0:y1, x0:x1], dens)
    sig = max(w, h) / 50
    out = {g: cv2.GaussianBlur(m, (0, 0), sig) for g, m in out.items()}
    yy, xx = np.mgrid[0:h, 0:w]
    out["center"] = np.exp(-(((xx - w / 2) / (0.35 * w)) ** 2 + ((yy - 0.45 * h) / (0.35 * h)) ** 2) / 2).astype(np.float32)
    out["top"] = np.exp(-3.0 * yy / h).astype(np.float32)  # верх екрана (UEyes: мобільні — спершу верхня панель)
    return out


def finetuned_available() -> bool:
    return all(os.path.exists(os.path.join(FINETUNED, m, "best.pt")) for m in ("tempsal", "transalnet"))


class FineTunedMaps:
    """Три карти кроку 3 з донавчених моделей: tempsal | transalnet | mix (середнє нормованих карт)."""

    def __init__(self, device: str | None = None, mode: str = FINETUNED_MODE):
        torch.backends.cudnn.benchmark = True  # розмір входу сталий (256² і 384×288) — автопідбір згорток окупається
        self.mode = mode
        self.tempsal = FineTunedTempSAL(device) if mode in ("tempsal", "mix") else None
        self.transalnet = FineTunedTranSalNet(device) if mode in ("transalnet", "mix") else None

    def predict_maps(self, rgb: np.ndarray, work: int = 384) -> dict[str, np.ndarray]:
        """Карти змішуємо на робочій сітці (≤ 384 px по довшій стороні), а до розміру макета
        збільшуємо один раз у float32 — у float64 на 1920×1080 це забирало ~0,7 с."""
        H, W = rgb.shape[:2]
        sc = min(1.0, work / max(H, W))
        size = (max(1, round(W * sc)), max(1, round(H * sc)))
        parts = [m.predict3_small(rgb, size) for m in (self.tempsal, self.transalnet) if m is not None]
        out = {}
        for k in KEYS3:
            mix = sum(p[k] / max(float(p[k].mean()), 1e-12) for p in parts).astype(np.float32)
            mix *= np.float32(1.0 / (float(mix.mean(dtype=np.float64)) * W * H))  # сума ≈ 1 після збільшення
            out[k] = cv2.resize(mix, (W, H), interpolation=cv2.INTER_LINEAR)
        return out


class NeuralTemporalMaps:
    """Три карти кроку 3 з TempSAL і TranSalNet.

    Варіанти вибрано за CC на 300 зображеннях UEyes Train, тест у виборі не брав участі:
    1s = TempSAL 0-3 с; 3s = TempSAL 0-5 с + TranSalNet; 7s = TranSalNet.
    Для 3s суміш дає CC 0,439 проти 0,443 у самого TranSalNet, зате 3s і 7s не збігаються.
    """

    def __init__(self, device: str | None = None):
        self.transalnet = TranSalNetModel(device)
        self.tempsal = TempSALModel(device)

    def predict_maps(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        ts = self.tempsal.predict_all(rgb)
        tn = self.transalnet.predict(rgb)
        late = _unit(ts["0-5s"]) + _unit(tn)
        early = np.mean([ts["0-1s"], ts["1-2s"], ts["2-3s"]], axis=0)
        return {"1s": _unit(early), "3s": _unit(late), "7s": _unit(tn)}


def _unit(m: np.ndarray) -> np.ndarray:
    total = float(m.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("модель повернула порожню карту")
    return m.astype(np.float64) / total


_neural_model = None
_neural_tried = False


def get_neural_model() -> NeuralTemporalMaps | None:
    """Нейромережі лише на GPU: на CPU працює резервний метод."""
    global _neural_model, _neural_tried
    if not _neural_tried:
        _neural_tried = True
        if torch.cuda.is_available():
            try:
                # донавчені на UEyes ваги, якщо вони є; інакше авторські (SALICON / DHF1K)
                _neural_model = FineTunedMaps("cuda") if finetuned_available() else NeuralTemporalMaps("cuda")
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Нейромережі кроку 3 не завантажились, буде резервний метод: {e!r}")
    return _neural_model
