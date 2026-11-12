"""Крок 1. Структурна UI-сегментація та OCR (повна версія).

Поєднує три моделі:
- YOLO11-UI   — детектор UI-компонентів із класами (button, input, icon, image,
                text, link, ...), навчений на Rico / WebUI;
- OmniParser v2 (icon_detect) — детектор інтерактивних елементів від Microsoft
                (YOLO, один клас «інтерактивна область»);
- PaddleOCR (PP-OCRv5, модель eslav для української/російської/білоруської)
                — знаходить і читає текст.

Результат — Layout (ті самі елементи, що й у JSON-макеті) + метадані:
Whitespace Ratio, час кожної моделі, які моделі були доступні.

Graceful Degradation: якщо немає ваг YOLO11-UI — працюємо з OmniParser + OCR;
немає OmniParser або OCR — додаємо елементи з OpenCV-фолбека (segmentation.py).
"""
from __future__ import annotations

import os
import re
import time

import cv2
import numpy as np

from .color import delta_e_rgb, hex_to_rgb, rgb_to_hex
from .layout import Element, Layout
from .segmentation import segment as opencv_segment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNI_WEIGHTS = os.path.join(ROOT, "models", "omniparser", "icon_detect", "model.pt")
# YOLO11x, 25 класів Rico / Enrico / WebUI (E5); E1 (YOLO11n, 4 класи) — лише якщо ваг E5 немає
YOLO_UI_E5 = os.path.join(ROOT, "models", "yolo11x_rico25", "best.pt")
YOLO_UI_E1 = os.path.join(ROOT, "models", "yolo11_ui", "best.pt")
YOLO_UI_WEIGHTS = YOLO_UI_E5 if os.path.exists(YOLO_UI_E5) else YOLO_UI_E1

# класи датасету UI-Elements-Detection -> типи нашого макета
YOLO_UI_TYPE = {
    "button": "button", "clickable": "button",
    "input": "input", "textarea": "input", "select": "input", "dropdown": "input",
    "checkbox": "icon", "radio": "icon", "toggle": "icon", "slider": "icon",
    "icon": "icon", "image": "image",
    "text": "text", "label": "text", "link": "text", "menu_item": "text",
    "heading": "text",  # WebUI; заголовок vs текст далі визначає кегль з OCR
    # 25 класів Rico
    "Text": "text", "Text Button": "button", "Input": "input", "Number Stepper": "input", "Date Picker": "input",
    "Image": "image", "Background Image": "image", "Advertisement": "image", "Map View": "image", "Video": "image",
    "Icon": "icon", "Radio Button": "icon", "Checkbox": "icon", "On/Off Switch": "icon", "Slider": "icon",
    "Pager Indicator": "icon",
}
# класи-контейнери містять інші елементи: це не окремі елементи макета, інакше вони перекрили б
# кнопки й текст усередині та зіпсували частку «повітря»; їх передаємо окремо в meta["containers"]
YOLO_UI_CONTAINERS = {"List Item", "Card", "Toolbar", "Drawer", "Modal", "Multi-Tab", "Button Bar",
                      "Bottom Navigation", "Web View"}

# дієслова головних дій (CTA) — укр./рос./англ.
CTA_WORDS = re.compile(
    r"купи|замов|оформ|кошик|корзин|оплат|придбат|почат|продовж|зареєстр|регистр|підпис|"
    r"подпис|отрима|спробу|завантаж|скача|увійти|войти|записат|"
    r"buy|order|checkout|cart|sign ?up|get started|subscribe|start|continue|download|try|book|join",
    re.IGNORECASE)


# плитки для широких екранів (крок 1, покращення для сайтів): E5 навчено на 960 px
TILE_MIN_W, TILE_W, TILE_OVERLAP = 1400, 1000, 96

# дії (кнопки, а не чипи-варіанти): CTA + вхід, пошук, надсилання
ACTION_WORDS = re.compile(CTA_WORDS.pattern + r"|sign ?in|log ?in|login|log ?out|register|search|submit|send|"
                          r"вхід|увійд|вийти|пошук|знайти|надісл|відправ|поиск|найти|войти", re.IGNORECASE)

FACE_YUNET = os.path.join(ROOT, "models", "face", "face_detection_yunet_2023mar.onnx")


class FaceDetector:
    """Пошук облич: YuNet (нейромережа, якщо є файл моделі) або каскад Хаара (вбудований в OpenCV)."""

    def __init__(self, min_size: int = 24):
        self.min_size = min_size
        if os.path.exists(FACE_YUNET):
            self.name = "yunet"
            # поріг 0,5 — найкращий F1 на фото з розміченими обличчями (OSIE)
            self.net = cv2.FaceDetectorYN.create(FACE_YUNET, "", (320, 320), score_threshold=0.5)
        else:
            self.name = "haar"
            self.net = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,
                                                          "haarcascade_frontalface_alt2.xml"))

    def detect(self, rgb: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        h, w = rgb.shape[:2]
        if self.name == "yunet":
            self.net.setInputSize((w, h))
            _, faces = self.net.detect(rgb[..., ::-1].copy())
            out = []
            for f in faces if faces is not None else []:
                x, y, fw, fh = (int(v) for v in f[:4])
                if min(fw, fh) >= self.min_size:
                    out.append(((x, y, x + fw, y + fh), float(f[-1])))
            return out
        # зменшуємо лише екрани ширші за 1920 px: перевірка на 76 скріншотах показала, що
        # зменшення 1920 -> 960 прискорює лише з 0.22 до 0.12 с, але губить 8 облич із 19
        s = min(1.0, 1920 / w)
        gray = cv2.cvtColor(cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                            if s < 1 else rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        # detectMultiScale2 швидший за detectMultiScale3 і теж дає міру впевненості —
        # кількість сусідніх збігів (≥ 6 за порогом minNeighbors; 20+ — дуже впевнено)
        # ultralytics при імпорті вимикає потоки OpenCV (cv2.setNumThreads(0)) — каскад тоді
        # працює на одному ядрі (0.41 с замість 0.15 с); вмикаємо потоки лише на час пошуку
        prev = cv2.getNumThreads()
        cv2.setNumThreads(os.cpu_count() or 4)
        try:
            rects, neighbors = self.net.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=6,
                                                          minSize=(self.min_size, self.min_size))
        finally:
            cv2.setNumThreads(prev)
        return [((int(x / s), int(y / s), int((x + fw) / s), int((y + fh) / s)), float(min(1.0, nb / 20)))
                for (x, y, fw, fh), nb in zip(rects, neighbors)]


# ---------------------------------------------------------------- геометрія

def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside_frac(inner, outer) -> float:
    """Яка частка площі inner лежить усередині outer."""
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def _xywh(b) -> list[int]:
    return [int(round(b[0])), int(round(b[1])), int(round(b[2] - b[0])), int(round(b[3] - b[1]))]


# ---------------------------------------------------------------- кольори і шрифт

def _two_colors(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-means: (колір тексту/знака, колір фону, маска тексту)."""
    data = patch.reshape(-1, 3).astype(np.float32)
    if len(data) < 8:
        m = data.mean(0)
        return m, m, np.zeros(patch.shape[:2], bool)
    _, labels, centers = cv2.kmeans(data, 2, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    1, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(patch.shape[:2])
    counts = np.bincount(labels.ravel(), minlength=2)
    fg = int(counts.argmin())
    return centers[fg], centers[1 - fg], labels == fg


# ΔE00 між заливкою елемента і фоном навколо, з якого вважаємо, що заливка є
# (посилання без заливки ≈ 0.2; бліда «пігулка» #EEF0F3 на білому ≈ 3.5)
FILL_JND = 2.0
BG_IMAGE_FRAC = float(os.environ.get("UX_BG_IMAGE_FRAC", "0.5"))  # картинка на таку частку екрана й більше — фон: тексти поверх неї окремі
BOLD_STROKE_RATIO = 0.118  # звичайний текст ≈ 0.09–0.10, жирний ≈ 0.13–0.15 (виміряно)


X_HEIGHT, CAP_HEIGHT = 0.52, 0.70  # частки кеглю для типових гротесків (Segoe UI, Arial, Roboto)


def _core_band(mask: np.ndarray) -> int:
    """Висота «тіла» рядка: найдовша смуга рядків з густиною ≥ 45 % від максимуму."""
    prof = mask.sum(axis=1).astype(float)
    if prof.max() == 0:
        return 0
    best = cur = 0
    for on in prof >= 0.45 * prof.max():
        cur = cur + 1 if on else 0
        best = max(best, cur)
    return best


def _font_metrics(patch: np.ndarray, fg: np.ndarray, bg: np.ndarray, mask: np.ndarray,
                  text: str = "") -> tuple[float, bool]:
    """Кегль (px) і жирність одного рядка тексту.

    Кегль: висота «тіла» рядка / x-height (0.52) для звичайного тексту або
    / cap-height (0.70), якщо переважають великі літери й цифри. Якщо ця оцінка
    розходиться з оцінкою за повною висотою «чорнила» (/0.85) більше ніж на 30 %
    (короткі рядки, розірваний профіль) — беремо повну висоту.
    Жирність: середня товщина штриха = 2 × площа / периметр гліфів, де площа
    рахується з частковим покриттям пікселів (антиаліасинг) — це дає точність
    до долі пікселя навіть для дрібного тексту 12–15 px.
    """
    rows = np.where(mask.sum(axis=1) > 0)[0]
    if len(rows) < 2:
        return 12.0, False
    span_est = (rows[-1] - rows[0] + 1) / 0.85
    chars = [c for c in text if c.isalnum()]
    tall = sum(c.isupper() or c.isdigit() for c in chars) / max(1, len(chars)) >= 0.5
    core_est = _core_band(mask) / (CAP_HEIGHT if tall else X_HEIGHT)
    font_px = core_est if abs(core_est - span_est) <= 0.3 * span_est else span_est
    v = fg.astype(np.float64) - bg.astype(np.float64)
    cov = np.clip(((patch.astype(np.float64) - bg) @ v) / (v @ v + 1e-9), 0, 1)
    cnts, _ = cv2.findContours((cov > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in cnts)
    stroke = 2 * float(cov.sum()) / max(perimeter, 1.0)
    return float(font_px), stroke / font_px >= BOLD_STROKE_RATIO


# ---------------------------------------------------------------- парсер

class UIParser:
    def __init__(self, device: str | None = None, use_omni=True, use_yolo_ui=True, use_ocr=True,
                 use_faces=True, omni_conf=0.05, yolo_ui_conf=0.25, yolo_ui_weights: str = YOLO_UI_WEIGHTS,
                 tiles: bool = True, plates: bool = True, modals: bool = True):
        self.device = device
        self.tiles, self.plates, self.modals = tiles, plates, modals
        self.omni_conf, self.yolo_ui_conf = omni_conf, yolo_ui_conf
        # окремі пороги класів (підібрані на валідації), якщо лежать поруч із вагами: models/<модель>/thresholds.json
        thr_path = os.path.join(os.path.dirname(yolo_ui_weights), "thresholds.json")
        self.class_conf = {}
        if os.path.exists(thr_path):
            import json
            with open(thr_path, encoding="utf-8") as f:
                self.class_conf = json.load(f).get("per_class", {})
        self.omni = self.yolo_ui = self.ocr = None
        self.available = {}
        if use_omni:
            self.omni = self._load_yolo(OMNI_WEIGHTS, "omniparser")
        if use_yolo_ui:
            self.yolo_ui = self._load_yolo(yolo_ui_weights, "yolo_ui")
        if use_ocr:
            self.ocr = self._load_ocr()
        self.faces = FaceDetector() if use_faces else None
        if self.faces:
            self.available["faces"] = f"OK ({self.faces.name})"

    # -- завантаження моделей (кожна може бути відсутня)

    def _load_yolo(self, path, key):
        if not os.path.exists(path):
            self.available[key] = "немає ваг"
            return None
        try:
            from ultralytics import YOLO, settings
            settings.update({"sync": False})  # без анонімної аналітики (google-analytics) і зайвих 4 с
            if self.device is None:
                import torch
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = YOLO(path)
            self.available[key] = f"OK ({self.device})"
            return model
        except Exception as e:  # noqa: BLE001
            self.available[key] = f"помилка: {e}"
            return None

    def _load_ocr(self):
        try:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            import paddle
            from paddleocr import PaddleOCR
            # GPU, якщо встановлено paddlepaddle-gpu і є CUDA-відеокарта; інакше CPU з oneDNN
            dev = "gpu:0" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() else "cpu"
            ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det",
                            text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
                            use_doc_orientation_classify=False, use_doc_unwarping=False,
                            use_textline_orientation=False, device=dev,
                            enable_mkldnn=dev == "cpu", cpu_threads=8)
            self.available["paddleocr"] = f"OK ({dev})"
            return ocr
        except Exception as e:  # noqa: BLE001
            self.available["paddleocr"] = f"помилка: {e}"
            return None

    # -- запуск окремих моделей

    def _run_omni(self, rgb):
        r = self.omni.predict(rgb[..., ::-1], conf=self.omni_conf, iou=0.7, imgsz=1280,
                              device=self.device, verbose=False)[0]
        return [(b, float(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]

    def _run_yolo_ui(self, rgb):
        """E5 на всьому екрані (960 px, як у навчанні) + на плитках, якщо екран ширший за TILE_MIN_W.

        Сайт 1920 px при 960 px стискається вдвічі, і дрібні елементи (іконки, чекбокси, поля) зникають.
        Плитки ~1000 px з перекриттям дають моделі рідну роздільність; великі елементи лишаються з
        повного кадру, з плиток беремо лише те, що не торкається внутрішнього шва.
        """
        low = min([self.yolo_ui_conf, *self.class_conf.values()])
        h, w = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        crops = [(0, 0, bgr)]
        if self.tiles and w >= TILE_MIN_W:
            n = int(np.ceil(w / TILE_W))
            tw = int(np.ceil(w / n)) + TILE_OVERLAP
            for i in range(n):
                x0 = min(max(0, i * (w // n) - TILE_OVERLAP // 2), w - tw)
                crops.append((x0, 0, bgr[:, x0:x0 + tw]))
        res = self.yolo_ui.predict([c for _, _, c in crops], conf=low, iou=0.6, imgsz=960,
                                   device=self.device, verbose=False)
        names = res[0].names
        boxes, cls, conf = [], [], []
        for (ox, oy, crop), r in zip(crops, res):
            cw = crop.shape[1]
            for b, k, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if ox or cw < w:  # плитка: відкидаємо рамки на внутрішньому шві
                    if (ox > 0 and b[0] < 4) or (ox + cw < w and b[2] > cw - 4):
                        continue
                boxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]); cls.append(int(k)); conf.append(c)
        if len(crops) > 1 and boxes:
            import torch
            from torchvision.ops import batched_nms
            keep = batched_nms(torch.tensor(boxes), torch.tensor(conf), torch.tensor(cls), 0.5).tolist()
            boxes, cls, conf = [boxes[i] for i in keep], [cls[i] for i in keep], [conf[i] for i in keep]
        return [(b, names[k], float(c)) for b, k, c in zip(boxes, cls, conf)
                if c >= self.class_conf.get(names[k], self.yolo_ui_conf)]

    def _run_ocr(self, rgb):
        res = self.ocr.predict(rgb[..., ::-1])[0]
        return [(b, t, float(s)) for b, t, s in zip(res["rec_boxes"].tolist(), res["rec_texts"],
                                                       res["rec_scores"]) if t.strip()]

    # -- головний метод

    def parse(self, rgb: np.ndarray) -> tuple[Layout, dict]:
        h, w = rgb.shape[:2]
        times = {}

        def timed(name, fn):
            t = time.perf_counter()
            out = fn()
            times[name] = round((time.perf_counter() - t) * 1000)
            return out

        # OCR (Paddle) іде в окремому потоці паралельно з YOLO (PyTorch): обидві бібліотеки відпускають GIL,
        # а вхід і моделі ті самі, тож результат не змінюється; на GitHub 1920×941 це −0,25 с (1,40 → 1,14 с)
        t_models = time.perf_counter()
        ocr_job = _pool().submit(timed, "paddleocr", lambda: self._run_ocr(rgb)) if self.ocr else None
        omni = timed("omniparser", lambda: self._run_omni(rgb)) if self.omni else []
        typed = timed("yolo_ui", lambda: self._run_yolo_ui(rgb)) if self.yolo_ui else []
        ocr_lines = ocr_job.result() if ocr_job else []
        times["models_parallel"] = round((time.perf_counter() - t_models) * 1000)

        t0 = time.perf_counter()
        bg_hex = rgb_to_hex(_dominant(rgb))
        self._cache: dict = {}  # аналіз кольорів кожного рядка OCR — один раз за виклик

        # 1) кандидати: типізовані боксы YOLO11-UI мають пріоритет над OmniParser
        containers = [{"label": n, "box": _xywh(b), "confidence": round(c, 3)} for b, n, c in typed
                      if n in YOLO_UI_CONTAINERS]
        # спливні вікна: E5 (клас Modal) часто їх не бачить на сайтах — доповнюємо правилом затемнення
        if self.modals:
            for mb, mc in _find_modals(rgb):
                if not any(c["label"] == "Modal" and _iou(mb, _xyxy(c["box"])) > 0.5 for c in containers):
                    containers.append({"label": "Modal", "box": _xywh(mb), "confidence": mc, "source": "rule"})
        cands = [{"box": b, "type": YOLO_UI_TYPE.get(n, "icon"), "label": n, "conf": c, "src": "yolo_ui"}
                 for b, n, c in typed if n not in YOLO_UI_CONTAINERS]
        for b, c in omni:
            if not any(_iou(b, k["box"]) > 0.5 for k in cands):
                cands.append({"box": b, "type": None, "label": "interactive", "conf": c, "src": "omniparser"})

        # 2) прив'язка рядків OCR до кандидатів (напис на кнопці, у полі вводу)
        for k in cands:
            k["lines"] = [i for i, (lb, _, _) in enumerate(ocr_lines) if _inside_frac(lb, k["box"]) > 0.6]
        used: set[int] = set()  # рядки, що стали написами кнопок/полів (а не вільним текстом)

        bar_top = _system_bar_top(rgb)  # системна панель Android: елементи над нею обрізаємо по її межі
        elements: list[Element] = []
        for n, k in enumerate(cands):
            x0, y0, x1, y1 = (int(round(v)) for v in k["box"])
            if (y0 + y1) / 2 < bar_top < y1:
                y1 = bar_top
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            patch = rgb[y0:y1, x0:x1]
            text = " ".join(ocr_lines[i][1] for i in k["lines"])
            fg, fill, _ = _two_colors(patch)
            ring = _ring(rgb, (x0, y0, x1, y1))
            fill_vs_ring = delta_e_rgb(fill, ring)
            area_frac = (x1 - x0) * (y1 - y0) / (w * h)
            aspect = (x1 - x0) / max(1, y1 - y0)
            color_std = float(patch.reshape(-1, 3).std(0).mean())

            typ = k["type"]
            if typ is None:  # OmniParser не дає класу — визначаємо за вмістом
                if text and (_has_frame(rgb, (x0, y0, x1, y1)) if self.plates else fill_vs_ring > FILL_JND):
                    typ = "button"  # напис на власній заливці або в рамці
                elif text:
                    typ = "text"    # посилання / пункт меню без заливки
                elif area_frac > 0.03 or (color_std > 40 and area_frac > 0.01):
                    typ = "image"
                else:
                    typ = "icon"
            if typ == "text" and k["src"] == "yolo_ui" and text and y1 - y0 < 0.06 * h:
                # Rico позначає «пігулки» на власній заливці як Text. Заливку перевіряємо замкненою плашкою:
                # на сайтах різниця «заливка/фон» часто хибна через градієнти й фото під текстом
                if self.plates:
                    lb = [ocr_lines[i][0] for i in k["lines"]]
                    plate = _find_plate(rgb, (min(b[0] for b in lb), min(b[1] for b in lb),
                                              max(b[2] for b in lb), max(b[3] for b in lb))) if lb else None
                    if plate is not None:
                        typ = "button"
                        x0, y0, x1, y1 = (int(v) for v in plate[0])
                        fill, aspect = plate[1], (x1 - x0) / max(1, y1 - y0)
                elif fill_vs_ring > FILL_JND:
                    typ = "button"
            elif k["label"] == "Text Button" and text and fill_vs_ring <= FILL_JND:
                typ = "text"  # «плоска» Text Button з Rico без заливки — це посилання, межі беремо з OCR
            if typ == "button" and self.plates and k["lines"] and k["label"] not in ("Text Button",):
                lb = [ocr_lines[i][0] for i in k["lines"]]
                tb = (min(b[0] for b in lb), min(b[1] for b in lb), max(b[2] for b in lb), max(b[3] for b in lb))
                if _has_frame(rgb, (x0, y0, x1, y1)) and (_looks_like_input(tb, (x0, y0, x1, y1)) or _looks_like_dropdown(
                        [ocr_lines[i][1] for i in sorted(k["lines"], key=lambda i: ocr_lines[i][0][0])],
                        (x0, y0, x1, y1))):
                    typ = "input"  # поле з підказкою біля лівого краю або випадний список
            if typ == "text" and k["lines"]:
                continue  # текст оформимо з рядків OCR (точніші межі й кегль)
            if typ == "image" and area_frac > BG_IMAGE_FRAC and k["lines"]:
                # фон-ілюстрація на пів екрана й більше: тексти поверх неї — окремі елементи, а не «підпис картинки»
                k["lines"], text = [], ""
            used.update(k["lines"])
            if (typ == "button" and aspect < 3.2 and y1 - y0 < 0.06 * h and text and (x1 - x0) < 0.45 * w
                    and not ACTION_WORDS.search(text)):
                typ = "chip"  # невелика кнопка-«пігулка» (фільтр, тег, варіант), а не дія (увійти, пошук)

            el = Element(id=f"{typ}_{n}", type=typ, box=_xywh((x0, y0, x1, y1)), role="secondary",
                         confidence=round(k["conf"], 3), source=k["src"], text=text)
            if typ in ("button", "chip", "input"):
                el.fill = rgb_to_hex(fill)
            elif typ in ("image", "icon"):
                el.fill = rgb_to_hex(fill)
                el.role = "decor"
            if text:
                fpx, bold = self._line_font(rgb, [ocr_lines[i][:2] for i in k["lines"]])
                el.font_size, el.bold = round(fpx, 1), bold
                el.color = rgb_to_hex(self._text_color(rgb, [ocr_lines[i][0] for i in k["lines"]]))
            else:
                el.color = rgb_to_hex(fg)
            elements.append(el)

        # 3) вільний текст: рядки OCR, що не належать кнопкам, групуємо в абзаци
        free = [i for i in range(len(ocr_lines)) if i not in used]
        for block in _group_lines([ocr_lines[i] for i in free]):
            boxes = [ocr_lines[free[j]][0] for j in block]
            text = " ".join(ocr_lines[free[j]][1] for j in block)
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            fpx, bold = self._line_font(rgb, [ocr_lines[free[j]][:2] for j in block])
            color = self._text_color(rgb, boxes)
            plate = _find_plate(rgb, (x0, y0, x1, y1)) if self.plates and len(block) == 1 and len(text) <= 40 else None
            if plate is not None and not any(e.type in ("button", "chip", "input") and
                                             _iou(plate[0], _xyxy(e.box)) > 0.5 for e in elements):
                (px0, py0, px1, py1), pcol = plate
                # поле вводу: широка світла плашка, текст притиснутий ліворуч (підказка/введене значення)
                typ = "input" if _looks_like_input((x0, y0, x1, y1), (px0, py0, px1, py1)) else "button"
                if (typ == "button" and (px1 - px0) / max(1, py1 - py0) < 3.2 and (px1 - px0) < 0.45 * w
                        and not ACTION_WORDS.search(text)):
                    typ = "chip"
                elements.append(Element(
                    id=f"{typ}_{len(elements)}", type=typ, box=_xywh((px0, py0, px1, py1)), role="secondary",
                    text=text, color=rgb_to_hex(color), fill=rgb_to_hex(pcol), font_size=round(fpx, 1), bold=bold,
                    confidence=round(float(np.mean([ocr_lines[free[j]][2] for j in block])), 3), source="ocr+plate"))
                continue
            bg_under = self._text_bg(rgb, boxes) if BG_FROM_INSIDE else _ring(rgb, (x0, y0, x1, y1), pad=3)
            elements.append(Element(
                id=f"text_{len(elements)}", type="text", box=_xywh((x0, y0, x1, y1)), role="secondary",
                text=text, color=rgb_to_hex(color), bg=rgb_to_hex(bg_under), font_size=round(fpx, 1),
                bold=bold, confidence=round(float(np.mean([ocr_lines[free[j]][2] for j in block])), 3),
                source="ocr"))
        texts = [e for e in elements if e.type == "text"]
        if texts:
            med = float(np.median([e.font_size for e in texts]))
            # пікселі скріншота -> CSS-пікселі: телефон у портреті має ширину ~390 CSS px
            # (скріншот 1080 px = щільність ×2,8), інакше поріг 22 px робить заголовком весь текст
            css = max(1.0, w / 390) if h > 1.3 * w else 1.0
            for e in texts:
                if e.box[1] + e.box[3] / 2 > bar_top:
                    e.type, e.role = "icon", "decor"  # «◁ ○ □» системної панелі OCR читає як літери
                elif PRICE.search(e.text) and len(e.text) <= 40:
                    e.type, e.role = "price", "primary"  # ціна — окремий бізнес-елемент
                elif (e.font_size / css >= 22 or e.font_size >= 1.5 * med
                      or (e.font_size >= 1.3 * med and e.bold)
                      # назва екрана у верхній панелі застосунку (Android/iOS): трохи більша за основний текст
                      or (css > 1 and e.box[1] + e.box[3] / 2 < 0.12 * h and e.font_size >= 1.15 * med)):
                    e.type = "heading"

        # 4) великі картинки, яких не знайшли детектори (фото без інтерактиву)
        if not any(e.type == "image" for e in elements):
            scale = min(1.0, 640 / w)  # фолбек на зменшеній копії: картинки великі, деталі не потрібні
            small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            fb, _ = opencv_segment(small)
            for e in fb.elements:
                e.box = [round(v / scale) for v in e.box]
                if e.type == "image" and not any(_iou(_xyxy(e.box), _xyxy(o.box)) > 0.3 for o in elements):
                    e.id, e.role, e.source, e.confidence = f"image_{len(elements)}", "decor", "opencv", 0.5
                    elements.append(e)

        # 5) обличчя — найсильніший «магніт» ранньої уваги (Bottom-Up)
        if self.faces is not None:
            t_f = time.perf_counter()
            text_boxes = [lb for lb, _, _ in ocr_lines]
            for (fx0, fy0, fx1, fy1), conf in self.faces.detect(rgb):
                if any(_inside_frac(tb, (fx0, fy0, fx1, fy1)) > 0.3 or _inside_frac((fx0, fy0, fx1, fy1), tb) > 0.3
                       for tb in text_boxes):
                    continue  # каскад Хаара плутає літери з обличчями
                elements.append(Element(id=f"face_{len(elements)}", type="face", role="decor",
                                        box=_xywh((fx0, fy0, fx1, fy1)), confidence=round(conf, 3),
                                        source=self.faces.name))
            times["faces"] = round((time.perf_counter() - t_f) * 1000)

        elements.sort(key=lambda e: (e.box[1], e.box[0]))
        _pick_cta(elements, rgb, h)
        times["fusion"] = round((time.perf_counter() - t0) * 1000) - times.get("faces", 0)  # без облич

        covered = np.zeros((h, w), bool)
        for e in elements:
            x, y, bw, bh = e.box
            covered[y:y + bh, x:x + bw] = True
        meta = {"whitespace_ratio": round(1.0 - float(covered.mean()), 3), "background": bg_hex,
                "n_elements": len(elements), "models": dict(self.available), "timings_ms": times,
                "n_ocr_lines": len(ocr_lines), "n_omniparser": len(omni), "n_yolo_ui": len(typed),
                "containers": containers}
        return Layout(w, h, bg_hex, elements), meta

    # -- допоміжні

    def _split_line(self, rgb, b):
        """(фрагмент, колір тексту, колір фону, маска гліфів) рядка — рахується один раз за аналіз."""
        key = tuple(int(round(v)) for v in b)
        if key not in self._cache:
            x0, y0, x1, y1 = key
            patch = rgb[max(0, y0):y1, max(0, x0):x1]
            self._cache[key] = (patch, *_two_colors(patch)) if patch.size else None
        return self._cache[key]

    def _line_font(self, rgb, lines):
        """lines: [(box, text), ...]. Медіана по рядках з вагою = кількість літер."""
        sizes, bolds, weights = [], [], []
        for b, text in lines:
            split = self._split_line(rgb, b)
            if split is None:
                continue
            patch, fg, bgc, mask = split
            s, bold = _font_metrics(patch, fg, bgc, mask, text)
            sizes.append(s)
            bolds.append(bold)
            weights.append(max(1, sum(c.isalnum() for c in text)))
        if not sizes:
            return 14.0, False
        order = np.argsort(sizes)
        cum = np.cumsum(np.array(weights)[order])
        median = sizes[order[np.searchsorted(cum, cum[-1] / 2)]]
        bold_w = sum(w for w, bd in zip(weights, bolds) if bd)
        return float(median), bold_w > sum(weights) / 2

    def _text_bg(self, rgb, boxes):
        """Фон під текстом: медіана пікселів між літерами всередині рядків (не гліфи).
        На фото й градієнтах фон під словом відрізняється від смуги навколо рамки, тож беремо саме його."""
        px = []
        for b in boxes:
            split = self._split_line(rgb, b)
            if split is not None:
                patch, fg, bgc, mask = split
                v = patch[~mask].reshape(-1, 3)
                if len(v):
                    px.append(v)
        if not px:
            return _ring(rgb, (min(b[0] for b in boxes), min(b[1] for b in boxes),
                               max(b[2] for b in boxes), max(b[3] for b in boxes)), pad=3)
        return np.median(np.concatenate(px), axis=0)

    def _text_color(self, rgb, boxes):
        cols = []
        for b in boxes:
            split = self._split_line(rgb, b)
            if split is not None:
                patch, fg, bgc, mask = split
                # беремо «найнасиченіші» пікселі гліфів, а не середнє з антиаліасингом
                px = patch[mask].astype(np.float64)
                if len(px) > 10:
                    d = np.abs(px - bgc).sum(1)
                    fg = px[d >= np.percentile(d, 70)].mean(0)
                cols.append(fg)
        return np.median(np.array(cols), axis=0) if cols else np.array([34, 34, 34])


def _dominant(rgb: np.ndarray) -> np.ndarray:
    """Найчастіший колір (квантований до 32 рівнів на канал) — через bincount, без сортування."""
    q = (rgb[..., :3] >> 3).astype(np.int32).reshape(-1, 3)
    code = np.bincount((q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2], minlength=1 << 15).argmax()
    return np.array([(code >> 10) & 31, (code >> 5) & 31, code & 31]) * 8 + 4


def _ring(rgb, box, pad: int = 6) -> np.ndarray:
    """Медіанний колір смуги навколо рамки — «фон під елементом»."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = rgb[Y0:Y1, X0:X1]  # працюємо лише з ділянкою навколо рамки, а не з усім екраном
    # смуга навколо рамки = ділянка мінус внутрішній прямокутник; беремо її чотирма шматками (верх, низ, ліво, право)
    # замість маски на всю ділянку — ті самі пікселі (порядок для медіани неважливий), без обходу середини рамки
    iy0, iy1 = max(0, y0) - Y0, max(0, y1 - Y0)
    ix0, ix1 = max(0, x0) - X0, max(0, x1 - X0)
    H, W = crop.shape[:2]
    iy0, iy1, ix0, ix1 = min(iy0, H), min(max(iy1, iy0), H), min(ix0, W), min(max(ix1, ix0), W)
    parts = [crop[:iy0].reshape(-1, crop.shape[2]), crop[iy1:].reshape(-1, crop.shape[2]),
             crop[iy0:iy1, :ix0].reshape(-1, crop.shape[2]), crop[iy0:iy1, ix1:].reshape(-1, crop.shape[2])]
    px = np.concatenate(parts)
    if not len(px):
        return _dominant(rgb)
    if px.dtype == np.uint8:
        return _median_u8(px)
    return np.median(px, axis=0)


def _median_u8(px: np.ndarray) -> np.ndarray:
    """np.median(px, axis=0) для цілих 0..255 — через гістограму каналу (bincount), той самий результат
    (для парної кількості — середнє двох середніх значень, як у numpy), але без сортування."""
    n = len(px)
    out = np.empty(px.shape[1], dtype=np.float64)
    for c in range(px.shape[1]):
        cum = np.cumsum(np.bincount(px[:, c], minlength=256))
        lo = int(np.searchsorted(cum, (n - 1) // 2, side="right"))  # значення з індексом (n−1)//2 у відсортованому
        hi = int(np.searchsorted(cum, n // 2, side="right"))        # значення з індексом n//2
        out[c] = (lo + hi) / 2
    return out


_POOL = None


def _pool():
    """Один фоновий потік для OCR (створюється при першому розборі)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(1, thread_name_prefix="ocr")
    return _POOL


def _xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def _find_modals(rgb: np.ndarray) -> list[tuple[list[int], float]]:
    """Спливне вікно: світла картка по центру, а сторінка навколо затемнена напівпрозорим шаром.

    Беремо зв'язні області «світлих» пікселів (L > 225), закриваємо дірки від тексту й кнопок,
    і лишаємо ту, що: займає 4–60 % екрана, не торкається країв, стоїть по центру по горизонталі,
    а смуга навколо неї помітно темніша (ΔL ≥ 35) — ознака затемнення.
    """
    h, w = rgb.shape[:2]
    s = min(1.0, 640 / w)
    small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    L = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255
    sh, sw = L.shape
    light = (L > 88).astype(np.uint8)
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(light, 4)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = bw * bh / (sw * sh)
        if not (0.04 <= frac <= 0.6) or area < 0.8 * bw * bh:
            continue
        if x < 0.03 * sw or y < 0.03 * sh or x + bw > 0.97 * sw or y + bh > 0.97 * sh:
            continue
        if abs(x + bw / 2 - sw / 2) > 0.12 * sw or bw > 0.85 * sw:
            continue
        pad = max(4, round(0.03 * sw))
        ring = np.ones((sh, sw), bool)
        ring[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad] = False
        near = np.zeros((sh, sw), bool)
        near[max(0, y - 3 * pad):y + bh + 3 * pad, max(0, x - 3 * pad):x + bw + 3 * pad] = True
        around = L[ring & near]
        inside = L[lab == i]
        if len(around) == 0 or inside.mean() - np.median(around) < 35:
            continue
        box = [round(x / s), round(y / s), round((x + bw) / s), round((y + bh) / s)]
        out.append((box, round(float(min(1.0, (inside.mean() - np.median(around)) / 60)), 3)))
    return out


def _has_frame(rgb: np.ndarray, box, min_sides: int = 3) -> bool:
    """Чи має рамка видиму межу хоча б з 3 боків: лінію на краю (±2 px) або інший колір одразу за краєм
    (власна заливка). Рядок списку без рамки (лише роздільник зверху) полем вводу не вважається."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return False
    inner = np.median(rgb[y0 + 4:y1 - 4, x0 + 4:x1 - 4].reshape(-1, 3), axis=0)
    bands = (lambda d: rgb[y0 + d, x0:x1] if 0 <= y0 + d < h else None,       # верх (d<0 — зовні)
             lambda d: rgb[y1 - 1 - d, x0:x1] if 0 <= y1 - 1 - d < h else None,  # низ
             lambda d: rgb[y0:y1, x0 + d] if 0 <= x0 + d < w else None,       # ліво
             lambda d: rgb[y0:y1, x1 - 1 - d] if 0 <= x1 - 1 - d < w else None)  # право
    sides = 0
    for band in bands:
        for d in (2, 1, 0, -1, -2, -4, -6):
            px = band(d)
            if px is not None and len(px) and np.mean(np.abs(px.astype(np.float64) - inner).sum(1) > 30) > 0.7:
                sides += 1
                break
    return sides >= min_sides


def _looks_like_input(text_box, box) -> bool:
    """Поле вводу: широка рамка (≥ 4:1), текст (підказка або значення) притиснутий до лівого краю
    й займає меншу частину ширини. Кнопка натомість має напис по центру."""
    x0, ty0, x1, ty1 = text_box
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    # поле має внутрішні відступи (рамка ≥ 1,4 висоти рядка); ліворуч може стояти іконка (лупа, ліжко).
    # Текст притиснутий ліворуч: лівий відступ менший за половину правого (у кнопки напис по центру).
    left, right = x0 - bx0, bx1 - x1
    return bw > 4 * bh and bh >= 1.4 * (ty1 - ty0) and left < 0.3 * bw and left < 0.5 * right


CHEVRON = {"v", "V", "˅", "⌄", "▾", "▼", "∨", "﹀"}


def _looks_like_dropdown(texts: list[str], box) -> bool:
    """Випадний список: широка рамка, останній «рядок» OCR — стрілка-галочка праворуч."""
    bx0, by0, bx1, by1 = box
    return bool(texts) and texts[-1].strip() in CHEVRON and len(texts) >= 2 and (bx1 - bx0) > 3 * (by1 - by0)


PLATE_STEP = int(os.environ.get("UX_PLATE_STEP", 4))


def _find_plate(rgb: np.ndarray, box) -> tuple[list[int], np.ndarray] | None:
    """Чи лежить рядок тексту на власній замкненій «плашці» (кнопка, поле вводу).

    Заливаємо (flood fill) колір фону одразу біля тексту. Якщо залита область замкнена —
    не доходить до країв вікна пошуку і не надто висока — це плашка; її рамка = рамка елемента.
    Відкритий фон сторінки або великої картки доходить до країв і плашкою не вважається.
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    th, tw = max(1, y1 - y0), max(1, x1 - x0)
    mx, my = int(max(4 * th, 1.5 * tw)), int(2.5 * th)
    X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    crop = rgb[Y0:Y1, X0:X1].copy()  # floodFill потребує записуваного масиву
    sx, sy = x0 - 3 - X0, (y0 + y1) // 2 - Y0  # зерно ліворуч від тексту, посередині рядка
    if sx < 0 or not (0 <= sy < crop.shape[0]):
        return None
    mask = np.zeros((crop.shape[0] + 2, crop.shape[1] + 2), np.uint8)
    # плаваючий діапазон (сусідні пікселі різняться ≤ PLATE_STEP): проходить плавні градієнти всередині
    # кнопки, але зупиняється на рамці та різкій межі заливки
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    d = (PLATE_STEP,) * 3
    cv2.floodFill(crop, mask, (sx, sy), 0, d, d, flags)
    m = mask[1:-1, 1:-1] > 0
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    px0, py0, px1, py1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if px0 == 0 or py0 == 0 or px1 >= crop.shape[1] or py1 >= crop.shape[0]:
        return None
    ph, pw = py1 - py0, px1 - px0
    if not (ph <= 3.5 * th and ph >= th and pw >= tw and m.sum() >= 0.5 * ph * pw):
        return None  # плашка має щільно обгортати рядок
    return [X0 + px0, Y0 + py0, X0 + px1, Y0 + py1], crop[m].mean(0)


# рядки OCR в одному ряду об'єднуємо, лише якщо проміжок < SAME_ROW_GAP × висота рядка:
# більший проміжок на сайтах означає окремі посилання меню (підібрано на валідації)
SAME_ROW_GAP = float(os.environ.get("UX_SAME_ROW_GAP", 0.3))

# фон тексту — з пікселів між літерами (True) або зі смуги навколо рамки (False)
BG_FROM_INSIDE = os.environ.get("UX_BG_INSIDE", "0") == "1"

PRICE = re.compile(r"\d[\d\s.,]*\s*(грн|₴|\$|€|£|uah|руб|₽|zł)|(\$|€|£|₴)\s*\d", re.IGNORECASE)


def _group_lines(lines):
    """Об'єднує рядки OCR в абзаци: близькі по вертикалі, вирівняні, схожої висоти.

    Ціна завжди окремий елемент (вона має власну вагу в увазі й конверсії).
    """
    order = sorted(range(len(lines)), key=lambda i: (lines[i][0][1], lines[i][0][0]))
    blocks: list[list[int]] = []
    for i in order:
        b, text = lines[i][0], lines[i][1]
        hb = b[3] - b[1]
        placed = False
        for blk in blocks if not PRICE.search(text) else []:
            if any(PRICE.search(lines[j][1]) for j in blk):
                continue
            last = lines[blk[-1]][0]
            hl = last[3] - last[1]
            gap = b[1] - last[3]
            short = len(text.strip()) <= 3  # «X», «→»: рамка нижча, бо немає виносних
            ratio_ok = 0.4 < hb / max(hl, 1) < 1.33 if short else 0.75 < hb / max(hl, 1) < 1.33
            same_row = abs(b[1] - last[1]) < 0.5 * hl and b[0] - last[2] < SAME_ROW_GAP * hl and b[0] > last[0]
            next_row = (-0.3 * hl <= gap < 0.6 * hl and abs(b[0] - lines[blk[0]][0][0]) < 1.5 * hl
                        and ratio_ok)
            if same_row or next_row:
                blk.append(i)
                placed = True
                break
        if not placed:
            blocks.append([i])
    return blocks


def _system_bar_top(rgb: np.ndarray) -> int:
    """Верх системної панелі навігації Android (суцільна темна смуга внизу портретного екрана) або h."""
    h, w = rgb.shape[:2]
    if h < 1.3 * w:
        return h
    rows = rgb[int(0.85 * h):].astype(np.float32).mean(axis=2)
    dark = (rows.mean(axis=1) < 30) & (np.percentile(rows, 80, axis=1) < 60)
    if not dark[-3:].all():
        return h
    top = len(dark)
    while top > 0 and dark[top - 1]:
        top -= 1
    bar = len(dark) - top
    return int(0.85 * h) + top if 0.03 * h <= bar <= 0.1 * h else h


def _pick_cta(elements: list[Element], rgb: np.ndarray, h: int):
    """Головна дія: слово-дієслово CTA + помітність + ширина + розташування нижче.
    Кнопки системної панелі Android («назад», «додому») — не дії інтерфейсу, їх пропускаємо."""
    best, best_score = None, 0.0
    bar_top = _system_bar_top(rgb)
    for e in elements:
        cy = e.box[1] + e.box[3] / 2
        if e.type not in ("button", "chip") or cy > bar_top or (bar_top < h and cy < 0.035 * h):
            continue  # системна панель знизу і рядок стану зверху (годинник, сповіщення)
        contrast = delta_e_rgb(np.array(hex_to_rgb(e.fill or "#FFFFFF")), _ring(rgb, _xyxy(e.box)))
        score = (1 + contrast) * e.box[2] * (0.3 + (e.box[1] + e.box[3] / 2) / h)
        if CTA_WORDS.search(e.text or ""):
            score *= 4
        if score > best_score:
            best, best_score = e, score
    if best:
        best.role = "cta"
        best.type = "button"
