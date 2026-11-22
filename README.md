# UX Heatmap Pipeline

Прогнозування теплових карт уваги, колориметрії та UX-ергономіки макетів інтерфейсів.
За скриншотом або фреймом Figma система будує карти уваги для 1, 3 і 7 секунд, траєкторію погляду,
перевіряє контраст і кольори, оцінює візуальний шум та зручність натискання й складає план виправлень.

## Структура

| Шлях | Що робить |
|---|---|
| `ux_eval/ui_parser.py`, `segmentation.py` | крок 1 — розпізнавання елементів (YOLO11x + OmniParser + PaddleOCR; без GPU — OpenCV) |
| `ux_eval/color.py` | крок 2 — CIELAB, CIEDE2000, Oklab, APCA, симуляція порушень кольоросприйняття |
| `ux_eval/saliency_nn.py`, `saliency.py` | крок 3 — карти уваги 1 / 3 / 7 с (TempSAL + TranSalNet; без GPU — класичний метод) |
| `ux_eval/scanpath.py` | крок 4 — траєкторія погляду |
| `ux_eval/clutter.py` | крок 5 — візуальний шум (Feature Congestion, Subband Entropy) |
| `ux_eval/ergonomics.py` | крок 6 — закони Фіттса й Хіка, зона великого пальця |
| `ux_eval/evaluate.py` | кроки 1–7 разом і крок 7 — частка уваги кнопки дії |
| `ux_eval/auditor.py`, `cta_boost.py` | крок 8 — план виправлень і перевірка «було → стало» |
| `ux_eval/metrics.py` | метрики KL, CC, NSS, AUC-Judd |
| `figma_plugin/` | локальний сервер і плагін для Figma |
| `demo.py` | демонстрація всіх кроків на прикладах з `examples/` |

## Встановлення (Windows, NVIDIA GPU)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
.venv/Scripts/python -m pip install -r requirements.txt
```

Без GPU замість `paddlepaddle-gpu` встановіть `paddlepaddle==3.0.0`; пайплайн автоматично перейде на резервні
методи для кроків 1, 3 і 4.

## Ваги моделей

Ваги не зберігаються в репозиторії через розмір і ліцензії. Їх треба покласти в такі папки:

- `models/omniparser/icon_detect/` — OmniParser v2 (Microsoft, AGPL-3.0):
  ```bash
  .venv/Scripts/python -c "from huggingface_hub import hf_hub_download as d; [d('microsoft/OmniParser-v2.0', f, local_dir='models/omniparser') for f in ('icon_detect/model.pt','icon_detect/model.yaml','icon_detect/train_args.yaml')]"
  ```
- `models/yolo11x_rico25/best.pt` — YOLO11x, навчений на 25 класах Rico / WebUI (пороги класів уже в `thresholds.json`).
- `models/step3/` — код і ваги TempSAL і TranSalNet (`third_party/`, `weights/`, донавчені ваги в `weights/finetuned/`).
  Іншу папку можна задати змінною `UX_STEP3_DIR`.

Моделі PaddleOCR завантажуються автоматично при першому запуску. Якщо ваг немає, відповідний крок працює
резервним методом.

## Запуск

```bash
.venv/Scripts/python demo.py [скріншот_сайту.png]
```

Результати (панелі з усіма кроками, звіти, план дій) зберігаються в `out/`.

## Плагін для Figma

1. Запустіть сервер: `.venv/Scripts/python figma_plugin/server.py` (адреса `http://localhost:8765`).
2. У Figma desktop: Plugins → Development → Import plugin from manifest → `figma_plugin/plugin/manifest.json`.
3. Виділіть фрейм і натисніть «Аналізувати».

Аналіз виконується локально, дані нікуди не надсилаються.
