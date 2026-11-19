"""Перевірка сервера плагіна без Figma: надсилає макет examples/*.json і його картинку так, як це робить плагін.

Запуск (сервер уже має працювати):  .venv/Scripts/python figma_plugin/test_server.py [examples/product_bad.json]
Картинки шарів зберігаються в out/figma_test_<назва>/ для перегляду.
"""
import base64
import io
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image  # noqa: E402

from ux_eval.layout import Layout, render  # noqa: E402

URL = os.environ.get("UX_PLUGIN_URL", "http://127.0.0.1:8765")


def post(path, payload):
    req = urllib.request.Request(URL + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        return json.load(f)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/product_bad.json"
    d = json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
    buf = io.BytesIO()
    Image.fromarray(render(Layout.from_dict(d))).save(buf, "PNG")
    for _ in range(120):  # чекаємо, поки сервер завантажить моделі
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as f:
                h = json.load(f)
            if h["models_ready"] or h["error"]:
                break
        except OSError:
            pass
        time.sleep(2)
    print("health:", h)
    t = time.perf_counter()
    res = post("/analyze", {"layout": d, "png_b64": base64.b64encode(buf.getvalue()).decode()})
    print(f"аналіз: {res['elapsed_ms']} мс на сервері, {time.perf_counter() - t:.1f} с разом")
    print("оцінки:", res["scores"])
    print("проблем:", len(res["issues"]), "· з виправленням:", sum(1 for i in res["issues"] if i["fix"]))
    for i in res["issues"][:6]:
        print(f"  [{i['severity']}] {i['message']}\n      → {i['fix_text']}  {json.dumps(i['fix'], ensure_ascii=False)}")
    print("ефект:", res["effect"])
    out = os.path.join(ROOT, "out", "figma_test_" + os.path.splitext(os.path.basename(path))[0])
    os.makedirs(out, exist_ok=True)
    for n, o in enumerate(res["overlays"]):
        im = Image.open(io.BytesIO(base64.b64decode(o["jpeg_b64"])))
        assert im.size == (d["width"], d["height"]), (im.size, d["width"], d["height"])
        im.save(os.path.join(out, f"{n}.jpg"))
        print("  шар:", o["name"], im.size)
    adv = post("/advice", {"analysis_id": res["analysis_id"]})
    print("поради LLM:", adv.get("message") or adv.get("plan", "")[:300])


if __name__ == "__main__":
    main()
