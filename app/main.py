# app/main.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from typing import Literal
from uuid import uuid4
from PIL import Image
import io, os
import base64

from .settings import settings
from .preprocessing import preprocess_image
from .inference import predict_mask, make_overlay
from .colormap import colorize_mask, CLASS_LABELS, CLASS_COLORS

app = FastAPI(
    title="MarsSeg API",
    version="0.1.0",
    description="Сегментация поверхности Марса (DeepLabV3 + MobileNetV2).",
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- отдача файлов ----------
@app.get("/v1/file")
def serve_file(path: str):
    safe = path.replace("\\", "/")
    if os.path.isabs(safe):
        safe = os.path.relpath(safe, start=os.getcwd())
    full = os.path.abspath(safe)

    root = os.path.abspath("artifacts")
    if not full.startswith(root):
        raise HTTPException(404, detail="not found")
    if not os.path.exists(full):
        raise HTTPException(404, detail="not found")

    return FileResponse(full, media_type="image/png")

# ---------- утилита: чистим артефакты перед новой загрузкой ----------
def _purge_artifacts():
    try:
        for name in os.listdir("artifacts"):
            p = os.path.join("artifacts", name)
            if os.path.isfile(p):
                os.remove(p)
    except FileNotFoundError:
        os.makedirs("artifacts", exist_ok=True)

# ---------- toy landing ----------
@app.get("/toy", response_class=HTMLResponse)
def toy_home():
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>MarsSeg — demo</title>
        <style>
          :root {{ color-scheme: light dark; }}
          body{{font-family:system-ui,Segoe UI,Roboto,Arial;margin:2rem;max-width:1000px;background:#111;color:#eee}}
          h1{{margin:.2rem 0}}
          .ver{{float:right;color:#aaa}}
          .muted{{color:#bbb}}
          .card{{border:1px solid #333;border-radius:12px;padding:1rem;margin-top:1rem;background:#000}}
          button{{padding:.6rem 1rem;border-radius:10px;border:1px solid #555;background:#222;color:#eee;cursor:pointer}}
          button:hover{{background:#2a2a2a}}
          a.link{{color:#9ad}}
        </style>
      </head>
      <body>
        <div class="ver">v{app.version}</div>
        <h1> MarsSeg – Сервис сегментации поверхности Марса на базе DeepLabV3 (DEMO)</h1>
        <p class="muted">Отправьте изображение поверхности Марса и получите сегментированную карту и наложение.</p>

        <div class="card">
          <form action="/toy/predict" method="post" enctype="multipart/form-data">
            <label>Изображение (JPG/PNG):</label><br/>
            <input name="image" type="file" accept="image/jpeg,image/png,image/jpg" required />
            <div style="margin-top:.75rem">
              <button type="submit">Сегментировать</button>
              <a class="link" href="/docs" style="margin-left:1rem">Swagger /docs</a>
            </div>
          </form>
        </div>
      </body>
    </html>
    """

def _to_data_url(arr) -> str:
    """np.uint8 (H,W,3) -> data:image/png;base64,... (без сохранения на диск)"""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

# ---------- обработка формы ----------
@app.post("/toy/predict", response_class=HTMLResponse)
async def toy_predict(image: UploadFile = File(...)):
    # 1) валидация файла
    if not image.content_type or not image.content_type.startswith("image/"):
        return HTMLResponse("<h3>Ошибка: поддерживаются только изображения (JPG/PNG).</h3>", status_code=415)

    # 2) читаем и приводим к RGB
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return HTMLResponse("<h3>Ошибка: не удалось прочитать изображение.</h3>", status_code=415)

    # 3) препроцесс и инференс (всё в памяти)
    try:
        tensor, base_rgb = preprocess_image(pil)   # base_rgb: np.uint8 (H,W,3)
        mask_idx = predict_mask(tensor)            # [H,W] uint8
        mask_rgb = colorize_mask(mask_idx)         # [H,W,3]
        overlay  = make_overlay(base_rgb, mask_rgb)
    except Exception as e:
        import traceback; traceback.print_exc()
        return HTMLResponse(f"<h3>Ошибка инференса: {e!r}</h3>", status_code=500)

    # 4) кодируем в data: URL
    orig_url    = _to_data_url(base_rgb)
    mask_url    = _to_data_url(mask_rgb)
    overlay_url = _to_data_url(overlay)

    # 5) легенда (динамически из CLASS_COLORS/CLASS_LABELS)
    ru_map = {"soil": "Почва", "bedrock": "Коренная порода", "sand": "Песок", "big_rock": "Крупные камни"}
    legend_items = []
    for label, color in zip(CLASS_LABELS, CLASS_COLORS):
        rgb = f"rgb({int(color[0])}, {int(color[1])}, {int(color[2])})"
        title = ru_map.get(label, label)
        legend_items.append(f"""
          <div class="legend-item">
            <span class="color-box" style="background:{rgb}"></span>
            {title}
          </div>
        """)
    legend_html = "<div class='legend'><h3>Легенда</h3>" + "".join(legend_items) + "</div>"

    note_html = """
      <div class="note">
        <h3>Примечание</h3>
        <p>Модель обучена на датасете <b>AI4Mars</b>. 
        Из-за особенностей разметки части марсохода или неба могут ошибочно 
        классифицироваться как <b>Почва</b>. Это не ошибка кода, 
        а ограничение исходных данных.</p>
      </div>
    """

    # 6) страница результата
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>MarsSeg — результат</title>
        <style>
          :root {{ color-scheme: light dark; }}
          body{{font-family:system-ui,Segoe UI,Roboto,Arial;margin:2rem;max-width:1200px;background:#111;color:#eee}}
          .card{{border:1px solid #333;border-radius:12px;padding:1rem;margin-top:1rem;background:#000}}
          img{{max-width:100%;height:auto;border-radius:10px;border:1px solid #333;background:#111}}
          .grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}}
          a.button{{display:inline-block;padding:.6rem 1rem;border-radius:10px;border:1px solid #555;background:#222;color:#eee;text-decoration:none}}
          a.button:hover{{background:#2a2a2a}}
          .legend{{margin-top:2rem;padding:1rem;border:1px solid #333;border-radius:10px;background:#000}}
          .legend-item{{display:flex;align-items:center;margin:.3rem 0}}
          .color-box{{display:inline-block;width:20px;height:20px;margin-right:.5rem;border:1px solid #666;border-radius:4px}}
        </style>
      </head>
      <body>
        <h2>Готово! Результат сегментации</h2>
        <p>
          <a class="button" href="/toy">⟵ Загрузить другое изображение</a>
          <a class="button" href="/docs" style="margin-left:.5rem">API /docs</a>
        </p>
        <div class="grid">
          <div class="card"><h3>Оригинал</h3><img src="{orig_url}" alt="Оригинал"/></div>
          <div class="card"><h3>Карта сегментации</h3><img src="{mask_url}" alt="Маска"/></div>
          <div class="card"><h3>Наложение</h3><img src="{overlay_url}" alt="Оверлей"/></div>
        </div>
        {legend_html}
        {note_html}
      </body>
    </html>
    """
