# 🚀 MarsSeg API — Сегментация поверхности Марса

Демо-сервис для **семантической сегментации марсианской поверхности** на базе **DeepLabV3 (MobileNetV2)**, обученной на датасете **AI4Mars**.  
Сервис обёрнут во **FastAPI** и контейнеризован в **Docker** — запускается одной командой и работает в браузере.

> Загрузите изображение поверхности Марса — получите **маску** классов и **оверлей** поверх исходника.

---

## ✨ Возможности

- ⚡ REST API на **FastAPI** + **Swagger UI** (`/docs`).
- 🖼️ Toy-страница `/toy`: загрузка изображения → оригинал, маска, оверлей.
- 🧪 Препроцессинг: resize 256×256, **CLAHE**, яркость/контраст (alpha/beta).
- 🎨 Легенда классов: `Soil`, `Bedrock`, `Sand`, `Big Rock`.
- 🧹 Авто-очистка старых артефактов (новая загрузка — старые картинки удаляются).
- 🐳 **Docker** (CPU-сборка): быстрые повторные билды с кэшем.

---

## 🗂️ Структура проекта

```
marsseg-api/
│── app/
│   ├── main.py           # эндпоинты + toy-страница
│   ├── inference.py      # инференс (унифицированный для dict/тензора)
│   ├── preprocessing.py  # CLAHE, alpha/beta, resize/normalize
│   ├── loader.py         # загрузка модели и весов
│   ├── model_def.py      # DeepLabV3 + MobileNetV2 (кастом)
│   ├── colormap.py       # цвета классов, легенда
│   └── settings.py       # конфигурация (pydantic-settings, .env)
│── models/
│   └── .gitkeep          # положите сюда `deeplabv3.pth`
│── artifacts/            # результаты инференса (игнорируется git)
│── requirements.txt
│── Dockerfile
│── .env.example
│── README.md
```

---

## ⚠️ Веса модели

Репозиторий **не содержит** бинарные веса, чтобы не раздувать размер.  
Скачайте `deeplabv3.pth` по ссылке https://huggingface.co/1Lev/marsseg-deeplabv3/resolve/main/deeplabv3.pth и поместите в папку `models/`.

---

## 🔧 Конфигурация через `.env`

Скопируйте и при необходимости отредактируйте:
```bash
cp .env.example .env
```

Пример (синхронизирован с `settings.py`):
```env
MARSSEG_MODEL_NAME=deeplabv3
MARSSEG_MODEL_WEIGHTS=models/deeplabv3.pth
MARSSEG_NUM_CLASSES=4
MARSSEG_CLASS_LABELS=["soil","bedrock","sand","big_rock"]
MARSSEG_RESIZE_TO=256
MARSSEG_CLAHE_CLIP_LIMIT=2.0
MARSSEG_CLAHE_TILE_GRID=8
MARSSEG_ALPHA=1.5
MARSSEG_BETA=50
MARSSEG_USE_MEAN_STD=False
MARSSEG_OVERLAY_ALPHA=0.5
# MARSSEG_WEIGHTS_URL=  # опционально: URL для автоскачивания весов
```

---

## ▶️ Запуск локально (без Docker)

1) Установите зависимости:
```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

2) Убедитесь, что `models/deeplabv3.pth` лежит на месте.

3) Запустите сервер:
```bash
uvicorn app.main:app --reload --port 8000
```

Откройте:
- Toy-страница: http://127.0.0.1:8000/toy  
- Swagger UI: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health

---

## 🐳 Запуск в Docker (CPU)

1) Собрать образ:
```bash
docker build -t marsseg-api:0.1.0 .
```

2) Запустить контейнер (вариант A: веса внутри проекта):
```bash
docker run --rm -p 8000:8000 --env-file .env marsseg-api:0.1.0
```

или (вариант B: подключить папку с весами томом):
```bash
# Windows PowerShell:
docker run --rm -p 8000:8000 --env-file .env -v ${PWD}/models:/app/models marsseg-api:0.1.0
# Linux/Mac:
docker run --rm -p 8000:8000 --env-file .env -v $(pwd)/models:/app/models marsseg-api:0.1.0
```

Откройте: http://127.0.0.1:8000/toy

> Примечание: Dockerfile собирает **CPU-версию** (без CUDA).

---

## 🧪 Использование API

### Health
```
GET /health
→ {"status":"ok"}
```

### Загрузка изображения и инференс
```
POST /v1/predict?output=both
Content-Type: multipart/form-data
Form field: image=@<file>
```

**Пример (curl):**
```bash
curl -X POST "http://127.0.0.1:8000/v1/predict?output=both"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "image=@test.jpg"
```

**Ответ:**
```json
{
  "request_id": "ce0ab476",
  "outputs": [
    "artifacts/ce0ab476_mask.png",
    "artifacts/ce0ab476_overlay.png"
  ],
  "classes": ["soil","bedrock","sand","big_rock"]
}
```

Файлы можно открыть через:
```
GET /v1/file?path=artifacts/<имя_файла>.png
```

### Toy-страница
- Загрузка изображения прямо из браузера.
- Отображение: **Оригинал**, **Маска**, **Оверлей**.
- Ниже — **легенда** цветов классов.
- При каждой новой загрузке прошлые артефакты удаляются.

---

## 🎨 Легенда классов

| Класс      | Цвет    |
|------------|---------|
| Soil       | 🟥 красный |
| Bedrock    | 🟩 зелёный |
| Sand       | 🟨 жёлтый  |
| Big Rock   | 🟪 фиолетовый |

---

## 📌 Особенности

- Принимаются только **квадратные** изображения
- Модель может ошибочно помечать **части марсохода** или **небо** как `Soil` из-за ограничение разметки в наборе данных AI4Mars.
- На вход подаются RGB-изображения; внутри используется resize 256×256, CLAHE и коррекция контраста/яркости.

---

## 🛠️ Troubleshooting

- **`Model weights not found`** → проверьте, что `models/deeplabv3.pth` существует, или задайте `MARSSEG_WEIGHTS_URL`.
- **Пустые/странные предсказания** → убедитесь, что препроцессинг на инференсе согласован с обучением (CLAHE, alpha/beta, размер 256).
- **Docker билд долгий** → последующие сборки быстрые за счёт кэша; не меняйте `requirements.txt` без необходимости.
- **Windows + Docker** → при маппинге папок используйте `${PWD}` (PowerShell) или `%cd%` (cmd).

---

## 📚 Стек

**Python**, **FastAPI**, **Uvicorn**, **PyTorch**, **Torchvision**, **OpenCV (headless)**, **Pydantic v2**, **Docker**.

---

## 🔮 Дальнейшие планы

- CI/CD (GitHub Actions): билд образа, smoke-тест `/health`.
- Пакетный инференс и сохранение метрик.
- Экспорт модели в ONNX/TorchScript и профилирование.