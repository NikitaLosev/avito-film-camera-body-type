"""FastAPI прод-сервиса: страница (GET/POST /), JSON POST /predict, GET /health, /metrics

Страница - серверный рендер (Jinja2 + Bootstrap, как aaa-frontend-app). Модель и
энкодеры грузятся один раз при старте (lifespan). Метрики Prometheus на /metrics
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator

from app.avito import fetch_listing
from app.config import (
    DEPLOY_DIR,
    LABELS_RU,
    REASON_NO_PHOTO,
    REASON_URL_FAIL,
    REASONS_RU,
    REPO_ROOT,
)
from app.encoders import Encoders
from app.images import image_to_img_src, open_image
from app.metrics import model_loaded, predict_latency, record
from app.model import Predictor, abstain
from app.schemas import PredictResponse

# подхватываем HF_TOKEN (и пр.) из .env, чтобы не держать токен в переменных шелла
# в Docker .env нет - load_dotenv просто no-op
load_dotenv(REPO_ROOT / '.env')

_DESCRIPTION = (
    'Сервис определяет тип корпуса плёночной камеры по объявлению. На вход - заголовок, описание и '
    'фото камеры либо ссылка на объявление Авито. На выход - один из типов (зеркальная, '
    'двухобъективная, дальномерная, компактная, моментальная) или «не определено», если данных мало '
    'или на фото не камера'
)


@asynccontextmanager
async def lifespan(app):
    """Грузим бандл и энкодеры один раз на старте"""
    app.state.predictor = Predictor(encoders=Encoders())
    model_loaded.set(1)
    yield


app = FastAPI(title='Определение типа корпуса плёночной камеры', version='1.0.0',
              description=_DESCRIPTION, lifespan=lifespan)
Instrumentator().instrument(app).expose(app)
app.mount('/static', StaticFiles(directory=DEPLOY_DIR / 'static'), name='static')
templates = Jinja2Templates(directory=DEPLOY_DIR / 'templates')


def _run(title, description, url, photos):
    """Логика входа -> (PredictResponse, фото для превью или None)"""
    image = None
    if photos and photos[0].filename:
        image = open_image(photos[0].file)
    elif url:
        listing = fetch_listing(url)
        if listing is None:
            return abstain(REASON_URL_FAIL), None
        title, description, image = listing
    if image is None:
        return abstain(REASON_NO_PHOTO), None
    with predict_latency.labels(stage='inference').time():
        result = app.state.predictor.predict(title, description, image)
    return result, image


def _predict(title, description, url, photos):
    """То же, что _run, плюс учёт ответа в метриках"""
    result, image = _run(title, description, url, photos)
    record(result)
    return result, image


def _render(request, result, image):
    """Отрисовать страницу с опциональным результатом"""
    image_src = image_to_img_src(image) if image is not None else None
    context = {'result': result, 'image_src': image_src, 'labels_ru': LABELS_RU, 'reasons_ru': REASONS_RU}
    return templates.TemplateResponse(request, 'index.html', context)


@app.get('/health', tags=['Сервис'], summary='Проверить, что сервис работает')
def health():
    """Возвращает ok, если сервис запущен и отвечает"""
    return {'status': 'ok'}


@app.get('/', response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    """Страница с формой"""
    return _render(request, None, None)


@app.post('/', response_class=HTMLResponse, include_in_schema=False)
def index_predict(
    request: Request,
    title: str = Form(''),
    description: str = Form(''),
    url: str = Form(''),
    photos: list[UploadFile] = File(default=[]),
):
    """Сабмит формы -> та же страница с карточкой результата"""
    result, image = _predict(title, description, url, photos)
    return _render(request, result, image)


@app.post('/predict', response_model=PredictResponse, tags=['Предсказание'],
          summary='Определить тип корпуса камеры',
          response_description='Тип корпуса, уверенность и вероятности по классам')
def predict(
    title: str = Form('', description='Заголовок объявления'),
    description: str = Form('', description='Описание объявления'),
    url: str = Form('', description='Ссылка на объявление Авито - используется, если не загружено фото'),
    photos: list[UploadFile] = File(default=[], description='Фото камеры (используется первое)'),
):
    """Принимает заголовок, описание и фото камеры либо ссылку на объявление Авито

    Если переданы и фото, и ссылка - берётся загруженное фото. Без фото и без рабочей
    ссылки сервис не угадывает, а возвращает `other_unknown`
    """
    result, _ = _predict(title, description, url, photos)
    return result
