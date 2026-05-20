"""Общие настройки и хелперы для запросов в Gemini

Сюда вынесены вещи которые одинаковые в смоук-тесте, замере на holdout и
двух прод-прогонах: модель, ценники, retry с бэкоффом, обёртка над
generate_content с одинаковым конфигом
"""

import random
import time

from google.genai import errors, types

from .schema import LabelResponse

MODEL = 'gemini-3.1-flash-lite'

# тарифы Flex (50% от standard) для Gemini 3.1 Flash-Lite
PRICE_IN_FLEX = 0.125 / 1e6
PRICE_CACHED_FLEX = 0.0125 / 1e6
PRICE_OUT_FLEX = 0.75 / 1e6
PRICE_CACHE_WRITE = 0.25 / 1e6

# тарифы Standard - используются на калибровочных прогонах
PRICE_IN_STANDARD = 0.25 / 1e6
PRICE_CACHED_STANDARD = 0.025 / 1e6
PRICE_OUT_STANDARD = 1.50 / 1e6

# бэкофф для Flex: на пиках может стоять в очереди до 15 минут
BACKOFF_BASE_SEC = 10
BACKOFF_MULT = 3


def generate(client, cache_name, contents, service_tier=None, retries=5):
    """Вызов модели с retry и единым конфигом

    Возвращает GenerateContentResponse либо кидает APIError если все попытки кончились
    """
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    cached_content=cache_name,
                    response_mime_type='application/json',
                    response_schema=LabelResponse,
                    temperature=0,
                    max_output_tokens=200,
                    service_tier=service_tier,
                    http_options=types.HttpOptions(timeout=900_000),
                ),
            )
        except errors.APIError:
            if attempt == retries - 1:
                raise
            wait = BACKOFF_BASE_SEC * (BACKOFF_MULT ** attempt) * random.uniform(0.7, 1.3)
            time.sleep(wait)


def parse_response(raw: str) -> tuple[LabelResponse | None, str | None]:
    try:
        return LabelResponse.model_validate_json(raw), None
    except Exception as e:
        return None, str(e)[:200]


def split_prompt(template: str, marker: str = '# ОБЪЯВЛЕНИЕ ДЛЯ РАЗМЕТКИ') -> tuple[str, str]:
    """Промпт делится на статичную часть (которую кешируем) и шаблон с {TITLE}/{DESCRIPTION}"""
    if marker not in template:
        raise ValueError(f'в промпте нет маркера {marker}')
    static_part, dynamic_template = template.split(marker)
    return static_part.strip(), dynamic_template


def fill_prompt(dynamic_template: str, title: str | None, description: str | None) -> str:
    return (dynamic_template
            .replace('{TITLE}', title or '')
            .replace('{DESCRIPTION}', description or ''))


def image_part(image_id: int, img_dir) -> types.Part:
    p = img_dir / str(image_id % 1000) / f'{image_id}.jpg'
    return types.Part.from_bytes(data=p.read_bytes(), mime_type='image/jpeg')
