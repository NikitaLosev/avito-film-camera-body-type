"""Загрузка объявления Авито по ссылке: заголовок + главное фото

Авито отдаёт реальную страницу (SSR), но данные не в JSON-LD/og-тегах, а в html:
заголовок - <h1>, фото - <img> на CDN img.avito.st. Мобильная версия (m.avito.ru)
чище для парсинга. Описание Авито в статике надёжно не отдаёт (ленивая подгрузка) -
для url-входа берём title+фото, этого хватает (в заголовке есть модель камеры).
Любая неудача (блок, нет фото, таймаут) -> None, и сервис отвечает input_unavailable
"""

import re
from io import BytesIO

import httpx
from PIL import Image

_USER_AGENT = (
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'
)
_HEADERS = {'User-Agent': _USER_AGENT, 'Accept-Language': 'ru-RU,ru;q=0.9', 'Referer': 'https://m.avito.ru/'}
_TIMEOUT = 12.0


def fetch_listing(url):
    """url объявления Авито -> (title, description, PIL.Image) либо None при любой неудаче"""
    try:
        mobile_url = re.sub(r'://(www\.)?avito\.ru', '://m.avito.ru', url, count=1)
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
            html = client.get(mobile_url).text
            title, image_url = _parse(html)
            content = client.get(image_url).content
        return title, '', Image.open(BytesIO(content)).convert('RGB')
    except Exception:
        return None


def _parse(html):
    """(title, image_url) из html мобильной страницы Авито; без фото - ValueError"""
    image_url = _main_photo(html)
    if not image_url:
        raise ValueError('не нашли фото объявления')
    return _title(html), image_url


def _title(html):
    """Заголовок из <h1>, фолбэк - alt главного фото"""
    h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.S)
    if h1:
        return re.sub(r'<[^>]+>', '', h1.group(1)).strip()
    alt = re.search(r'<img[^>]+alt="([^"]+)"', html)
    return alt.group(1).strip() if alt else ''


def _main_photo(html):
    """URL главного фото: первый <img> с CDN Авито img.avito.st"""
    match = re.search(r'<img[^>]+src="(https?://\d+\.img\.avito\.st/image/[^"]+)"', html)
    return match.group(1) if match else None
