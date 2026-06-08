"""Хелперы для фото: открыть из файла и сериализовать в data-URI для страницы"""

from base64 import b64encode
from io import BytesIO

from PIL import Image


def open_image(file):
    """Открыть загруженное фото как RGB"""
    return Image.open(file).convert('RGB')


def image_to_img_src(image):
    """PIL Image -> data:image/png;base64,... для <img> на странице"""
    buffer = BytesIO()
    image.save(buffer, format='png')
    return f'data:image/png;base64,{b64encode(buffer.getvalue()).decode()}'
