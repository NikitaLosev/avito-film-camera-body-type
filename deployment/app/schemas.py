"""Pydantic-схема ответа сервиса (как в solution-design + вероятности)"""

from pydantic import BaseModel, Field

from app.config import ATTRIBUTE

_EXAMPLE = {
    'attribute': 'film_camera_body_type',
    'value': 'instant',
    'decision': 'auto_fill',
    'confidence': 0.993,
    'reason': 'model_argmax',
    'probabilities': {
        'SLR': 0.0, 'TLR': 0.0, 'rangefinder_viewfinder': 0.0,
        'compact_point_and_shoot': 0.0, 'instant': 0.993, 'other_unknown': 0.007,
    },
}


class PredictResponse(BaseModel):
    """Ответ: тип корпуса плёночной камеры по объявлению"""

    attribute: str = Field(ATTRIBUTE, description='Заполняемый атрибут товара')
    value: str = Field(description='Один из 5 типов корпуса либо other_unknown (отказ)')
    decision: str = Field(description='auto_fill (заполнить) или abstain (отказ)')
    confidence: float = Field(description='Уверенность 0..1 (максимальная вероятность)')
    reason: str = Field(description='Причина решения (model_argmax, photo_required, input_unavailable и т.п.)')
    probabilities: dict[str, float] = Field(default_factory=dict, description='Вероятности по всем 6 классам')

    model_config = {'json_schema_extra': {'example': _EXAMPLE}}
