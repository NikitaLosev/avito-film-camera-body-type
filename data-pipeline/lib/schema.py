"""Схема ответа Gemini и проверка бизнес-инварианта

Pydantic-модель уходит в SDK как response_schema, он сам делает из неё JSON schema
Бизнес-инвариант (если object_status = valid, body_type обязан быть, и наоборот)
проверяем отдельно - чтобы кривые ответы тоже сохранялись и попадали в метрики,
а не падали с исключением посреди прогона
"""

from typing import Literal

from pydantic import BaseModel, Field

ObjectStatus = Literal[
    'valid_single_film_camera', 'multi_camera_lot', 'accessory_or_part',
    'film_or_consumable', 'digital_camera', 'box_manual_packaging',
    'not_camera', 'insufficient_info', 'image_unavailable', 'conflicting_evidence',
]
BodyType = Literal[
    'SLR', 'TLR', 'rangefinder_viewfinder', 'compact_point_and_shoot', 'instant',
]
FinalLabel = Literal[
    'SLR', 'TLR', 'rangefinder_viewfinder', 'compact_point_and_shoot', 'instant',
    'other_unknown',
]


class LabelResponse(BaseModel):
    object_status: ObjectStatus
    body_type: BodyType | None
    final_label: FinalLabel
    confidence: float = Field(ge=0.0, le=1.0)


def is_business_valid(r: LabelResponse) -> tuple[bool, str | None]:
    if r.object_status == 'valid_single_film_camera':
        if r.body_type is None:
            return False, 'valid camera but body_type is None'
        if r.final_label != r.body_type:
            return False, 'final_label != body_type for valid camera'
    else:
        if r.body_type is not None:
            return False, f'non-valid status but body_type={r.body_type}'
        if r.final_label != 'other_unknown':
            return False, 'non-valid status but final_label != other_unknown'
    return True, None
