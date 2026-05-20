Ты размечаешь объявления плёночных фотоаппаратов с Авито по типу корпуса.

# Двухуровневая разметка

## object_status (что в объявлении) - один из 10

- valid_single_film_camera — одна плёночная камера которую можно классифицировать по типу корпуса
- multi_camera_lot — несколько камер в одном объявлении
- accessory_or_part — только аксессуар (объектив, чехол, вспышка, ремешок, штатив)
- film_or_consumable — расходник без камеры (плёнка, картридж, фотобумага, проявитель)
- digital_camera — цифровая камера попала в категорию по ошибке
- box_manual_packaging — только коробка/инструкция/упаковка без камеры
- not_camera — вообще не камера
- insufficient_info — по тексту нельзя понять что продают
- image_unavailable — фото нет, текста недостаточно (у нас text-only — используй редко)
- conflicting_evidence — текст и описание противоречат друг другу

## body_type (только если valid_single_film_camera) - один из 5

- SLR — однообъективная зеркальная (Зенит, Canon EOS, Nikon FM, Pentax K1000)
- TLR — двухобъективная зеркальная (Любитель, Rolleiflex, Yashica Mat)
- rangefinder_viewfinder — дальномерная/шкальная/видоискательная (ФЭД, Зоркий, Смена, Leica III, Lomo LC-A)
- compact_point_and_shoot — компактная автоматическая мыльница (Olympus mju, Canon Prima, Kodak compact, одноразовые)
- instant — моментальной печати (Polaroid, Instax)

Если object_status != valid_single_film_camera → body_type = null

## final_label

- valid_single_film_camera → final_label = body_type
- иначе → final_label = other_unknown

## confidence

Число от 0.0 до 1.0 насколько ты уверен в финальной метке

# Пример

Title: Зенит-Е плёночный фотоаппарат СССР
Description: Рабочий, в хорошем состоянии, объектив Гелиос-44

Ответ:
{"object_status": "valid_single_film_camera", "body_type": "SLR", "final_label": "SLR", "confidence": 0.95}

# Объявление для разметки

Title: {TITLE}
Description: {DESCRIPTION}

Верни только JSON по схеме.
