'''Нормализация названий моделей фотоаппаратов для дедупа в kb.yaml и матчинга в kb_pass

Делает (по порядку):
- lowercase
- ё→е, э→е (опечатки в русских моделях вроде ФЭД vs ФЕД)
- per-token конвертация букв-омографов между латиницей и кириллицей
  (a c e o p x y k b m t n  <-> а с е о р х у к в м т н):
    * токен с 2+ буквами: переводит свои омографы в дом. скрипт самого токена
      (так "olympus" остаётся Latin даже когда вокруг русские слова,
       а "мju" с опечаткой превращается в чистый "mju")
    * одно-буквенный токен: переводит сам себя по дом. скрипту остальной строки
      (так "B" в "Зенит B" становится "в", чтобы совпасть с "Зенит В")
- всё кроме букв и цифр заменяется на пробел
- буквы и цифры разделяются пробелом (mju2 -> mju 2)
- римские I/II/III как отдельные слова -> 1/2/3
- схлопнуть пробелы и strip

Идемпотентно: normalize(normalize(x)) == normalize(x)
Используется и build_kb.py при сборке справочника, и kb_pass.py при матчинге заголовков
'''

import re

LAT_TO_CYR = str.maketrans('aceopxykbmtn', 'асеорхуквмтн')
CYR_TO_LAT = str.maketrans('асеорхуквмтн', 'aceopxykbmtn')


def _count(text):
    cyr = sum(1 for c in text if 'а' <= c <= 'я')
    lat = sum(1 for c in text if 'a' <= c <= 'z')
    return cyr, lat


def normalize(name):
    s = name.lower()
    s = s.replace('ё', 'е').replace('э', 'е')
    s = re.sub(r'[^a-zа-я0-9]+', ' ', s)

    tokens = s.split()
    total_cyr, total_lat = _count(s)

    for i, t in enumerate(tokens):
        t_cyr, t_lat = _count(t)
        if t_cyr + t_lat >= 2:
            if t_cyr > t_lat:
                tokens[i] = t.translate(LAT_TO_CYR)
            elif t_lat > t_cyr:
                tokens[i] = t.translate(CYR_TO_LAT)
        elif t_cyr + t_lat == 1:
            rest_cyr = total_cyr - t_cyr
            rest_lat = total_lat - t_lat
            if rest_cyr > rest_lat:
                tokens[i] = t.translate(LAT_TO_CYR)
            elif rest_lat > rest_cyr:
                tokens[i] = t.translate(CYR_TO_LAT)

    s = ' '.join(tokens)
    s = re.sub(r'([a-zа-я])(\d)', r'\1 \2', s)
    s = re.sub(r'(\d)([a-zа-я])', r'\1 \2', s)

    s = re.sub(r'\biii\b', '3', s)
    s = re.sub(r'\bii\b', '2', s)
    s = re.sub(r'\bi\b', '1', s)

    return ' '.join(s.split())
