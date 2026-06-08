# syntax=docker/dockerfile:1
# Прод-образ сервиса film_camera_body_type: FastAPI + 3 энкодера + logreg-бандл
# torch ставим CPU-колесом, веса энкодеров запекаем на build (HF_TOKEN как BuildKit-секрет),
# рантайм офлайн. Build context = корень репо (см .dockerignore)

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

# непривилегированный юзер (нужен для HF Spaces)
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# зависимости: сперва torch+torchvision CPU-колесом (без CUDA), потом остальное
COPY deployment/requirements.txt deployment/requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r deployment/requirements.txt

# код сервиса и бандл модели - только нужное в рантайме
COPY --chown=user:user deployment/app deployment/app
COPY --chown=user:user deployment/static deployment/static
COPY --chown=user:user deployment/templates deployment/templates
COPY --chown=user:user training/final_model training/final_model

USER user
WORKDIR /home/user/app/deployment

# запекаем веса 3 энкодеров в образ (DINOv3 gated -> токен через секрет), заодно
# проверяем загрузку бандла. Секрет монтируется только тут и в слои образа не попадает
RUN --mount=type=secret,id=hf_token,uid=1000 \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    python -c "from app.model import Predictor; from app.encoders import Encoders; Predictor(encoders=Encoders())"

# веса уже в образе -> рантайм не ходит в сеть
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
# exec-форма: SIGTERM от docker stop идёт прямо в uvicorn -> чистое завершение
# порт фиксированный 7860 (в HF Spaces задаётся через app_port)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
