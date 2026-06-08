"""Энкодеры Qwen3/DINOv3/PE-Core - паритет 1-в-1 с training/notebooks/extract_*.ipynb

Текст: (title+' '+desc) -> Qwen3 (L2). Фото: DINOv3 pooler_output (raw) и PE-Core
encode_image(normalize=False) (raw). L2 на эмбеддинги накладывает model.build_vector.
Три модели грузятся один раз при создании Encoders
"""

import open_clip
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

from app.config import DINO_MODEL, MAX_SEQ_LEN, PE_MODEL, QWEN_MODEL


class Encoders:
    """Три замороженных энкодера, загруженные один раз"""

    def __init__(self):
        self.text_model = SentenceTransformer(QWEN_MODEL)
        self.text_model.max_seq_length = MAX_SEQ_LEN
        self.dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self.dino_model = AutoModel.from_pretrained(DINO_MODEL).eval()
        self.pe_model, _, self.pe_preprocess = open_clip.create_model_and_transforms(PE_MODEL)
        self.pe_model = self.pe_model.eval()

    def encode_text(self, text):
        """Qwen3 текстовый эмбеддинг, L2-нормированный (1024)"""
        return self.text_model.encode([text], normalize_embeddings=True)[0]

    @torch.no_grad()
    def encode_dino(self, image):
        """DINOv3 pooler_output, сырой (384)"""
        pixels = self.dino_processor(images=image, return_tensors='pt')
        out = self.dino_model(**pixels)
        return out.pooler_output[0].numpy()

    @torch.no_grad()
    def encode_pe(self, image):
        """PE-Core image embedding, сырой (1024)"""
        pixels = self.pe_preprocess(image).unsqueeze(0)
        feat = self.pe_model.encode_image(pixels, normalize=False)
        return feat[0].numpy()

    def embed(self, text, image):
        """Текст + фото -> (qwen, dino, pe) для model.build_vector"""
        return self.encode_text(text), self.encode_dino(image), self.encode_pe(image)
