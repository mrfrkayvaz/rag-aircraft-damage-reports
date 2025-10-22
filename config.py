# -*- coding: utf-8 -*-

# os modülünü import et
import os
import streamlit as st

# .env bilgilerinin yüklenmesi için dotenv modülünü import et
from dotenv import load_dotenv

# ortam değişkenlerini yüklemeyi dene
try:
    # .env dosyasını yükle
    load_dotenv()
except Exception as e:
    # Hata durumunda hata mesajı ver
    st.error(f"Ortam değişkenleri yüklenemedi: {e}")
    st.stop()

# GEMINI_API_KEY değişkenini al
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# GEMINI_API_KEY değişkeni bos ise hata mesajı ver
if GEMINI_API_KEY is None:
    raise ValueError("❌ GEMINI_API_KEY bulunamadı. Lütfen .env dosyasını kontrol et.")

# HUGGING_FACE_TOKEN değişkenini al
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# HUGGING_FACE_TOKEN değişkeni bos ise hata mesajı ver
if HUGGING_FACE_TOKEN is None:
    raise ValueError(
        "❌ HUGGING_FACE_TOKEN bulunamadı. Lütfen .env dosyasını kontrol et."
    )

# oluşturduğum veri setinin hugging face linki
DATASET = "mrfrk/aircraft_reports"

# dil modeli

LANGUAGE_MODEL = "BAAI/bge-small-en-v1.5"

if __name__ == "__main__":
    pass
