# -*- coding: utf-8 -*-

# streamlit modülünü import et
import streamlit as st
from datasets import load_dataset
import config as config

from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter


def generate_document(row):
    # rapor başlık ve içeriğini oluştur.
    content = f"{row['damage_type']}\n\n{row['report']}"
    # meta bilgilerini oluştur.
    meta = {
        "report_id": str(row["report_id"]),
        "damage_type": str(row["damage_type"]),
        "severity": str(row["severity"]),
        "report_title": str(row["aircraft_type"]) + " " + str(row["damage_type"]),
        "date": str(row["date"]),
        "report": str(row["report"]),
    }

    return Document(content=content, meta=meta)


def get_dataset():
    dataset = load_dataset(
        config.DATASET,
        # RAG yaptığımız için tüm veri setini train kısmı olarak al.
        split="train",
        # hugging face tokeni al.
        token=config.HUGGING_FACE_TOKEN,
    )
    # pandas dataframe oluştur
    df = dataset.to_pandas()

    # satır indexlerini sıfırla
    df.reset_index(drop=True, inplace=True)

    return df


# veri yükleme işleminin cache'lenmesi için cache_resource kullan.
@st.cache_resource
def prepare_data():
    """
    veri setini yükle, haystack nesneleri oluştur.
    """

    # stramlit spinner kullanarak yükleme ekranını göster. Bu kod bloğu çalıştığı sürece st.spinner'i göster.
    with st.spinner("Lütfen uçak tamir raporları veri seti yüklenirken biraz bekleyin"):
        # veri seti üykleme işlemini try içinde yaparak hataları yakala.
        try:
            # dataset yükleme işlemi.
            df = get_dataset()

            # haystack nesneleri oluştur.
            documents = [generate_document(row) for _, row in df.iterrows()]

            # oluşturduğumuz dokümanlar için splitter nesnesi oluştur. Bu nesne metinlerin nasıl ayrılacağını tanımlar.
            splitter = DocumentSplitter(
                # split by bölme kriteri
                split_by="word",
                # her parçada kaç kelime olacak
                split_length=200,
                # her iki parça arasında 30 kelimelik ortak alan bırak.
                split_overlap=30,
            )
            # splitter'i document verisi ile çalıştır ve documents listesini döndür.
            return splitter.run(documents)["documents"]
        except Exception as e:
            st.error(f"veri seti yüklenemedi: {e}")
            return None
