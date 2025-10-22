# -*- coding: utf-8 -*-

# streamlit modülünü import et
import streamlit as st
import config as config

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack.components.writers import DocumentWriter


# yine cache mekanizması kullanıyoruz.
@st.cache_resource
def create_vector_store(_split_docs):
    """
    RAG sistemine uygun hale getirilen dokümanları alıp embedding üreterek
    FAISS vektör veritabanına yaz.
    """
    # dokümanlar oluşturulamadıysa None return ediyoruz.
    if not _split_docs:
        return None

    # streamlit spinner kullanarak yükleme ekranını göster
    with st.spinner("vektör veritabanı oluşturuluyor..."):
        # try ile hata durumunda hata mesajı ver
        try:
            # dokümanları RAM'den kullanabilmek için vektör veritabanına kaydet.
            document_store = InMemoryDocumentStore()

            # metinleri sayılarla ifade edilen vektörlere çevir.
            # burda basit bir türkçe dil destekli model kullanıyoruz.
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model=config.LANGUAGE_MODEL,
                # aynı anda işlenen örnek sayısı.
                batch_size=32,
            )

            # verileri depoya yazabilmek için pipeline oluştur.
            indexing_pipeline = Pipeline()
            # embedder'i ekliyoruz. embedder metni sayıya çevirir.
            indexing_pipeline.add_component("embedder", doc_embedder)
            # dokümanları depoya yazacak writer'ı ekliyoruz. writer sayıları veritabanına yazar.
            indexing_pipeline.add_component(
                "writer", DocumentWriter(document_store=document_store)
            )
            # embedder.documents ile writer.documents arasındaki bağlantıyı kuruyoruz.
            indexing_pipeline.connect("embedder.documents", "writer.documents")

            # pipeline'i çalıştırarak dokümanları depoya yaz.
            indexing_pipeline.run({"embedder": {"documents": _split_docs}})

            # oluşturduğumuz bu veritabanını return ediyoruz.
            return document_store
        except Exception as e:
            st.error(f"Vektör indeksi oluşturulurken hata oluştu: {e}")
            return None
