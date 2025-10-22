# -*- coding: utf-8 -*-

# streamlit modülünü import et
import streamlit as st
import config as config

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.google_genai import (
    GoogleGenAIChatGenerator,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline


def get_prompt_builder():
    # prompt'u bu şekilde gönderiyoruz.
    template = """
    Sağlanan belgelere dayanarak soruyu yanıtlayın.
    Eğer belgeler soruyu yanıtlamak için yeterli bilgi içermiyorsa, 'Belgelerde bu konu hakkında yeterli bilgi bulamadım.' deyin.
    Yanıtınızı yalnızca sağlanan belgelere dayandırın ve kendi bilginizi eklemeyin.
    Yanıtınızın sonunda, kullandığınız belgelerin başlıklarını 'Kaynaklar:' başlığı altında listeleyin.

    Belgeler:
    {% for doc in documents %}
        Başlık: {{ doc.meta['report_title'] }}
        İçerik: {{ doc.content }}
    {% endfor %}

    Soru: {{question}}
    Yanıt:
    """
    return PromptBuilder(template=template)


@st.cache_resource
def build_rag_pipeline(_document_store):
    """
    RAG sistemini oluşturduğumuz yer burası. önce verilerimizi vektörize edip veritabanına kaydettik.
    """
    if not _document_store:
        return None

    try:
        # veritabanından dokümanları alıyoruz. en benzer 3 belgeyi döndürür.
        retriever = InMemoryEmbeddingRetriever(document_store=_document_store, top_k=3)

        # prompt'u oluştur.
        prompt_builder = get_prompt_builder()

        # gemini api key ve modeli kullanarak chat generator oluştur
        generator = GoogleGenAIChatGenerator(
            model="gemini-2.5-flash", api_key=config.GEMINI_API_KEY
        )

        # bu sefer de sorumuzu dil modeli ile sayılara çeviriyoruz.
        text_embedder = SentenceTransformersTextEmbedder(model=config.LANGUAGE_MODEL)

        # rag pipeline oluştur
        rag_pipeline = Pipeline()
        # sayılara çeviren embedder'i bağla.
        rag_pipeline.add_component("text_embedder", text_embedder)
        # veritabanından verileri alan retriever'i bağla.
        rag_pipeline.add_component("retriever", retriever)
        # prompt builder'i bağla.
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        # gemini burda sonuca göre bir cevap oluşturma kısmında yer alacak.
        rag_pipeline.add_component("generator", generator)

        # bileşenleri birbirine bağla.
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "generator.prompt")

        # oluşturulan RAG pipeline'i return et.
        return rag_pipeline
    except Exception as e:
        st.error(f"RAG boru hattı oluşturulurken hata oluştu: {e}")
        return None
