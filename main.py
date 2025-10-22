# -*- coding: utf-8 -*-

import streamlit as st
import data_processing as dp
import rag_pipeline as rp
import vector_store_manager as vsm


def get_pipeline():
    # veri setini indirip RAG sistemine uygun dokümanlara çevirdik.
    prepared_docs = dp.prepare_data()
    if not prepared_docs:
        return None

    # vektör veritabanı oluşturarak dil modelinin sayısal vektör karşılıklarını veritabanına yazdık.
    store = vsm.create_vector_store(prepared_docs)
    if not store:
        return None

    # rag pipeline'i oluşturduk ve aldık.
    pipeline = rp.build_rag_pipeline(store)
    if not pipeline:
        return None

    return pipeline


def generate_response(pipeline, prompt):
    # pipeline içerisine soruyu gönderiyoruz.
    result = pipeline.run(
        {
            "text_embedder": {"text": prompt},
            "prompt_builder": {"question": prompt},
        }
    )

    # dönen replies array içerisinden yanıt alıyoruz.
    if result and "generator" in result:
        replies = result["generator"].get("replies")
        if isinstance(replies, list) and replies:
            return replies[0]
        if isinstance(replies, str) and replies:
            return replies

    return "yanıt alınırken bir hata oluştu."


def load_messages():
    # streamlit içinde oturum geçmişi yoksa boş bir dict oluşturduk.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # eğer oturum geçmişi varsa mesajları yükle.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    # streamlit sayfa üst başlığı.
    st.set_page_config(page_title="Uçak tamir raporları asistanı", page_icon="✈️")
    # streamlit sayfa başlığı
    st.title("✈️ Uçak Tamir Raporları Asistanı")
    # streamlit sayfa açıklaması
    st.caption(
        "Envanterdeki uçaklar hakkında geçmiş hasar ve tamir kayıtlarına ulaşabilmek için sorular sorun."
    )

    pipeline = get_pipeline()

    # rag pipeline oluşturulamadıysa burda uygulamadan çıkıyoruz.
    if not pipeline:
        st.warning("asistan başlatılırken hata oluştu.")
        st.stop()

        return

    # oturumda önceden kalan mesajlar varsa yükle.
    load_messages()

    # Kullanıcıdan girdi al.
    prompt = st.chat_input(
        "Örnek: 2021 tarihinde hiç uçağın gyro sisteminde sorun yaşandı mı? Eğer yaşandıysa bu sorun hangi uçaklarda oldu ve nasıl bir tamir uygulandı?"
    )

    if prompt:
        # mesajı oturum geçmişine ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG boru hattını çalıştır ve yanıt al
        with st.spinner("raporlar inceleniyor ve yanıt alınıyor..."):
            try:
                response = generate_response(pipeline, prompt)
            except Exception as e:
                response = f"sorgu yapılırken bir hata oluştu: {e}"

        # Asistanın yanıtını sohbet geçmişine ekle ve göster
        st.session_state.messages.append({"role": "rag", "content": response})
        with st.chat_message("rag"):
            st.markdown(response)


if __name__ == "__main__":
    main()
