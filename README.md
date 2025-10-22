# Uçak Tamir Raporları Asistanı

Hangarda kim, hangi uçağın hangi arızayı ne zaman yaşadığını sorarsa sorsun, yanıtı bulmak saatler sürüyordu; PDF’ler, Excel dosyaları ve dağınık raporlar arasında kaybolan bilgiler uçuş planlamasını yavaşlatıyordu. Bu proje, o dağınık geçmiş kayıtları saniyeler içinde aranabilir hale getiren bir RAG (Retrieval-Augmented Generation) asistanı sunar. Streamlit arayüzü veri hazırlamadan vektör veritabanı kurulumuna ve yanıt üretimine kadar tüm adımları tek bir akışta birleştirir.

## Gereksinimler
- Python 3.10 veya üzeri
- `.env` dosyasında `GEMINI_API_KEY` ve `HUGGING_FACE_TOKEN` değerlerinin tanımlanması

## Çalıştırma
uv paket yöneticisinin global olarak kurulu olduğundan emin olun.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run main.py
```

## Verisetinin Hazırlanışı
Gerçek belgelere erişim olmadığı için python pandas kullanarak 100k satır içeren bir veri seti oluşturuldu.
Bu veri setinin oluşturulmasıyla ilgili veriler dataset_preparation/ klasörü içerisinde yer alıyor.

Uygulama açıldığında, örnek sorudan yararlanarak veya kendi sorunuzu yazarak Uçak Tamir Raporları Asistanı ile etkileşime geçebilirsiniz.

## Proje Yapısı
- `main.py`: Streamlit arayüzü ve sohbet akışını yönetir.
- `data_processing.py`: Veri setini indirir, işler ve Haystack dokümanlarını üretir.
- `vector_store_manager.py`: Doküman embedding'lerini oluşturup vektör mağazasını hazırlar.
- `rag_pipeline.py`: Retriever ve Gemini tabanlı generator bileşenleriyle RAG pipeline'ını kurar.
- `config.py`: Ortam değişkenleri ve model ayarlarını yönetir.
- `requirements.txt`: Gerekli Python bağımlılıklarını listeler.

## Örnek Sorular
- `2021 yılında gyro sisteminde arıza yaşayan uçak oldu mu?`
- `Seviyesi yüksek olarak sınıflanan raporlar hangi uçaklar için?`
- `Motor hasarı yaşayan uçaklar için hangi tamir işlemleri uygulanmış?`

## Elde Edilen Sonuçlar
- **Gyro sistemi arızası**  
  Soru: `2021 yılında gyro sisteminde arıza yaşayan uçak oldu mu?`  
  Yanıt senaryosu: Asistan, ilgili raporları filtreleyerek hangi uçaklarda tekrar eden gyro arızası bulunduğunu ve uygulanan tamir yöntemini özetler.
- **Kritik seviye hasar takibi**  
  Soru: `Seviyesi yüksek olarak sınıflanan raporlar hangi uçaklar için?`  
  Yanıt senaryosu: Filtrenin sonucunda yüksek öncelikli bakım gerektiren uçak listesi ve rapor tarihleri bir arada döner.
- **Tamir süreçlerini kıyaslama**  
  Soru: `Motor hasarı yaşayan uçaklar için hangi tamir işlemleri uygulanmış?`  
  Yanıt senaryosu: Motor arızası raporları karşılaştırılır, her bir uçak için onarım adımları ve süreleri özetlenir.
- **Parça tedarik planlaması**  
  Soru: `İniş takımı sorunlarında en sık değişen parçalar hangileri?`  
  Yanıt senaryosu: Belgelerden çekilen bilgiler aynı parçanın farklı raporlarda tekrarlandığını gösterir, tedarik listesine girdi sağlar.
