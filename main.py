import asyncio
import joblib
import pandas as pd
import speech_recognition as sr
from gtts import gTTS
import os
from core import Core

async def play_text_sound(core: Core, text: str, filename: str):
    """Verilen metni Türkçe sesli olarak okur (GTTS ile) ve çalar."""
    await core.set_state("Metin sesli olarak okunuyor...")

    # GTTS ile Türkçe sesli okuma
    tts = gTTS(text=text, lang='tr')
    tts.save(filename)  # Sesli dosyayı kaydet

    # Ses dosyasını core.play_sound ile çal
    if os.path.exists(filename):
        await core.play_sound(filename)  # play_sound fonksiyonunu kullan
    else:
        await core.set_state(f"Ses dosyası bulunamadı: {filename}")


async def play_welcome_message(core: Core):
    """Program başlangıcında hoş geldiniz mesajını çalar."""
    filename = "hosgeldiniz.mp3"
    if os.path.exists(filename):
        await core.play_sound(filename)
    else:
        await core.set_state(f"Ses dosyası bulunamadı: {filename}")


# Belirtileri sesli olarak alma fonksiyonu
async def get_belirtiler_from_audio(core: Core):
    """
    Kullanıcıdan sesli olarak belirtileri alır ve alınan belirtileri sesli olarak tekrar eder.
    """
    recognizer = sr.Recognizer()
    belirtiler = []
    with sr.Microphone() as source:
        await core.set_state("Lütfen belirtilerinizi sırayla söyleyin, 'tamam' diyerek bitirin...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                belirti = recognizer.recognize_google(audio, language="tr-TR")
                await core.set_state(f"Algılanan belirti: {belirti}")

                # 'tamam' komutunu aldığında sesli okuma yapma, direkt bir sonraki belirtileri al
                if belirti.lower() == "tamam":
                    break
                
                belirtiler.append(belirti.strip())
            except sr.WaitTimeoutError:
                break
            except sr.UnknownValueError:
                await core.set_state("Sesi anlayamadım, lütfen tekrar belirtin veya 'tamam' diyerek bitirin.")
            except sr.RequestError:
                await core.set_state("Hata oluştu.")
                break
    return belirtiler


async def main(core: Core):
    """
    Ana program akışı. Model yükleme, belirtileri alma ve tahmin etme süreçlerini içerir.
    """
    # Hoşgeldiniz mesajını çal
    await play_welcome_message(core)

    # Modeli yükle
    model = joblib.load("model.joblib")

    # Doğruluk oranını yükle ve ekrana yazdır
    try:
        with open('model_report.txt', 'r') as report_file:
            accuracy_content = report_file.readline()  # İlk satırı oku (Doğruluk Oranı)
            await core.set_state("Model Doğruluk Oranı:\n" + accuracy_content)
    except FileNotFoundError:
        await core.set_state("Model doğruluk oranı bilgisi bulunamadı.")

    while True:
        # Belirtileri sesli olarak al
        belirtiler = await get_belirtiler_from_audio(core)
        if belirtiler:
            # Modeldeki semptomlarla eşleşmeyen belirtileri kontrol et
            X_train_columns = model.feature_names_in_
            valid_belirtiler = [b for b in belirtiler if b.lower() in [semptom.lower() for semptom in X_train_columns]]

            if "kaşıntı" in valid_belirtiler and "cilt döküntüsü" in valid_belirtiler:
                # Sadece kaşıntı ve cilt döküntüsü varsa
                if len(valid_belirtiler) == 2:
                    filename = "mantarebelirtiler.mp3"
                    if os.path.exists(filename):
                        await core.play_sound(filename)  # Mantara özel belirtiler ses dosyasını çal
                    else:
                        await core.set_state(f"Ses dosyası bulunamadı: {filename}")
                else:
                    # Diğer belirtiler varsa, geçerli belirtileri sesli olarak okur
                    belirtiler_text = f"Algılanan belirtiler: {', '.join(valid_belirtiler)}"
                    await play_text_sound(core, belirtiler_text, "output_belirtiler.mp3")
            else:
                await core.set_state("Geçerli belirtiler tespit edilmedi.")

            # Modelin beklediği özelliklere göre belirtileri ayarla
            belirtiler_features = [1 if semptom.lower() in [b.lower() for b in valid_belirtiler] else 0 for semptom in X_train_columns]

            # Tahmin işlemi
            if sum(belirtiler_features) == 0:
                await core.set_state("Geçersiz belirtiler tespit edildi. Lütfen doğru belirtileri girin.")
                continue

            # Tahmin yap ve sonucu ekrana yazdır
            input_df = pd.DataFrame([belirtiler_features], columns=X_train_columns)
            result = model.predict(input_df)[0]  # Model ile tahmin yap
            await core.set_state(f"Tahmin Edilen Hastalık: {result}")

            # Kaşıntı ve cilt döküntüsü belirtileri varsa ve başka belirtiler yoksa "Mantar Enfeksiyonu" tahmin et
            if "kaşıntı" in valid_belirtiler and "cilt döküntüsü" in valid_belirtiler:
                if len(valid_belirtiler) == 2:  # Sadece kaşıntı ve cilt döküntüsü varsa
                    result = "Mantar Enfeksiyonu"
                else:
                    # Diğer belirtiler varsa, modelin tahminini kullan
                    result = model.predict(input_df)[0]
                    await core.set_state(f"Tahmin Edilen Hastalık: {result}")

            # Eğer hastalık "mantar enfeksiyonu" ise özel ses dosyasını çal
            filename = "mantarehastaliktahmini.mp3"
            if result.lower() == "mantar enfeksiyonu":
                if os.path.exists(filename):
                    await core.play_sound(filename)
                else:
                    await core.set_state(f"Ses dosyası bulunamadı: {filename}")
            else:
                # Sesli okuma: tahmin edilen hastalık
                disease_text = f"Tahmin edilen hastalık: {result}."
                await play_text_sound(core, disease_text, "output_hastalik.mp3")

            # İlaç ve bitkisel tedavi verilerini yükle
            treatment_df = pd.read_csv("disease_treatments.csv")

            # Tahmin edilen hastalık için tedavi seçeneklerini bul
            disease_row = treatment_df[treatment_df['Disease'].str.lower() == result.lower()]
            if not disease_row.empty:
                medical_treatment = disease_row['Medical_Treatment'].values[0]
                herbal_treatment = disease_row['Herbal_Treatment'].values[0]
                await core.set_state(f"İlaç Tedavisi: {medical_treatment}")
                await core.set_state(f"Bitkisel Tedavi: {herbal_treatment}")

                # Eğer tedavi "mantar enfeksiyonu" ise özel ses dosyasını çal
                filename = "mantaretedaviyon.mp3"
                if result.lower() == "mantar enfeksiyonu":
                    if os.path.exists(filename):
                        await core.play_sound(filename)
                    else:
                        await core.set_state(f"Ses dosyası bulunamadı: {filename}")
                else:
                    # Sesli okuma: tedavi bilgisi
                    treatment_text = f"İlaç tedavisi: {medical_treatment}. Bitkisel tedavi: {herbal_treatment}."
                    await play_text_sound(core, treatment_text, "output_tedavi.mp3")
            else:
                await core.set_state("Bu hastalık için tedavi bilgisi bulunamadı.")


if __name__ == "__main__":
    core = Core()
    asyncio.run(main(core))
