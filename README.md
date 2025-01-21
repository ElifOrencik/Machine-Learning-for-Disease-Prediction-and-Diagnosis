# Yapay zeka robot eklentisi

```{tip}
Projenize başlamadan önce python'da asenkron (async) programlama hakkında bilgi sahibi olmanız tavsiye edilir.
```

## Başlamadan önce

`AUTHOR.txt` dosyasına adınızı ve soyadınızı yazın.

`TITLE.txt` dosyasına projenizin adını yazın.


## Kurulum

Saçma hatalar ile karşılaşmamak için python 3.13 kullanmanızı tavsiye ederiz.

### Visual studio code ile kurulum

1. Komut paletini açın (`Ctrl+Shift+P` veya `F1`)
2. `Python: Create Environment` yazın (Türkçe kullanıyorsanız `Python: Ortam Oluştur` yazın)
3. `Venv` seçin
4. `Python 3.13.x 64-bit` seçin (x yerine herhangi bir sayı gelebilir)
5. `requirements.txt` kutucuğunu işaretleyin ve `OK` butonuna tıklayın
6. Paketlerin yüklenmesini bekleyin

`main.py` dosyasını sağ üstten çalıştırın

vscodenin içindeki terminalleri kullanacaksanız ve hali hazırda açık terminaliniz varsa kapatın ve yeniden açın

### Cmd ile kurulum

```{warning}
2. adımı her yeni cmd açtığınızda yapmanız gerekmektedir.
```

1. virtualenv oluşturun `python -m venv venv`
2. virtualenv'i aktif edin `venv\Scripts\activate`
3. gerekli kütüphaneleri yükleyin `pip install -r requirements.txt`
4. eklentinizi çalıştırın `python main.py`

## Dosya yapısı

- `main.py` Yapay zekanın çalışacağı ve robotun kontrol edileceği ana dosya (zorunlu)
- `train.py` Yapay zekayı eğittiğimiz dosya
- `core.py` Eklentinizin yapay zeka ile haberleşmesini sağlayan sınız (sizdeki ile robottaki dosya içerikleri farklı olacak)

## Kullanım

1. `train.py` dosyasına yapay zekanızı eğitip kaydetmek için gerekli kodları yazın.
2. `main.py` dosyasında yapay zekanızı yükleyin ve robotunuzu kontrol etmek için gerekli kodları yazın.
3. `main.py` dosyasını çalıştırın.
4. Eklentinizin aldığı ve gönderdiği verileri görün.

## Test süreci

Test süreci için size özel bir sınıf yazdık `main.py` dosyasını çalıştırarak sanki robot varmış gibi girdi ve çıktılar alabilirsiniz.

Örneğin

```
23:18:03.393 ❯❯ Sıcaklık: 25
23:18:03.394 ❯❯ Nem: 50
23:18:03.395 ❮❮ Motor: 107.84341637010681
```

Burada gördüğünüz gibi eğer robot sıcaklık olarak 25 ve nem olarak 50 verilerini alsaydı motor olarak 107.84 verisini gönderirdi.

Konsolda gördüğünüz bu satırlar robotta görünmeyecek ancak bunun yerine girdiler (❯❯) gerçek sensörlerden gelecek ve
çıktılar (❮❮) gerçek motorlara gidecek. Burada gerçek verileri ve sahte test verilerini gönrederen sınıf aynı olduğu için (core) kendi bilgisayarınızda robot ile aynı şekilde çalışacaktır.

## Paket kurulumu

Yeni paket kurarken (örneğin `pandas`) `requirements.txt` dosyasına paket adını yazın ve `pip install -r requirements.txt` komutunu çalıştırın.

Eğer yinede `pip install pandas` şekline kurmak istiyorsanız paket adını `requirements.txt` dosyasına eklemeyi unutmayın
