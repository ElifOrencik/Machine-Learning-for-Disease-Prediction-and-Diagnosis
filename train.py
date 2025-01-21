import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Eğitim verilerini yükle
train_data = pd.read_csv('veri_seti4.csv').drop(columns=['Unnamed: 133'], errors='ignore')
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']

# Modeli oluştur ve eğit
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Modeli kaydet
joblib.dump(model, 'model.joblib')
print("Model eğitildi ve model.joblib olarak kaydedildi.")

# Test verilerini yükle
test_data = pd.read_csv('testing2.csv').drop(columns=['Unnamed: 133'], errors='ignore')
X_test = test_data.drop('prognosis', axis=1)
y_test = test_data['prognosis']

# Model doğruluğu ve sınıflandırma raporunu hesapla
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Raporu dosyaya kaydet
with open('model_report.txt', 'w') as report_file:
    report_file.write(f"Doğruluk Oranı: {accuracy * 100:.2f}%\n")
    report_file.write("Sınıflandırma Raporu:\n")
    report_file.write(report)

print("Model doğruluk oranı ve sınıflandırma raporu model_report.txt dosyasına kaydedildi.")
