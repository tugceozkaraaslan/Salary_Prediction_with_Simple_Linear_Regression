import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1. Veri Yükleme / Oluşturma
df = pd.read_csv('C:\\Users\\ozkar\\OneDrive\\Desktop\\Salary_Prediction_with_Simple_Linear_Regression\\data\\Salary_Data.csv')

# 2. Veriyi Hazırlama
X = df[['YearsExperience']] # Bağımsız değişken (Matris formatında olmalı)
y = df['Salary']           # Bağımlı değişken

# Eğitim ve Test setine ayırma (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Eğitimi
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Tahmin Yapma
y_pred = model.predict(X_test)

# 5. Model Performansı Değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Performansı ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R2): {r2:.4f}")
print(f"Katsayı (Coefficient): {model.coef_[0]:.2f}")
print(f"Sabit Terim (Intercept): {model.intercept_:.2f}")

# 6. Görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Gerçek Veriler')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regresyon Doğrusu')
plt.title('Deneyim Yılına Göre Maaş Tahmini')
plt.xlabel('Deneyim Yılı')
plt.ylabel('Maaş')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 7. Manuel Tahmin Örneği
new_exp = [[5.5]]
predicted_salary = model.predict(new_exp)
print(f"\n5.5 yıllık deneyim için tahmin edilen maaş: {predicted_salary[0]:.2f}")