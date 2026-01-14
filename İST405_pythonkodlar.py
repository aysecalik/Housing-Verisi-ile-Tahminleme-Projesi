# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:38:46 2024

@author: HP
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("C:\\Users\\HP\\Desktop\\housing.csv")

# Veri incelemesi
data.info()

#EKSİK DEĞERLERİ KONTROL ETME
print("Eksik Veriler:", data.isnull().sum())


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
data['total_bedrooms'] = imputer.fit_transform(data[['total_bedrooms']])



#TANIMLAYICI İSTATİSTİKLER
data.describe()

# Eksik veriler için ileri düzey imputation (total_bedrooms)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Öncelikle sütunun değerlerini kontrol edelim
print(data["ocean_proximity"].value_counts())

# Kategorik değerleri sayısal değerlere dönüştürme
data["ocean_proximity"] = data["ocean_proximity"].replace({
    "<1H OCEAN": 1,
    "INLAND": 2,
    "NEAR OCEAN": 3,
    "NEAR BAY": 4,
    "ISLAND": 5
})

# Değişiklikten sonra sütunu kontrol edelim
print(data["ocean_proximity"].value_counts())

# İlk birkaç satırı kontrol etmek için:
print(data.head())

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Boxplot görselleştirmesi
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

############# kaggle kodlar
import matplotlib.pyplot as plt

# Histogramları oluşturuyoruz
data.hist(bins=50, figsize=(20, 15))
# Grafiği ekranda gösteriyoruz
plt.show()

####### pairplot(çiftli dağılım grafği)
def snsPairGrid(df):

    g = sns.PairGrid(df,diag_sharey=False)
    g.fig.set_size_inches(14,13)
    g.map_diag(sns.kdeplot, lw=2) 
    g.map_lower(sns.scatterplot,s=15,edgecolor="k",linewidth=1,alpha=0.4) 
    g.map_lower(sns.kdeplot,cmap='plasma',n_levels=10) 
    plt.tight_layout()

tlist = ['median_income','total_rooms','housing_median_age','latitude','median_house_value','population']
snsPairGrid(data[tlist]) 

# Korelasyon matrisini hesaplama
corr_matrix = data.corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="PRGn", cbar=True)
plt.title("Korelasyon Matrisi")
plt.show()

numeric_features = data.select_dtypes(include=['float64', 'int64'])

# Korelasyon matrisini hesaplayın
cor_matrix = numeric_features.corr()
print(cor_matrix)


#####harita

import geopandas as gpd
import geoplot as gplt
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPolygon, Polygon

# Veri Hazırlama
def prepare_data(data):
    # GeoDataFrame oluşturma
    gdf = gpd.GeoDataFrame(
        data, 
        geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
        crs="EPSG:4326"  # WGS 84 koordinat sistemi
    )
    return gdf

# MultiPolygon geometrilerini işle
def fix_multipolygons(gdf):
    gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom
    )
    return gdf

# Harita Çizme
def plot_california_map(data, value_col='median_house_value'):
    # Veri hazırlama
    gdf = prepare_data(data)

    # California sınırlarını yükle
    california = gpd.read_file(gplt.datasets.get_path('california_congressional_districts'))
    
    # MultiPolygon geometrilerini düzelt
    california = fix_multipolygons(california)

    # Haritayı çiz
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    gplt.polyplot(california, ax=ax, edgecolor='black', alpha=0.1)
    gplt.pointplot(
        gdf,
        ax=ax,
        hue=value_col,  # Haritada renk ile gösterilecek sütun
        cmap='viridis',  # Renk skalası
        legend=True,
        s=5,  # Nokta boyutu
        alpha=0.6  # Şeffaflık
    )

    # Harita başlıkları
    ax.set_title(f"California: {value_col} Dağılımı", fontsize=15)
    plt.show()

# Fonksiyon Çağrısı
plot_california_map(data)

############### ÖZELLİK SEÇME VE MODEL KURMA 

#######LASSO REGRESYON

# Veriyi hazırlayın
X = data.drop(columns=['median_house_value'])  # 'target' yerine hedef değişkenin ismini yazın
y = data['median_house_value']  # Hedef değişken

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso modelini kurma
lasso = Lasso(alpha=0.01)  # alpha, regularization parametresidir
lasso.fit(X_train, y_train)

# Lasso modelindeki katsayıları kontrol etme
coef = pd.Series(lasso.coef_, index=X.columns)
print("Özellikler ve Katsayıları:\n", coef)

# ÖZELLİK SEÇME
selected_features = coef[coef != 0].index
print("Seçilen Özellikler:", selected_features)

# Yeni veri seti ile model kurma
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Yeni model ile eğitim
lasso.fit(X_train_selected, y_train)

# Tahmin yapma
y_pred = lasso.predict(X_test_selected)
print(y_pred)

residuals=y_pred-y_test
print(residuals)

plt.figure(figsize=(10, 6))
coef_sorted = coef.sort_values(ascending=False)
sns.barplot(x=coef_sorted.values, y=coef_sorted.index, palette="viridis")
plt.title("LASSO Regresyon Modeli - Değişkenlerin Önemini Gösteren Grafik")
plt.xlabel("Katsayılar")
plt.ylabel("Özellikler")
plt.show()


# Performans metrikleri
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ortalama Kare Hatası (MSE): {mse}")
print(f"R2 Skoru: {r2}")

################ KORELASYON ÖZELLİK SEÇİMİ

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Korelasyon matrisini hesaplama
corr_matrix = X.corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='magma', fmt='.2f', linewidths=0.5)
plt.show()

# Korelasyon matrisinin üst üçgenini alıyoruz
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
print(upper)

plt.figure(figsize=(8, 6))
sns.heatmap(upper, annot=True, cmap='magma', mask=upper.isnull(), cbar=True, square=True)
plt.title("Üst Üçgen Korelasyon Matrisi")
plt.show()

# Yüksek korelasyona sahip özellikleri çıkarma (threshold=0.9)
threshold = 0.9
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print("Çıkarılacak Özellikler:", to_drop)

# Yüksek korelasyona sahip özellikleri veri kümesinden çıkarma
X_reduced = X.drop(columns=to_drop)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# 1. Random Forest modelini kurma
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#tahminler
y_pred = rf_model.predict(X_test)
print(y_pred)

# Kayıpları hesaplama (gerçek değerler - tahminler)
residuals = y_test - y_pred

# Kayıpların dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='#ff7f0e', bins=30)
plt.title('Random Forest Kayıpların Dağılımı')
plt.xlabel('Kayıplar (Gerçek - Tahmin)')
plt.ylabel('Frekans')
plt.show()


########### y_test y_pred uyum grafiği

y_test_sorted = np.sort(y_test)
y_pred_sorted = np.sort(y_pred)

cdf_test = np.arange(len(y_test_sorted)) / len(y_test_sorted)
cdf_pred = np.arange(len(y_pred_sorted)) / len(y_pred_sorted)

plt.figure(figsize=(8, 6))
plt.plot(y_test_sorted, cdf_test, label='Gerçek Değerler ', color='black')
plt.plot(y_pred_sorted, cdf_pred, label='Tahmin Değerler', color='red', linestyle='--')
plt.title('Random Forest Kümülatif Dağılım Fonksiyonu', fontsize=14)
plt.xlabel('Değerler', fontsize=12)
plt.ylabel('Kümülatif Dağılım', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Performans metrikleri (Random Forest)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# 2. Linear Regression modelini kurma
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

#tahminler
y_pred = lr_model.predict(X_test)
print(y_pred)

# Kayıpları hesaplama (gerçek değerler - tahminler)
residuals = y_test - y_pred

# Kayıpların dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue', bins=30)
plt.title('Linear Regresyon Kayıpların Dağılımı')
plt.xlabel('Kayıplar (Gerçek - Tahmin)')
plt.ylabel('Frekans')
plt.show()

########### y_test y_pred uyum grafiği
y_test_sorted = np.sort(y_test)
y_pred_sorted = np.sort(y_pred)

cdf_test = np.arange(len(y_test_sorted)) / len(y_test_sorted)
cdf_pred = np.arange(len(y_pred_sorted)) / len(y_pred_sorted)

plt.figure(figsize=(8, 6))
plt.plot(y_test_sorted, cdf_test, label='Gerçek Değerler ', color='black')
plt.plot(y_pred_sorted, cdf_pred, label='Tahmin Değerler', color='pink', linestyle='--')
plt.title('Linear Regresyon Kümülatif Dağılım Fonksiyonu', fontsize=14)
plt.xlabel('Değerler', fontsize=12)
plt.ylabel('Kümülatif Dağılım', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Performans metrikleri (Linear Regression)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 3. Gradient Boosting modelini kurma
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

#tahminler
y_pred = gb_model.predict(X_test)
print(y_pred)

# Kayıpları hesaplama (gerçek değerler - tahminler)
residuals = y_test - y_pred

# Kayıpların dağılımını görselleştirme
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='pink', bins=30)
plt.title('Gradient Boosting Kayıpların Dağılımı')
plt.xlabel('Kayıplar (Gerçek - Tahmin)')
plt.ylabel('Frekans')
plt.show()

########### y_test y_pred uyum grafiği
y_test_sorted = np.sort(y_test)
y_pred_sorted = np.sort(y_pred)

cdf_test = np.arange(len(y_test_sorted)) / len(y_test_sorted)
cdf_pred = np.arange(len(y_pred_sorted)) / len(y_pred_sorted)

plt.figure(figsize=(8, 6))
plt.plot(y_test_sorted, cdf_test, label='Gerçek Değerler ', color='black')
plt.plot(y_pred_sorted, cdf_pred, label='Tahmin Değerler', color='purple', linestyle='--')
plt.title('Gradient Boosting Kümülatif Dağılım Fonksiyonu', fontsize=14)
plt.xlabel('Değerler', fontsize=12)
plt.ylabel('Kümülatif Dağılım', fontsize=12)
plt.legend()
plt.grid()
plt.show()
# Performans metrikleri (Gradient Boosting)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Sonuçları yazdırma
print(f"Random Forest - MSE: {mse_rf}, R2: {r2_rf}")
print(f"Linear Regression - MSE: {mse_lr}, R2: {r2_lr}")
print(f"Gradient Boosting - MSE: {mse_gb}, R2: {r2_gb}")

############## RFE İLE ÖZELLİK SEÇİMİ
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# RFE ile en iyi 5 özelliği seçme
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)

# Seçilen özellikler
selected_features_rfe = X.columns[rfe.support_]
print("RFE ile Seçilen Özellikler:", selected_features_rfe)

X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)

# Random Forest modelini kurma
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_rf = rf_model.predict(X_test)
print(y_pred_rf)

# Performans metrikleri
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Sonuçları yazdırma
print(f"Random Forest - MSE: {mse_rf}, R2: {r2_rf}")

###### uyum grafiği
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=30, alpha=0.5, color='black', label='Gerçek Değerler')
plt.hist(y_pred_rf, bins=30, alpha=0.5, color='red', label='Tahmin Değerleri')
plt.title('Random Forest Modeli İçin Gerçek ve Tahmin Değerlerinin Histogramı', fontsize=14)
plt.xlabel('Değerler', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Linear Regression modelini kurma
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_lr = lr_model.predict(X_test)
print(y_pred_lr)

# Performans metrikleri
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

##### uyum grafiği
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=30, alpha=0.5, color='grey', label='Gerçek Değerler')
plt.hist(y_pred_lr, bins=30, alpha=0.5, color='pink', label='Tahmin Değerleri')
plt.title('Linear Regresyon Modeli İçin Gerçek ve Tahmin Değerlerinin Histogramı', fontsize=14)
plt.xlabel('Değerler', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Sonuçları yazdırma
print(f"Linear Regression - MSE: {mse_lr}, R2: {r2_lr}")

########## gradient boosting

# Gradient Boosting modelini kurma
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Tahmin yapma
y_pred_gb = gb_model.predict(X_test)
print(y_pred_gb)

# Performans metrikleri
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

######## uyum grafiği
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=30, alpha=0.5, color='purple', label='Gerçek Değerler')
plt.hist(y_pred_gb, bins=30, alpha=0.5, color='green', label='Tahmin Değerleri')
plt.title('Gradient Boosting Modeli İçin Gerçek ve Tahmin Değerlerinin Histogramı', fontsize=14)
plt.xlabel('Değerler', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Sonuçları yazdırma
print(f"Gradient Boosting - MSE: {mse_gb}, R2: {r2_gb}")


############## RİDGE REGRESYON

from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regresyon Modeli Kurma (L2 Regularizasyonu)
ridge_model = Ridge(alpha=1.0)  # alpha, regularizasyon parametresi
ridge_model.fit(X_train, y_train)

# Modelin katsayılarını alalım
coefficients = pd.Series(ridge_model.coef_, index=X.columns)

# Katsayıları sıralayalım
print("Ridge Model Katsayıları:")
print(coefficients)

# Katsayıları görselleştirelim
plt.figure(figsize=(10, 6))
coefficients.sort_values().plot(kind='barh', color='skyblue')
plt.title("Ridge Regresyon - Katsayılar")
plt.xlabel("Katsayı Değeri")
plt.ylabel("Özellikler")
plt.show()

# Model ile tahmin yapma
y_pred = ridge_model.predict(X_test)
print(y_pred)
# Kayıpları (Residuals) hesaplama
residuals = y_test - y_pred
print(residuals)

from scipy.stats import norm

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='grey', edgecolor='black', alpha=0.7)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, residuals.mean(), residuals.std())  # Normal dağılım çizgisi
plt.plot(x, p, 'k', linewidth=2)  # Çizgiyi ekle

# Başlık ve etiketler
plt.title("Ridge Regresyon - Kayıp (Residuals) Histogramı ve Dağılım Çizgisi")
plt.xlabel("Kayıp (Residuals)")
plt.ylabel("Frekans")
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=30, alpha=0.5, color='skyblue', label='Gerçek Değerler')
plt.hist(y_pred, bins=30, alpha=0.5, color='blue', label='Tahmin Değerleri')
plt.title('Ridge Regresyon Gerçek ve Tahmin Değerlerinin Histogramı', fontsize=14)
plt.xlabel('Değerler', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.legend()
plt.grid()
plt.show()
# Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ridge Regression - MSE: {mse}")
print(f"Ridge Regression - R2: {r2}")





