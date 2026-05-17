import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR


# Veri setini yükleme

df = pd.read_csv("train.csv")


# İlk 5 veri

print(df.head())


# Grafik oluşturma

plt.figure(figsize=(8,5))

plt.scatter(df["GrLivArea"], df["SalePrice"])

plt.xlabel("Metrekare")
plt.ylabel("Fiyat")

plt.title("Metrekare - Fiyat İlişkisi")

plt.show()


# Veri hazırlama

X = df[["GrLivArea", "BedroomAbvGr"]]

y = df["SalePrice"]


# Veriyi bölme

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Algoritmalar

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVR": SVR()
}


# Model karşılaştırması

for name, model in models.items():

    model.fit(X_train, y_train)

    tahmin = model.predict(X_test)

    hata = mean_absolute_error(y_test, tahmin)

    print(name)

    print("Ortalama Hata:", hata)

    print("----------------------")


# Final model

final_model = RandomForestRegressor()

final_model.fit(X_train, y_train)


# Yeni ev tahmini

yeni_ev = pd.DataFrame({
    "GrLivArea": [1500],
    "BedroomAbvGr": [3]
})


# Tahmin sonucu

sonuc = final_model.predict(yeni_ev)

print("Tahmini Ev Fiyatı:")

print(sonuc) 