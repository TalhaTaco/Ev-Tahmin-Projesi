import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Ev Fiyat Tahmini")

st.title("🏠 Ev Fiyat Tahmin Sistemi")

df = pd.read_csv("train.csv")

X = df[["GrLivArea", "BedroomAbvGr"]]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)

metrekare = st.number_input(
    "Metrekare (sqft)",
    min_value=500,
    max_value=5000,
    value=1500
)

oda = st.number_input(
    "Oda Sayısı",
    min_value=1,
    max_value=10,
    value=3
)

if st.button("Tahmin Et"):
    yeni_ev = pd.DataFrame({
        "GrLivArea": [metrekare],
        "BedroomAbvGr": [oda]
    })

    tahmin = model.predict(yeni_ev)

    st.success(
        f"Tahmini Ev Fiyatı: ${tahmin[0]:,.0f}"
    )