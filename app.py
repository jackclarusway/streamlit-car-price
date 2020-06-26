import streamlit as st
import pickle
import pandas as pd

st.title('Car Price Prediction by Jack')

html_temp = """
<div style="background-color:sandybrown;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML App by Jack</h2>
</div>"""

st.markdown(html_temp, unsafe_allow_html=True)

# model_rf = open("model_rf.pkl","rb")
# model = joblib.load(model_rf)

model = pickle.load(open('model_rf.pkl', 'rb'))

features = pickle.load(open('model_features.pkl', 'rb'))

hp = st.slider("What is the hp of your car", 60, 200, 80, 5)
age = st.selectbox("What is the age of your car", (1, 2, 3))
km = st.slider("What is the km of your car", 0, 100000, 5000, 100)
car_model = st.selectbox("Select model of your car", ('A1', 'A3', 'Astra', 'Clio', 'Corsa', 'Espace', 'Insignia'))

my_dict = {
    "hp": hp,
    "age": age,
    "km": km,
    "model": car_model
}

df = pd.DataFrame.from_dict([my_dict], orient='columns')

X = pd.get_dummies(df).reindex(columns=features, fill_value=0)

prediction = model.predict(X)

st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))
