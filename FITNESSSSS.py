import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción de la forma de una persona ''')
st.image("forma.jpg", caption="Un sujeto en forma")

st.header('Datos de la persona')

def user_input_features():
  # Entrada
  age = st.number_input('Edad:', min_value=1, max_value=3, value = 1, step = 1)
  height_cm = st.number_input('Altura (cm):', min_value=0, max_value=1, value = 0, step = 1)
  weight_kg = st.number_input('Peso (kg):', min_value=0, max_value=100, value = 0, step = 1)
  heart_rate = st.number_input('Frecuencia cardiaca:',min_value=0, max_value=10, value = 0, step = 1)
  blood_pressure = st.number_input('Presión arterial:', min_value=0, max_value=10, value = 0, step = 1)
  sleep_hours = st.number_input('Horas de sueño:', min_value=0, max_value=2, value = 0, step = 1)
  nutrition_quality = st.number_input('Calidad de nutrición:', min_value=1, max_value=3, value = 1, step = 1)
  activity_index = st.number_input('Índice de actividad:', min_value=0, max_value=1, value = 0, step = 1)
  smokes = st.number_input('Fuma (1=si, 0=no):', min_value=0, max_value=100, value = 0, step = 1)
  gender = st.number_input('Género (F=0, M=1):',min_value=0, max_value=10, value = 0, step = 1)
 
  user_input_data = {'Pclass': Pclass,
    'age':age,
    'height_cm':height_cm,
    'weight_kg' :weight_kg,
    'heart_rate' :heart_rate,
    'blood_pressure' :blood_pressure,
    'sleep_hours' : sleep_hours,
    'nutrition_quality' :nutrition_quality,
    'activity_index' :activity_index,
    'smokes':smokes,
    'gender':gender}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

titanic =  pd.read_csv('FIT2.csv', encoding='latin-1')
X = titanic.drop(columns='is_fit')
Y = titanic['is_fit']

classifier = DecisionTreeClassifier(max_depth=5, criterion='gini', min_samples_leaf=25, max_features=6, random_state=1615160)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No es fit')
elif prediction == 1:
  st.write('Sói es fit')
else:
  st.write('Sin predicción')
