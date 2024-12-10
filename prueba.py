import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import streamlit as st


# Path del modelo preentrenado
MODEL_PATH = 'Stacking_model.pkl'


# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):

    x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)

    return preds


def main():
    
    model=''

    # Se carga el modelo
    if model=='':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">Predicción de Ingreso </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    #Datos = st.text_input("Ingrese los valores : N P K Temp Hum pH lluvia:")
    A1 = st.text_input("Años de Estudio:")
    H1 = st.text_input("Horas Semanales:")
    N1 = st.text_
    input("Número de Trabajadores:")
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"): 
        #x_in = list(np.float_((Datos.title().split('\t'))))
        x_in =[np.float_(A1.title()),
                    np.float_(H1.title()),
                    np.float_(N1.title()),
        ]
        predictS = model_prediction(x_in, model)
        st.success('EL INGRESO PREVISTO ES: {}'.format(predictS[0]).upper())

#if __name__ == '__main__':
#    main()