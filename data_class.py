import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Cargar el archivo CSV y especificar que los valores vacíos se interpreten como NaN
data = pd.read_csv('Encuesta Permanente de Empleo Nacional.csv', sep=",", encoding='latin1', na_values=["", " "])

# Filtrar filas sin valores faltantes
data = data.dropna()

# Mostrar el resultado
data
# Función para eliminar outliers utilizando el rango intercuartílico
def remove_outliers_iqr(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filtra los datos dentro de los límites
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Columnas a analizar para detectar outliers
columns = ['IngresoMensual', 'HorasSemanales', 'NumTrabajadores', 'AñosEstudio']

# Aplicar la función para eliminar outliers
data = remove_outliers_iqr(data, columns)

#Entrenamiento del modelo
modelo = smf.ols('IngresoMensual ~ AñosEstudio + HorasSemanales + NumTrabajadores', data = data).fit()

nombres = ['estadística del multiplicador de Lagrange', 'valor p','valor f', 'valor p f']
test = sms.het_breuschpagan(modelo.resid, modelo.model.exog)
[nombres[1],test[1]]

#Coeficientes del modelo
modelo.params

#Error del modelo
#A menor valor, el modelo es más adecuado
modelo.mse_resid

#Intervalos de confianza para los coeficientes
intervalos_ci = modelo.conf_int(alpha=0.05)
intervalos_ci.columns = ['2.5%', '97.5%']
intervalos_ci

from sklearn.model_selection import train_test_split

# Dividir data en data_test (70%) y data_train (30%)
data_train, data_test = train_test_split(data, test_size=0.7, random_state=42)

data_test_Ing_pred = modelo.predict(exog = data_test)
data_test_Ing_pred

data_test_f = pd.concat([pd.DataFrame(data_test),pd.DataFrame(data_test_Ing_pred)],axis=1)
data_test_f.columns = ['IngresoMensual','AñosEstudio','HorasSemanales','NumTrabajadores','Ing_Pred']

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

#MSE
mse1 = mean_squared_error(y_true = data_test_f.IngresoMensual, y_pred = data_test_f.Ing_Pred)

#RMSE
mse1 = mean_squared_error(y_true = data_test_f.IngresoMensual, y_pred = data_test_f.Ing_Pred)
rmse1=np.sqrt(mse1)

#MAE
mae1 = mean_absolute_error(y_true = data_test_f.IngresoMensual, y_pred = data_test_f.Ing_Pred) 

#MAPE
mape1 = 100*mean_absolute_percentage_error(y_true = data_test_f.IngresoMensual, y_pred = data_test_f.Ing_Pred) 

#R2
r2 = r2_score(y_true = data_test_f.IngresoMensual, y_pred = data_test_f.Ing_Pred)

#R2 ajustado
n = len(data_test) # Cantidad de Filas del test
k = data_test.shape[1] - 1 #Cantidad de Columnas (variables independientes)
r2_adj = 1-((1-r2)*(n-1)/(n-k-1))

# Preparación de los datos

X_train = data_train[['AñosEstudio','HorasSemanales','NumTrabajadores']]
y_train = data_train['IngresoMensual']

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

X_test = data_test[['AñosEstudio','HorasSemanales','NumTrabajadores']]
y_test = data_test['IngresoMensual']

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

# Modelamiento

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

regresor_mlp = MLPRegressor(activation='relu',
                            alpha=0.0001,
                            hidden_layer_sizes=(50,50,50),
                            learning_rate='adaptive',
                            max_iter=500,
                            solver='adam').fit(X_train, y_train)

y_pred_mlp = regresor_mlp.predict(X_test)
mse1 = mean_squared_error(y_true=y_test, y_pred=y_pred_mlp)
rmse1 =np.sqrt(mse1)
mae1 = mean_absolute_error(y_true=y_test, y_pred=y_pred_mlp)
mape1 = 100 * mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred_mlp)

regresor_rf = RandomForestRegressor(max_depth=7, 
                                    max_features='sqrt', 
                                    n_estimators=400).fit(X_train, y_train)

y_pred_rf = regresor_rf.predict(X_test)
mse2 = mean_squared_error(y_true=y_test, y_pred=y_pred_rf)
rmse2 =np.sqrt(mse2)
mae2 = mean_absolute_error(y_true=y_test, y_pred=y_pred_rf)
mape2 = 100 * mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred_rf)

regresor_gb = GradientBoostingRegressor(max_depth=2,
                                        max_features='sqrt', 
                                        n_estimators=200).fit(X_train, y_train)

y_pred_gb = regresor_gb.predict(X_test)
mse3 = mean_squared_error(y_true=y_test, y_pred=y_pred_gb)
rmse3 =np.sqrt(mse3)
mae3 = mean_absolute_error(y_true=y_test, y_pred=y_pred_gb)
mape3 = 100 * mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred_gb)

stacking_model = StackingRegressor(
    estimators=[
        ('Redes Neuronales', regresor_mlp),
        ('Random Forest', regresor_rf),
        ('Gradient Boosting', regresor_gb)
    ],
    final_estimator = LinearRegression()
).fit(X_train, y_train)

y_pred_stacking = stacking_model.predict(X_test)
y_pred_stacking

mse4 = mean_squared_error(y_true = y_test, y_pred = y_pred_stacking)
rmse4 =np.sqrt(mse4)
mae4 = mean_absolute_error(y_true = y_test, y_pred = y_pred_stacking) 
mape4 = 100*mean_absolute_percentage_error(y_true = y_test, y_pred = y_pred_stacking) 

import pandas as pd

# Crear un diccionario con los datos de las métricas para cada modelo
data = {
    "Redes Neuronales": [mse1, mae1, mape1, rmse1],
    "Random Forest": [mse2, mae2, mape2, rmse2],
    "Gradient Boosting": [mse3, mae3, mape3, rmse3],
    "StackingRegressor": [mse4, mae4, mape4, rmse4],
}