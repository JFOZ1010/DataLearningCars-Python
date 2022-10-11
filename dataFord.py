#importando librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importar el data set
dataset = pd.read_csv('dataFord.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#dividir el data set en conjunto de testing y entrenamiento 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Crear modelo de Regresión Lineal Simple con el conjunto de entrenamiento
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color = "green")

plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Año de venta carro - Precio (Conjunto de Entrenamiento)")

plt.xlabel("Año de venta carro")
plt.ylabel("Precio (en $)")
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")

plt.title("Año de venta carro - Precio (Conjunto de Testing)")
plt.xlabel("Año de venta carro")
plt.ylabel("Precio (en $)")

plt.show()



# path: dataFord.csv
# Año de venta carro,Precio
# 2014,24000
# 2013,13900
# 2017,40000
# 2016,36900
# 2015,31900
# 2014,24900
# 2013,19900
# 2017,45000
# 2016,39900
# 2015,34900
# 2014,27900
# 2013,21900
# 2017,48000
# 2016,41900
# 2015,36900
# 2014,29900
# 2013,23900
# 2017,50000
# 2016,43900




