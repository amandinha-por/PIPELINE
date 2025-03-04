import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dados fictícios
data = {
    'Categoria': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol'],
    'Idade': [5, 3, 8, 6, 4, 7],
    'Quilometragem': [50000, 30000, 120000, 60000, 40000, 90000],
    'Preco': [30000, 25000, 15000, 28000, 26000, 17000]
}

# Convertendo para DataFrame
df = pd.DataFrame(data)

# Separando as variáveis independentes (X) e dependentes (y)
X = df.drop('Preco', axis=1)
y = df['Preco']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Idade', 'Quilometragem']),  # Para variáveis numéricas
        ('cat', OneHotEncoder(), ['Categoria'])  # Para variável categórica
    ])

# Criando o pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())  # Modelo de Regressão Linear
])

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Calculando o erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse}")
