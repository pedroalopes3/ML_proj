import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd




# Caminhos dos ficheiros
file1_path = '/Users/tiago/MEEC/Aaut/X_train.npy'
file2_path = '/Users/tiago/MEEC/Aaut/X_test.npy'
file3_path = '/Users/tiago/MEEC/Aaut/y_train.npy'

# Carregar os ficheiros .npy
array_X_train = np.load(file1_path)
array_X_test = np.load(file2_path)
array_y_train = np.load(file3_path)

# Imprimir o conteúdo dos arrays
# print("Array 1:", array_X_train)
# print("Array 2:", array_X_test)
# print("Array 3:", array_y_train)

# Função para calcular a matriz de covariância

def calculate_cov_matrix(model, X):
    # Prever valores
    y_pred = model.predict(X)

    # Calcular os resíduos
    residuals = array_y_train_clean - y_pred

    # Variância dos resíduos
    residual_variance = np.var(residuals, ddof=X.shape[1])

    # Calcular a matriz (X^T X)^-1
    XTX_inv = np.linalg.inv(X.T @ X)

    # Matriz de covariância dos coeficientes
    cov_matrix = residual_variance * XTX_inv

    return cov_matrix

# Função para remover outliers usando MAD-Median Rule
def remove_outliers_mad(X, y, threshold=3):
    # Concatenar X e y para aplicar a regra em todos os valores
    data = np.hstack((X, y.reshape(-1, 1)))

    # Calcular a mediana dos dados
    mediana = np.median(data, axis=0)

    # Calcular o MAD
    mad = np.median(np.abs(data - mediana), axis=0)

    # Aplicar a regra do threshold (limite) baseado no MAD
    mask = np.all(np.abs(data - mediana) <= threshold * mad, axis=1)

    # Filtrar os dados sem outliers
    return X[mask], y[mask]



# Aplicar a remoção de outliers ao conjunto de treino
array_X_train_clean, array_y_train_clean = remove_outliers_mad(array_X_train, array_y_train)
num_pontos_removidos = array_X_train.shape[0] - array_X_train_clean.shape[0]
print(num_pontos_removidos)


# Regressão linear
linear_model = LinearRegression()
linear_model.fit(array_X_train_clean, array_y_train_clean)
linear_coef = linear_model.coef_
print("Linear Coefficients (B parameters):", linear_coef)

# Regressão lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(array_X_train_clean, array_y_train_clean)
lasso_coef = lasso_model.coef_
print("Lasso Coefficients (B parameters):", lasso_coef)

# Regressão ridge
ridge_model = Ridge(alpha=10.0)
ridge_model.fit(array_X_train_clean, array_y_train_clean)
ridge_coef = ridge_model.coef_
print("Ridge Coefficients (B parameters):", ridge_coef)


# Plotar os coeficientes dos três modelos
plt.figure(figsize=(10, 6))

# Número de coeficientes para plotar
indices = np.arange(len(linear_coef))

# Plotar coeficientes de regressão linear
plt.bar(indices - 0.2, linear_coef, width=0.2, label='Linear Regression', color='b')

# Plotar coeficientes do Lasso
plt.bar(indices, lasso_coef, width=0.2, label='Lasso Regression', color='r')

# Plotar coeficientes do Ridge
plt.bar(indices + 0.2, ridge_coef, width=0.2, label='Ridge Regression', color='g')

# Legendas e labels
plt.xlabel('Coeficientes (B1, B2, ..., Bn)')
plt.ylabel('Valores dos Coeficientes')
plt.title('Comparação dos Coeficientes: Linear, Lasso e Ridge Regression')
plt.xticks(indices, [f'B{i+1}' for i in indices])  # Renomear os eixos com B1, B2, etc.
plt.legend()

# Mostrar o gráfico
plt.show()

df = pd.DataFrame(array_X_train_clean)

# Cálculo dos quartis
quartis = df.quantile([0.25, 0.5, 0.75])
print(quartis)

# Calcular a matriz de covariância para cada modelo
cov_matrix_linear = calculate_cov_matrix(linear_model, array_X_train_clean)
cov_matrix_lasso = calculate_cov_matrix(lasso_model, array_X_train_clean)
cov_matrix_ridge = calculate_cov_matrix(ridge_model, array_X_train_clean)

# Obter o erro padrão dos coeficientes
std_errors_lasso = np.sqrt(np.diag(cov_matrix_linear))
std_errors_ridge = np.sqrt(np.diag(cov_matrix_lasso))
std_errors_linear = np.sqrt(np.diag(cov_matrix_ridge))

print(f"Erro padrão dos coeficientes do lasso {std_errors_lasso}")
print(f"Erro padrão dos coeficientes do linear {std_errors_linear}")
print(f"Erro padrão dos coeficientes do ridge {std_errors_ridge}")


# Calcular o R² nos dados de treino e teste para cada modelo
r2_train_linear = linear_model.score(array_X_train_clean, array_y_train_clean)

r2_train_lasso = lasso_model.score(array_X_train_clean, array_y_train_clean)

r2_train_ridge = ridge_model.score(array_X_train_clean, array_y_train_clean)


# Imprimir os valores de R²
print(f"R² - Linear Regression (train): {r2_train_linear}")

print(f"R² - Lasso (train): {r2_train_lasso}")

print(f"R² - Ridge (train): {r2_train_ridge}")

