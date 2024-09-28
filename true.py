import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso, Ridge, LassoCV, RidgeCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Caminhos para os ficheiros
file1_path = '/Users/tiago/MEEC/Aaut/X_train.npy'
file2_path = '/Users/tiago/MEEC/Aaut/X_test.npy'
file3_path = '/Users/tiago/MEEC/Aaut/y_train.npy'

# Carregar os ficheiros .npy
array_X_train = np.load(file1_path)
array_X_test = np.load(file2_path)
array_y_train = np.load(file3_path)

# -------------------------------------------------Normalizar os dados-------------------------------------------------#

# Normalizar os dados de entrada (X_train)
scaler_x = StandardScaler()
array_X_train_scaled = scaler_x.fit_transform(array_X_train)

# Normalizar os dados de saída (Y_train)
scaler_y = StandardScaler()
array_y_train_scaled = scaler_y.fit_transform(array_y_train.reshape(-1, 1))
array_y_train_scaled = array_y_train_scaled.ravel()

# -------------------------------------------Remover outliers com o Ransac-----------------------------------------#

# Inicializar o modelo de regressão linear
estimator = LinearRegression()
estimator.fit(array_X_train_scaled, array_y_train_scaled)

# Inicializar o RANSAC
ransac = RANSACRegressor(estimator=estimator,
                         residual_threshold=0.3,  # Limite de resíduo para considerar inlier
                         # ajustámos ao analisar os gráficos dos outliers vs inliers
                         random_state=42)  # Observamos que é um random state comum

# Ajustar o RANSAC aos dados de treino (X_train e y_train)
ransac.fit(array_X_train_scaled, array_y_train_scaled)

# Obter a máscara dos inliers (pontos que não são considerados outliers) (a máscara é um vetor boliano)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Filtrar os dados sem outliers
array_X_train_clean = array_X_train_scaled[inlier_mask]
array_y_train_clean = array_y_train_scaled[inlier_mask]


# Contar o número de pontos removidos
num_pontos_removidos = array_X_train.shape[0] - array_X_train_clean.shape[0]
print(f'Número de pontos removidos: {num_pontos_removidos}')

# ------------------------------Realizamos 3 regressões lineares (normal, ridge e lasso)------------------------------#

#  Regressão linear (normal)
linear_model = LinearRegression()
linear_model.fit(array_X_train_clean, array_y_train_clean)
linear_coef = linear_model.coef_
print("Linear Coefficients (B parameters):", linear_coef)


# Lasso with cross-validation (função utilizada para encontrar o melhor alpha para o lasso)
lasso_cv_model = LassoCV(cv=5).fit(array_X_train_clean, array_y_train_clean)
best_alpha_lasso = lasso_cv_model.alpha_
print(f'alfa lasso {best_alpha_lasso}')

# Regressão lasso
lasso_model = Lasso(best_alpha_lasso)
lasso_model.fit(array_X_train_clean, array_y_train_clean)
lasso_coef = lasso_model.coef_
print("Lasso Coefficients (B parameters):", lasso_coef)


# Ajustar o Ridge com validação cruzada (função utilizada para encontrar o melhor alpha para o ridge)
ridge_cv_model = RidgeCV(alphas=[0.1, 0.9, 1, 1.1, 2.5, 50, 100], cv=5).fit(array_X_train_clean, array_y_train_clean)
best_alpha_ridge = ridge_cv_model.alpha_
print(f'alfa ridge {best_alpha_ridge}')

# Regressão ridge
ridge_model = Ridge(best_alpha_ridge)
ridge_model.fit(array_X_train_clean, array_y_train_clean)
ridge_coef = ridge_model.coef_
print("Ridge Coefficients (B parameters):", ridge_coef)

# ------------------------------------Plot dos coeficientes de cada regressão-----------------------------#

labels = ['B1 (air temperature x1)', 'B2 (water temperature)', 'B3 (wind speed)',
          'B4 (wind direction)', 'B5 (illumination)']

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width, linear_coef, width, label='Linear', color='green')
ax.bar(x, lasso_coef, width, label='Lasso', color='yellow')
ax.bar(x + width, ridge_coef, width, label='Ridge', color='red')


ax.set_xlabel('Coefficient')
ax.set_ylabel('Coefficient Values')
ax.set_title('Comparison of Coefficients: Linear, Lasso, and Ridge Regression')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# plot
plt.tight_layout()
plt.show()

# ------------------------------Primeiro calculo do R² sem utilizar validação cruzada------------------------#
# Sabemos que esta não é a forma mais correta de estimar o R²,
# mas foi uma etapa importante do trabalho perceber o porquê e como contornar a situação,
# logo achamos por bem deixar aqui essa parte do código.

#  Calcular o R² nos dados de treino e teste para cada modelo
r2_train_linear = linear_model.score(array_X_train_clean, array_y_train_clean)

r2_train_lasso = lasso_model.score(array_X_train_clean, array_y_train_clean)

r2_train_ridge = ridge_model.score(array_X_train_clean, array_y_train_clean)


# Calcular o R² ajustado
n = len(array_y_train_clean)  # número de observações
p = array_X_train_clean.shape[1]  # número de preditores


r2_train_linear_adj = 1 - (1 - r2_train_linear) * (n - 1) / (n - p - 1)
r2_train_lasso_adj = 1 - (1 - r2_train_lasso) * (n - 1) / (n - p - 1)
r2_train_ridge_adj = 1 - (1 - r2_train_ridge) * (n - 1) / (n - p - 1)

# Imprimir os valores de R²
print(f"R² - Linear Regression (train): {r2_train_linear} e o ajustado {r2_train_linear_adj}")

print(f"R² - Lasso (train): {r2_train_lasso} e o ajustado {r2_train_lasso_adj }")

print(f"R² - Ridge (train): {r2_train_ridge} e o ajustado {r2_train_ridge_adj}")

# ----------------------------------Primeiro calculo do R² sem utilizar validação cruzada------------------------------#

# Definir o número de folds para a validação cruzada
num_folds = 5

# Calcular o R² usando validação cruzada
r2_linear_cv = cross_val_score(linear_model, array_X_train_clean, array_y_train_clean, cv=num_folds, scoring='r2')
r2_lasso_cv = cross_val_score(lasso_model, array_X_train_clean, array_y_train_clean, cv=num_folds, scoring='r2')
r2_ridge_cv = cross_val_score(ridge_model, array_X_train_clean, array_y_train_clean, cv=num_folds, scoring='r2')

# Calcular a média e o desvio padrão dos R² (para cada regressão)
mean_r2_linear = np.mean(r2_linear_cv)
std_r2_linear = np.std(r2_linear_cv)

mean_r2_lasso = np.mean(r2_lasso_cv)
std_r2_lasso = np.std(r2_lasso_cv)

mean_r2_ridge = np.mean(r2_ridge_cv)
std_r2_ridge = np.std(r2_ridge_cv)

# Imprimir os resultados
print(f"Linear Regression CV -R²:{r2_linear_cv} R² médio: {mean_r2_linear:.4f} ± {std_r2_linear:.4f}")
print(f"Lasso Regression CV -R²:{r2_lasso_cv} R² médio: {mean_r2_lasso:.4f} ± {std_r2_lasso:.4f}")
print(f"Ridge Regression CV -R²:{r2_ridge_cv} R² médio: {mean_r2_ridge:.4f} ± {std_r2_ridge:.4f}")

# ------------------------------------------------- Previsões para enviar ---------------------------------------------#

# Normalizar os dados de teste (X_test)
array_X_test_scaled = scaler_x.fit_transform(array_X_test)

B0 = ridge_model.intercept_  # B0
B = ridge_model.coef_        # Coeficientes (array do tipo [B1, B2, B3, B4, B5])

# Multiplicar cada linha de X_test pelos coeficientes e somar o B0
pred_normalizadas = np.dot(array_X_test_scaled, B.T) + B0

# Desnormalizar as previsões
pred_originais = scaler_y.inverse_transform(pred_normalizadas.reshape(-1, 1)).ravel()

# Guardar as predições no ficheiro npy
np.save('/Users/tiago/MEEC/Aaut/pred.npy', pred_originais)


# -----------------------------------------Plot dos outliers vs inliers para cada variável-----------------------------#

# Plot dos dados antes da remoção dos outliers
plt.figure(figsize=(10, 6))

plt.scatter(array_X_train[inlier_mask][:, 0], array_y_train[inlier_mask], c='lightblue', marker='o', label='Inliers')
plt.scatter(array_X_train[outlier_mask][:, 0], array_y_train[outlier_mask], c='red', marker='s', label='Outliers')
plt.xlabel('X1 (Primeira Característica)')
plt.ylabel('y (Variável de Saída)')
plt.title('Identificação de Outliers com RANSAC')
plt.legend()
plt.show()

plt.scatter(array_X_train[inlier_mask][:, 1], array_y_train[inlier_mask], c='lightblue', marker='o', label='Inliers')
plt.scatter(array_X_train[outlier_mask][:, 1], array_y_train[outlier_mask], c='red', marker='s', label='Outliers')
plt.xlabel('X2 (Primeira Característica)')
plt.ylabel('y (Variável de Saída)')
plt.title('Identificação de Outliers com RANSAC')
plt.legend()
plt.show()

plt.scatter(array_X_train[inlier_mask][:, 2], array_y_train[inlier_mask], c='lightblue', marker='o', label='Inliers')
plt.scatter(array_X_train[outlier_mask][:, 2], array_y_train[outlier_mask], c='red', marker='s', label='Outliers')
plt.xlabel('X3 (Primeira Característica)')
plt.ylabel('y (Variável de Saída)')
plt.title('Identificação de Outliers com RANSAC')
plt.legend()
plt.show()

plt.scatter(array_X_train[inlier_mask][:, 3], array_y_train[inlier_mask], c='lightblue', marker='o', label='Inliers')
plt.scatter(array_X_train[outlier_mask][:, 3], array_y_train[outlier_mask], c='red', marker='s', label='Outliers')
plt.xlabel('X4 (Primeira Característica)')
plt.ylabel('y (Variável de Saída)')
plt.title('Identificação de Outliers com RANSAC')
plt.legend()
plt.show()

plt.scatter(array_X_train[inlier_mask][:, 4], array_y_train[inlier_mask], c='lightblue', marker='o', label='Inliers')
plt.scatter(array_X_train[outlier_mask][:, 4], array_y_train[outlier_mask], c='red', marker='s', label='Outliers')
plt.xlabel('X5 (Primeira Característica)')
plt.ylabel('y (Variável de Saída)')
plt.title('Identificação de Outliers com RANSAC')
plt.legend()
plt.show()

# -------------------------------------------------Conclusão-------------------------------------------------#
# Escolhemos os coeficientes provenientes da regressão linear do ridge,
# pois ao reler o enunciado não nos fazia sentido colocar certas variaveis com B=0
# o que acontecia com a utilizacao do metodo de regressao de lasso
