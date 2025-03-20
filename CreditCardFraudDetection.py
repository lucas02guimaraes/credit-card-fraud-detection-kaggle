import pandas as pd

# Carregar o dataset
data = pd.read_csv('creditcard.csv')

# Verificar as primeiras linhas do dataset
print(data.head())

# Verificando a distribuição das classes (fraude vs não fraude)
print(data['Class'].value_counts())

# Escalonamento das variáveis (normalizando os valores)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(['Class'], axis=1)) # Excluindo a coluna target

# Incluindo a coluna 'Class' de volta
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1])
data_scaled['Class'] = data['Class']

from sklearn.model_selection import train_test_split

X = data_scaled.drop(['Class'], axis=1)  # Features
y = data_scaled['Class']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Inicializando o modelo XGBoost
model = XGBClassifier(scale_pos_weight=10)  # scale_pos_weight ajuda a lidar com desbalanceamento

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=3, verbose=1)
grid_search.fit(X_train, y_train)
print(f"Melhores parâmetros: {grid_search.best_params_}")
