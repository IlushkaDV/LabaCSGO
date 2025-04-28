# Лабораторная работа: Исследование и оптимизация радиально-базисных сетей для анализа данных CS:GO
# Автор: [ТВОЕ ИМЯ]
# Датасет: CS:GO Professional Matches

# ==========================================
# Пункт 1. Подготовка данных и предварительный анализ (улучшенный EDA)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Загрузка данных
path = 'csgo_games.csv'  # Укажи путь к датасету

df = pd.read_csv(path)
print(f"Размер датасета: {df.shape}")
print(df.head())

# Удаляем нечисловые признаки
df = df.drop(columns=['match_date', 'team_1', 'team_2'])

# Кодируем целевую переменную
df['winner'] = LabelEncoder().fit_transform(df['winner'])

# Очистка данных
df = df.dropna()
df = df.drop_duplicates()

# Гистограммы всех признаков
df.hist(figsize=(20, 15))
plt.suptitle('Распределение признаков датасета', fontsize=16)
plt.tight_layout()
plt.show()

# Корреляционная матрица топ-10 признаков
top_features = df.corr()['winner'].abs().sort_values(ascending=False)[1:11].index.tolist()
plt.figure(figsize=(10,8))
sns.heatmap(df[top_features + ['winner']].corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Корреляция между топ-10 признаками и winner')
plt.show()

# Стандартизация признаков
features = df.drop(columns=['winner']).columns
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df['winner']

# Определение количества классов
n_classes = len(np.unique(y))

# Разделение данных
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ==========================================
# Пункт 2. Реализация и обучение базовой RBF-сети
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, basis_func='gaussian'):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.log_sigma = nn.Parameter(torch.zeros(out_features))
        self.basis_func = basis_func

    def gaussian(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(2)
        return torch.exp(-distances / (2 * torch.exp(self.log_sigma).pow(2)))

    def forward(self, input):
        if self.basis_func == 'gaussian':
            return self.gaussian(input)
        else:
            raise NotImplementedError('Только гауссовская функция реализована')

class RBFNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(RBFNetwork, self).__init__()
        self.rbf = RBFLayer(in_features, hidden_features)
        self.linear = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        return x

kmeans = KMeans(n_clusters=50).fit(X_train)
centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RBFNetwork(X_train.shape[1], hidden_features=50, out_features=n_classes).to(device)
model.rbf.centers.data = centers

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}: Loss = {loss.item():.4f}')

# ==========================================
# Пункт 3. Исследование "проклятия размерности"
# ==========================================

subset_sizes = [5, 10, 15, 20]
results = {}

for size in subset_sizes:
    print(f"\nТестируем размерность: {size}")
    X_sub = X_train[:, :size]
    X_val_sub = X_val[:, :size]

    model = RBFNetwork(size, hidden_features=20, out_features=n_classes).to(device)
    kmeans = KMeans(n_clusters=20).fit(X_sub)
    model.rbf.centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    X_sub_tensor = torch.tensor(X_sub, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)

    for epoch in range(30):
        optimizer.zero_grad()
        output = model(X_sub_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    X_val_tensor = torch.tensor(X_val_sub, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long).to(device)
    val_preds = model(X_val_tensor).argmax(dim=1)
    acc = (val_preds == y_val_tensor).float().mean().item()
    results[size] = acc
    print(f"Accuracy для {size} признаков: {acc:.4f}")

plt.plot(list(results.keys()), list(results.values()), marker='o')
for x, y in results.items():
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom')
plt.xlabel('Размерность признаков')
plt.ylabel('Точность')
plt.title('Влияние размерности на точность')
plt.grid(True)
plt.show()

# ==========================================
# Пункт 4. Оптимизация RBF-сети (расширение через MLP)
# ==========================================

class RBF_MLP_Network(nn.Module):
    def __init__(self, in_features, hidden_features, mlp_hidden, out_features):
        super(RBF_MLP_Network, self).__init__()
        self.rbf = RBFLayer(in_features, hidden_features)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_features)
        )

    def forward(self, x):
        x = self.rbf(x)
        x = self.mlp(x)
        return x

# Пример тренировки такой модели
model = RBF_MLP_Network(X_train.shape[1], 50, 30, n_classes).to(device)
kmeans = KMeans(n_clusters=50).fit(X_train)
model.rbf.centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'[MLP] Epoch {epoch+1}: Loss = {loss.item():.4f}')

# ==========================================
# Пункт 5. Сравнение с другими моделями
# ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

models = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC()
}

X_train_small, X_val_small, y_train_small, y_val_small = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

for name, clf in models.items():
    clf.fit(X_train_small, y_train_small)
    preds = clf.predict(X_val_small)
    acc = accuracy_score(y_val_small, preds)
    print(f"{name}: Accuracy = {acc:.4f}")

# ==========================================
# Пункт 6. Финальный анализ
# ==========================================

# Здесь ты напишешь свои выводы о модели: сравнишь результаты RBF-сети и других моделей.

print("\nРабота завершена!")
