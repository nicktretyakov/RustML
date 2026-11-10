# Нажмите Run, чтобы увидеть, как обучается модель
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 150 цветков ириса: их размеры и виды
iris = load_iris()
X_learn, X_new, y_learn, y_new = train_test_split(iris.data, iris.target, random_state=0)

# Модель изучает 112 цветков и правильные ответы для них...
model = KNeighborsClassifier()
model.fit(X_learn, y_learn)

# ...затем определяет виды 38 цветков, которых раньше не видела
guesses = model.predict(X_new)
right = (guesses == y_new).sum()
print(f"Studied {len(X_learn)} flowers")
print(f"Named {right} of {len(X_new)} new flowers correctly")

# Каждый новый цветок показан точкой, цвет которой соответствует предположению модели; красный X обозначает ошибку
fig, ax = plt.subplots()
for i, name in enumerate(iris.target_names):
    picked = guesses == i
    ax.scatter(X_new[picked, 2], X_new[picked, 3], label=f"guessed {name}")
wrong = guesses != y_new
ax.scatter(X_new[wrong, 2], X_new[wrong, 3], marker="x", color="red", s=120, label="mistake")
ax.set_title("The model's guesses for 38 new flowers")
ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.legend()
plt.show()

