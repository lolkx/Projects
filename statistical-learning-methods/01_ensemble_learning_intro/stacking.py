import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_moons


# Generar datos
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelos individuales

base_models = [('Decision tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
              ('K Neighbors Classifier', KNeighborsClassifier(n_neighbors=5)),
              ('Support Vector Classifier', SVC(kernel='rbf', probability=True, random_state=42))]

meta_model = RandomForestClassifier(n_estimators=50, random_state=42)

stacking_model = StackingClassifier(estimators = base_models, final_estimator = meta_model)

stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
accuracy = stacking_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

def plot_decision_boundary_subplot(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar fondo de decisión
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

    # Graficar puntos de datos
    figsize = (8,4)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, s=30, edgecolors='k')
    plt.title(title)
    plt.show()


plot_decision_boundary_subplot(stacking_model, X, y, title = 'Stacking Decision Boundary')
