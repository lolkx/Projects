import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Generate synthetic classification data
X, y = make_classification(n_samples=500,
                           n_features=200,
                           n_informative = 5, n_redundant=100,
                           n_clusters_per_class=1, flip_y=0.1,
                           class_sep=0.2,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# 2. Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 3. Define the cross-validation strategy (Stratified K-Fold for classification)
# Uses k=5 splits, a common practice
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Perform cross-validation and get the scores for each fold
scores = cross_val_score(rf_cv, X, y, cv=k_folds, scoring='accuracy')

rf_cv.fit(X_train, y_train)
y_pred_cv = rf.predict(X_test)
accuracy_cv = accuracy_score(y_test, y_pred_cv)

# --- 5. Visualización (El "Dibujo") ---
labels = ['Without CV', 'CV']
accuracies = [accuracy, accuracy_cv]

figsize=(8, 4)
bars = plt.bar(labels, accuracies, color=['skyblue', 'lightcoral'])
plt.ylabel('Test Accuracy Score')
plt.title('Comparison of Model Accuracy: Without Cross Validation vs. Cross Validation')
plt.ylim(0.0, 1.0) # Establecer límite del eje Y para precisión

# Añadir los valores de precisión encima de las barras
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.01,
        f'{yval:.2f}',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
