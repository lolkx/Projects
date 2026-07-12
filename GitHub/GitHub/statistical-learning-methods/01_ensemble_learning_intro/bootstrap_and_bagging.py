import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree  import DecisionTreeRegressor

#Random data

X = np.linspace(-3,3,100).reshape(-1, 1)
y = X**2 +np.random.normal(0,1,X.shape)

n_bootstrap = 10
sample_size = 80

plt.scatter(X, y, label= 'Data',color='orange')
models = []
for i in range(n_bootstrap):
    indexes = np.random.choice(len(X), size=sample_size, replace =True)
    X_sample, y_sample = X[indexes], y[indexes]
    model = DecisionTreeRegressor(max_depth = 3)
    model.fit(X_sample, y_sample)
    models.append(model)
    y_pred = model.predict(X)
    if i==n_bootstrap-1:
        plt.plot(X, y_pred, color='blue', alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Comparison between different models')
        plt.show()
    else:
        plt.plot(X, y_pred, color='blue', alpha=0.3)



y_bagging = np.mean([model.predict(X) for model in models], axis=0)

plt.scatter(X, y, label='Data', color='orange' )
plt.plot(X, y_bagging, label = 'Model prediction', color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Final prediction using the mean of all models')
plt.legend(loc='upper center')
plt.show()
