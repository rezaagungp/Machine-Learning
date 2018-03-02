import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def decision_regions(X, y, classifier, resolution=0.02):
        
    #setup generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot data
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl,1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


"""fetch the iris dataset"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

"""transform label from string to integer"""
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100,[0,2]].values

"""ploting data"""
plt.scatter(X[:50,0], X[:50,1], c='red',marker='o', label='Setosa')
plt.scatter(X[50:100,0],X[50:100,1], c='blue', marker="x", label='Versicolor')
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc= 'upper left')
plt.show()

"""training data and show the result in label"""
nn = Model.Perceptron(eta=0.1, n_iter = 10)
nn.fit(X,y)
plt.plot(range(1,len(nn.errors_)+1),nn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

"""plot training result using decision boundary"""
decision_regions(X,y, classifier=nn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada1 = Model.AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_)+1),np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = Model.AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-Squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

#data normalization
X_nor = np.copy(X)
X_nor[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_nor[:,1] = (X[:,1] - X[:,1].mean())/X[:,0].std()
ada = Model.AdalineGD(n_iter=15, eta=0.1)
ada.fit(X_nor,y)