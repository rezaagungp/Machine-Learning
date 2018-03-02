import sys
import numpy as np
import AdalineModel
import pandas as pd
import RosenblattModel
sys.path.append('../')
import matplotlib.pyplot as plt
from DecisionRegions import plot_regions

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
nn = RosenblattModel.Perceptron(eta=0.1, n_iter = 10)
nn.fit(X,y)
plt.plot(range(1,len(nn.errors_)+1),nn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

"""plot training result using decision boundary"""
plot_regions(X,y, classifier=nn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

"""plot SSE score for learing rate 0.01"""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
ada1 = AdalineModel.AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_)+1),np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

"""plot SSE score for learing rate 0.0001"""
ada2 = AdalineModel.AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-Squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

"""data normalization"""
X_nor = np.copy(X)
X_nor[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_nor[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()

"""plot decision regions after traing data using adaline"""
ada = AdalineModel.AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_nor,y)
plot_regions(X_nor,y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [normalization]')
plt.ylabel('petal length [normalization]')
plt.legend(loc='upper left')
plt.show()

"""plot SSE for normalization"""
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
