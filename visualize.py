import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(X, y, title=""):
    plt.scatter(X[:,0], X[:,1], c=y, s=1)
    plt.title(title)
    plt.show()

def plot_decision_boundary(model, X, y, title=""):

    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, s=1)
    plt.title(title)
    plt.show()



