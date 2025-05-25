import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(n_samples=10000, noise=0.2, test_size=0.3, random_state=42, plot=False):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='viridis')
        plt.title("make_moons Dataset")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    load_data(plot=True)