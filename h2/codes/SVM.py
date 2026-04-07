import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y-1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

kernels = ['linear', 'poly', 'rbf']
predictions = []
accuracies = []

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    predictions.append(y_pred)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"{kernel} kernel Accuracy = {acc:.4f}")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 4))

ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_test, cmap='viridis')
ax1.set_title("Ground Truth")

ax2 = fig.add_subplot(142, projection='3d')
ax2.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=predictions[0], cmap='viridis')
ax2.set_title(f"Linear | Acc={accuracies[0]:.2f}")

ax3 = fig.add_subplot(143, projection='3d')
ax3.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=predictions[1], cmap='viridis')
ax3.set_title(f"Poly | Acc={accuracies[1]:.2f}")

ax4 = fig.add_subplot(144, projection='3d')
ax4.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=predictions[2], cmap='viridis')
ax4.set_title(f"RBF | Acc={accuracies[2]:.2f}")

plt.tight_layout()
plt.show()