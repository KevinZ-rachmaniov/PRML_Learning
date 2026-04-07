import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y-1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

def gini(y):
    if len(y) == 0:
        return 0
    p0 = np.sum(y == 0) / len(y)
    p1 = np.sum(y == 1) / len(y)
    return 1 - p0**2 - p1**2

def best_split(X, y):
    n_samples, n_features = X.shape
    best_gini = float('inf')
    best_feat = 0
    best_thresh = 0

    for feat in range(n_features):
        feat_vals = X[:, feat]
        thresholds = np.unique(feat_vals)
        for thresh in thresholds:
            left_mask = feat_vals <= thresh
            right_mask = ~left_mask
            y_left = y[left_mask]
            y_right = y[right_mask]
            weight_gini = (len(y_left)/n_samples)*gini(y_left) + (len(y_right)/n_samples)*gini(y_right)
            if weight_gini < best_gini:
                best_gini = weight_gini
                best_feat = feat
                best_thresh = thresh
    return best_feat, best_thresh

class Node:
    def __init__(self, feat=None, thresh=None, left=None, right=None, leaf_val=None):
        self.feat = feat
        self.thresh = thresh
        self.left = left
        self.right = right
        self.leaf_val = leaf_val

def build_tree(X, y, max_depth=4):
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    if max_depth == 0 or n0 == len(y) or n1 == len(y):
        leaf = 0 if n0 >= n1 else 1
        return Node(leaf_val=leaf)
    feat, thresh = best_split(X, y)
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    left_tree = build_tree(X[left_mask], y[left_mask], max_depth-1)
    right_tree = build_tree(X[right_mask], y[right_mask], max_depth-1)
    return Node(feat=feat, thresh=thresh, left=left_tree, right=right_tree)

def predict_one(x, tree):
    if tree.leaf_val is not None:
        return tree.leaf_val
    if x[tree.feat] <= tree.thresh:
        return predict_one(x, tree.left)
    else:
        return predict_one(x, tree.right)

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

class AdaBoost:
    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators
        self.trees = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, 1 / n_samples)
        y_ada = np.where(y == 0, -1, 1)

        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, p=w)
            tree = build_tree(X[idx], y[idx])
            y_pred = predict(X, tree)
            y_pred_ada = np.where(y_pred == 0, -1, 1)

            err = np.sum(w[y_pred_ada != y_ada])
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

            w *= np.exp(-alpha * y_ada * y_pred_ada)
            w /= np.sum(w)

            self.trees.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        total = np.zeros(X.shape[0])
        for a, t in zip(self.alphas, self.trees):
            pred = predict(X, t)
            total += a * np.where(pred == 0, -1, 1)
        return np.where(total < 0, 0, 1)

X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

model = AdaBoost(n_estimators=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = np.mean(y_pred == y_test)
print("AdaBoost Test Accuracy =", acc)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_test, cmap='viridis')
ax1.set_title("Ground Truth")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_pred, cmap='viridis')
ax2.set_title(f"AdaBoost Prediction | Acc={acc:.2f}")
plt.show()