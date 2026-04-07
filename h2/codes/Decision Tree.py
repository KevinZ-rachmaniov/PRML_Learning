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

# 寻找最佳特征+最佳分裂阈值
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

#Tree building
class Node:
    def __init__(self, feat=None, thresh=None, left=None, right=None, leaf_val=None):
        self.feat = feat       # 分裂特征
        self.thresh = thresh   # 分裂阈值
        self.left = left       # 左子树
        self.right = right     # 右子树
        self.leaf_val = leaf_val # 叶节点分类值

def build_tree(X, y, depth=0, max_depth=4):
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    # 终止条件：深度到顶 / 样本全同一类
    if depth >= max_depth or n0 == len(y) or n1 == len(y):
        leaf = 0 if n0 >= n1 else 1
        return Node(leaf_val=leaf)
    # 找最优分裂
    feat, thresh = best_split(X, y)
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    # 递归建左右子树
    left_tree = build_tree(X[left_mask], y[left_mask], depth+1, max_depth)
    right_tree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)
    return Node(feat=feat, thresh=thresh, left=left_tree, right=right_tree)

# 单样本预测
def predict_one(x, tree):
    if tree.leaf_val is not None:
        return tree.leaf_val
    if x[tree.feat] <= tree.thresh:
        return predict_one(x, tree.left)
    else:
        return predict_one(x, tree.right)

# 批量预测
def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

# ----------------------3. 加载数据+训练+测试----------------------
# 训练集：1000个(500C0+500C1)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)
# 测试集：500个(250C0+250C1)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# 训练手写决策树
my_tree = build_tree(X_train, y_train, max_depth=4)

# 预测+算准确率
y_pred = predict(X_test, my_tree)
acc = np.sum(y_pred == y_test) / len(y_test)

print("===== results =====")
print(f"Accuracy：{acc:.4f}")


fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_test, cmap='viridis')
ax1.set_title("actual")
ax1.set_xlabel("X");ax1.set_ylabel("Y");ax1.set_zlabel("Z")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_pred, cmap='viridis')
ax2.set_title(f"Decision Tree | Acc:{acc:.2f}")
ax2.set_xlabel("X");ax2.set_ylabel("Y");ax2.set_zlabel("Z")
plt.show()