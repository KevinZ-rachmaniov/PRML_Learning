import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist


train_path = r"C:\Users\ALIENWARE\Desktop\PRML\h1\Data4Regression.xlsx"
test_path = r"C:\Users\ALIENWARE\Desktop\PRML\h1\Data4Regression_test.xlsx"

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

x_train = df_train.iloc[:, 0].values.reshape(-1, 1)
y_train = df_train.iloc[:, 1].values.reshape(-1, 1)
x_test = df_test.iloc[:, 0].values.reshape(-1, 1)
y_test = df_test.iloc[:, 1].values.reshape(-1, 1)


def gaussian_kernel(x, x_i, h):
    """
    高斯核函数
    x: 待预测点 (m, 1)
    x_i: 训练点 (n, 1)
    h: bandwidth
    return: 核矩阵 (m, n)
    """
    pairwise_dists = cdist(x, x_i, 'euclidean')
    return np.exp(- (pairwise_dists ** 2) / (2 * h ** 2))

def kernel_regression_predict(x_train, y_train, x_test, h):
   
    K = gaussian_kernel(x_test, x_train, h)
    K_sum = K.sum(axis=1, keepdims=True)
    # 避免除0
    K_sum[K_sum == 0] = 1e-8
    y_pred = (K @ y_train) / K_sum
    return y_pred

bandwidths = [0.1, 0.5, 1.0, 1.5]

for h in bandwidths:

    y_train_pred = kernel_regression_predict(x_train, y_train, x_train, h)
   
    y_test_pred = kernel_regression_predict(x_train, y_train, x_test, h)
    
    
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
  
    print(f"========== Bandwidth = {h} ==========")
    print(f"Train  R2 = {r2_train:.6f}    RMSE = {rmse_train:.6f}")
    print(f"Test   R2 = {r2_test:.6f}    RMSE = {rmse_test:.6f}\n")
    
    
    plt.figure(figsize=(7, 4))
    idx = np.argsort(x_train.flatten())
    plt.scatter(x_train, y_train, s=15, color='#4477AA', label='True')
    plt.plot(x_train[idx], y_train_pred[idx], color='#FF4444', linewidth=2, label='Fit')
    plt.title(f'Kernel Regression (h={h}) - Train | R2={r2_train:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    
    plt.figure(figsize=(7, 4))
    idx = np.argsort(x_test.flatten())
    plt.scatter(x_test, y_test, s=15, color='#4477AA', label='True')
    plt.plot(x_test[idx], y_test_pred[idx], color='#FF4444', linewidth=2, label='Fit')
    plt.title(f'Kernel Regression (h={h}) - Test | R2={r2_test:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)

plt.show()