import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


train_path = r"C:\Users\ALIENWARE\Desktop\PRML\h1\Data4Regression.xlsx"
test_path = r"C:\Users\ALIENWARE\Desktop\PRML\h1\Data4Regression_test.xlsx"

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)


x_train = df_train.iloc[:, 0].values.reshape(-1, 1)
y_train = df_train.iloc[:, 1].values.reshape(-1, 1)
x_test = df_test.iloc[:, 0].values.reshape(-1, 1)
y_test = df_test.iloc[:, 1].values.reshape(-1, 1)


scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train_s = scaler_x.fit_transform(x_train)
x_test_s = scaler_x.transform(x_test)
y_train_s = scaler_y.fit_transform(y_train)


x_train_t = torch.tensor(x_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
x_test_t = torch.tensor(x_test_s, dtype=torch.float32)


def poly_features(x, deg):
    feat = torch.ones(x.shape[0], 1)
    for i in range(1, deg+1):
        feat = torch.cat([feat, x**i], dim=1)
    return feat


def fit(x, y, deg, lr=0.001, epochs=30000):
    X = poly_features(x, deg)
    w = torch.randn(deg+1, 1, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(epochs):
        loss = ((X @ w - y)**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w

def predict(x, w, deg):
    X = poly_features(x, deg)
    return (X @ w).detach().numpy()


degrees = [3,7,11,20]

for d in degrees:
    w = fit(x_train_t, y_train_t, d)
    
    # prediction
    y_train_pred_s = predict(x_train_t, w, d)
    y_test_pred_s = predict(x_test_t, w, d)
    
    # inverse
    y_train_pred = scaler_y.inverse_transform(y_train_pred_s)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_s)
    
    # score
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"Order {d}")
    print(f"Train R2={r2_train:.4f} RMSE={rmse_train:.4f}")
    print(f"Test  R2={r2_test:.4f} RMSE={rmse_test:.4f}\n")


    plt.figure(figsize=(7,4))
    idx = np.argsort(x_train.flatten())  
    plt.scatter(x_train, y_train, s=15, color='#4477AA')  
    plt.plot(x_train[idx], y_train_pred[idx], color='#FF4444', linewidth=2)
    plt.title(f"Order {d} - Train | R2={r2_train:.3f}")
    plt.grid(alpha=0.3)


    plt.figure(figsize=(7,4))
    idx = np.argsort(x_test.flatten())
    plt.scatter(x_test, y_test, s=15, color='#4477AA')
    plt.plot(x_test[idx], y_test_pred[idx], color='#FF4444', linewidth=2)
    plt.title(f"Order {d} - Test | R2={r2_test:.3f}")
    plt.grid(alpha=0.3)

plt.show()