import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 数据读取与合并 ----------------------
# 已修复路径，直接运行
train_data = pd.read_csv(r"C:\Users\ALIENWARE\Desktop\STUDY\PRML\h3\archive\LSTM-Multivariate_pollution.csv")
test_data = pd.read_csv(r"C:\Users\ALIENWARE\Desktop\STUDY\PRML\h3\archive\pollution_test_data1.csv")

# 统一列名格式：小写 + 去除多余空格
train_data.columns = train_data.columns.str.strip().str.lower()
test_data.columns = test_data.columns.str.strip().str.lower()

# 合并两份数据并按时间排序
combined = pd.concat([train_data, test_data], ignore_index=True)

# 时间字段格式化与索引构建
if 'date' in combined.columns:
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values(by='date').reset_index(drop=True)
    combined.set_index('date', inplace=True)

# 对分类特征风向进行独热编码
if 'wnd_dir' in combined.columns:
    combined = pd.get_dummies(combined, columns=['wnd_dir'], prefix='wd')

# ---------------------- 缺失值与数据清洗 ----------------------
# 确保预测目标无缺失
combined = combined.dropna(subset=['pollution'])

# 其余缺失值使用均值填充
combined = combined.fillna(combined.mean())

# 仅保留数值型特征
combined = combined.select_dtypes(include=['number'])

# 将目标变量放在第一列，方便后续预测
feature_list = [col for col in combined.columns if col != 'pollution']
combined = combined[['pollution'] + feature_list]

print("Data processing completed, final shape:", combined.shape)

# ---------------------- 数据归一化 ----------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(combined)

# ---------------------- 构建时序样本 ----------------------
window_size = 24  # 使用前24小时数据预测下一小时
sequence_X, target_y = [], []

for idx in range(window_size, len(scaled_values)):
    sequence_X.append(scaled_values[idx - window_size:idx])
    target_y.append(scaled_values[idx, 0])

sequence_X = np.array(sequence_X)
target_y = np.array(target_y)

# ---------------------- 训练集与测试集划分 ----------------------
split_point = int(0.8 * len(sequence_X))
X_tr, X_te = sequence_X[:split_point], sequence_X[split_point:]
y_tr, y_te = target_y[:split_point], target_y[split_point:]

# 转换为PyTorch张量
X_tr_tensor = torch.FloatTensor(X_tr)
X_te_tensor = torch.FloatTensor(X_te)
y_tr_tensor = torch.FloatTensor(y_tr).unsqueeze(-1)
y_te_tensor = torch.FloatTensor(y_te).unsqueeze(-1)

# ---------------------- LSTM 模型定义 ----------------------
class PollutionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(PollutionLSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm_layer(x)
        final_out = lstm_out[:, -1, :]  # 取最后一个时间步输出
        return self.output_layer(final_out)

# 初始化模型
model = PollutionLSTM(input_dim=X_tr_tensor.shape[2], hidden_dim=50, layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ---------------------- 模型训练 ----------------------
total_epochs = 30
batch = 32
train_loader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=batch, shuffle=False)
loss_history = []

for epoch in range(total_epochs):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred_y = model(batch_X)
        loss_val = criterion(pred_y, batch_y)
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | Average Loss: {avg_loss:.4f}")

# ---------------------- 模型预测与反归一化 ----------------------
model.eval()
with torch.no_grad():
    pred_tr = model(X_tr_tensor).cpu().numpy()
    pred_te = model(X_te_tensor).cpu().numpy()

# 反归一化函数
def recover_values(pred_array):
    dummy = np.zeros((len(pred_array), combined.shape[1]))
    dummy[:, 0] = pred_array.ravel()
    return scaler.inverse_transform(dummy)[:, 0]

# 恢复真实尺度
y_tr_real = recover_values(y_tr_tensor.numpy())
y_te_real = recover_values(y_te_tensor.numpy())
pred_tr_real = recover_values(pred_tr)
pred_te_real = recover_values(pred_te)

# ---------------------- 模型评估指标 ----------------------
train_rmse = np.sqrt(mean_squared_error(y_tr_real, pred_tr_real))
test_rmse = np.sqrt(mean_squared_error(y_te_real, pred_te_real))
train_mae = mean_absolute_error(y_tr_real, pred_tr_real)
test_mae = mean_absolute_error(y_te_real, pred_te_real)

print("\n===== Model Performance Evaluation =====")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE:  {test_rmse:.2f}")
print(f"Train MAE:   {train_mae:.2f}")
print(f"Test MAE:    {test_mae:.2f}")

# ---------------------- 结果可视化 ----------------------
# 训练损失曲线
plt.figure(figsize=(10, 4))
plt.plot(loss_history, linewidth=2)
plt.title("Training Loss Trend")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.show()

# 测试集预测对比
plt.figure(figsize=(14, 6))
plt.plot(y_te_real[:200], label='Actual Pollution', color='#1f77b4', linewidth=1.5)
plt.plot(pred_te_real[:200], label='Predicted Pollution', color='#ff4b5c', linewidth=1.5)
plt.title("PM2.5 Forecasting Using Multivariate LSTM")
plt.xlabel("Time Step")
plt.ylabel("Pollution Concentration")
plt.legend()
plt.show()