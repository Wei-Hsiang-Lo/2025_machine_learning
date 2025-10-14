import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from scipy.interpolate import griddata
import numpy as np

# ===========================
# Classification model
# ===========================
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self,x):
        return self.network(x)

def train_classification(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        model.eval()
        val_running_loss = 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * features.size(0)
                all_labels.append(labels)
                all_preds.append(outputs)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_labels = torch.cat(all_labels)
        val_preds = torch.cat(all_preds)
        val_auc = roc_auc_score(val_labels.cpu(), val_preds.cpu())
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        if (epoch+1) % 5 == 0 or epoch==0:
            print(f"[Classification] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    return history

def evaluate_classification(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            all_labels.append(labels)
            all_preds.append(outputs)
    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)
    auc = roc_auc_score(labels.cpu(), preds.cpu())
    return auc

def plot_decision_boundary_geo(model, X, y, X_mean, X_std, device, threshold=0.4):
    """
    Plot decision boundary using original longitude-latitude coordinates.
    """
    model.eval()
    X_cpu, y_cpu = X.cpu(), y.cpu()

    plt.figure(figsize=(8,6))
    plt.scatter(X_cpu[y_cpu==1,0], X_cpu[y_cpu==1,1], color='blue', alpha=0.5, s=5, label='Valid (Class 1)')
    plt.scatter(X_cpu[y_cpu==0,0], X_cpu[y_cpu==0,1], color='silver', alpha=0.5, s=5, label='Invalid (Class 0)')

    # 建立原始經緯度格點
    x_min, x_max = X_cpu[:,0].min()-0.5, X_cpu[:,0].max()+0.5
    y_min, y_max = X_cpu[:,1].min()-0.5, X_cpu[:,1].max()+0.5
    gx, gy = np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
    GX, GY = np.meshgrid(gx, gy)

    # 正規化後送進模型
    grid = torch.tensor(np.column_stack([GX.ravel(), GY.ravel()]), dtype=torch.float32, device=device)
    grid_norm = (grid - X_mean) / X_std

    with torch.no_grad():
        probs = torch.sigmoid(model(grid_norm)).reshape(GX.shape).cpu().numpy()

    # 繪製 decision boundary
    plt.contour(GX, GY, probs, levels=[threshold], colors='black', linewidths=2, linestyles='--')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Decision Boundary (threshold={threshold})")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ===========================
# Regression model
# ===========================
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self,x):
        return self.network(x)

def train_regression(model, train_loader, val_loader, device, epochs=500, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for features, values in train_loader:
            features, values = features.to(device), values.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, values.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        model.eval()
        val_running_loss = 0
        all_vals, all_preds = [], []
        with torch.no_grad():
            for features, values in val_loader:
                features, values = features.to(device), values.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, values)
                val_running_loss += loss.item() * features.size(0)
                all_vals.append(values.view(-1, 1))
                all_preds.append(outputs.view(-1, 1))
        val_loss = val_running_loss / len(val_loader.dataset)
        val_mae = mean_absolute_error(torch.cat(all_vals).cpu(), torch.cat(all_preds).cpu())
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        if (epoch+1) % 50 == 0 or epoch==0:
            print(f"[Regression] Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
    return history

def evaluate_regression(model, loader, device, value_mean, value_std):
    model.eval()
    all_vals, all_preds = [], []
    with torch.no_grad():
        for features, values in loader:
            features, values = features.to(device), values.to(device)
            outputs = model(features).squeeze()
            vals_orig = values * value_std + value_mean
            preds_orig = outputs * value_std + value_mean
            all_vals.append(vals_orig)
            all_preds.append(preds_orig)
    mae = mean_absolute_error(torch.cat(all_vals).cpu(), torch.cat(all_preds).cpu())
    rmse = np.sqrt(mean_squared_error(torch.cat(all_vals).cpu(), torch.cat(all_preds).cpu()))
    print(f"[Regression] Test MAE: {mae:.3f}°C, RMSE: {rmse:.3f}°C")
    return torch.cat(all_preds), torch.cat(all_vals)

# ===========================
# Hybrid model
# ===========================
def hybrid_model(x, c_model, r_model, device, threshold=0.5, invalid_value=-999):
    c_model.eval()
    r_model.eval()
    x = x.to(device)
    with torch.no_grad():
        c_probs = torch.sigmoid(c_model(x).squeeze())
        c_pred = (c_probs >= threshold).float()
        r_pred = r_model(x).squeeze()
        h_pred = c_pred * r_pred + (1 - c_pred) * invalid_value
    return h_pred

# ===========================
# Hybrid plot (only valid region)
# ===========================

def plot_hybrid_prediction_geo(X, y, y_pred, invalid_value=-999, n_grid=100, device='cpu'):
    """
    繪製 Hybrid Prediction 與 MAE（只在有效區域），使用原始經緯度座標。
    X: Tensor, shape [N,2] (Longitude, Latitude)
    y: Tensor, shape [N] (原始溫度, 不正規化)
    y_pred: Tensor, shape [N] (預測值, 不正規化)
    invalid_value: 無效區域標記
    """
    X = X.to(device)
    y = y.to(device)
    y_pred = y_pred.to(device)

    # 只考慮有效區域
    mask_valid = y_pred != invalid_value
    X_valid = X[mask_valid]
    y_valid = y[mask_valid]
    y_pred_valid = y_pred[mask_valid]

    # 建立原始經緯度格點
    lon_min, lon_max = X[:,0].min().item(), X[:,0].max().item()
    lat_min, lat_max = X[:,1].min().item(), X[:,1].max().item()
    gx = np.linspace(lon_min, lon_max, n_grid)
    gy = np.linspace(lat_min, lat_max, n_grid)
    GX, GY = np.meshgrid(gx, gy)

    # 使用 griddata 做插值 (需 numpy)
    points = X_valid.cpu().numpy()
    values_pred = y_pred_valid.cpu().numpy()
    values_true = y_valid.cpu().numpy()

    Pred_grid = griddata(points, values_pred, (GX, GY), method='cubic')
    True_grid = griddata(points, values_true, (GX, GY), method='cubic')
    MAE_grid = np.abs(True_grid - Pred_grid)

    # 繪圖
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.pcolormesh(GX, GY, Pred_grid, cmap="turbo", shading='auto')
    plt.colorbar(label="Predicted Temperature (°C)")
    plt.title("Hybrid Prediction (valid region)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")

    plt.subplot(1,2,2)
    plt.pcolormesh(GX, GY, MAE_grid, cmap="turbo", shading='auto')
    plt.colorbar(label="Absolute Error (°C)")
    plt.title("Hybrid MAE (valid region)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Classification ====
    df_class = pd.read_csv("C:\\Users\\user\\2025_machine_learning\\Week_6\\classification_dataset.csv")
    X_class = torch.tensor(df_class[['Longitude','Latitude']].values, dtype=torch.float32).to(device)
    y_class = torch.tensor(df_class['Class'].values, dtype=torch.float32).to(device)
    X_mean, X_std = X_class.mean(dim=0), X_class.std(dim=0)
    X_class_norm = (X_class - X_mean) / X_std

    dataset = TensorDataset(X_class_norm, y_class)
    train_size = int(0.7*len(dataset))
    val_size = int(0.15*len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size,val_size,test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    c_model = ClassificationModel().to(device)
    history_c = train_classification(c_model, train_loader, val_loader, device, epochs=50)
    auc_test = evaluate_classification(c_model, test_loader, device)
    print(f"[Classification] Test AUC: {auc_test:.4f}")

    # Plot in original longitude-latitude space
    plot_decision_boundary_geo(c_model, X_class, y_class, X_mean, X_std, device, threshold=0.4)

    # ==== Regression ====
    df_reg = pd.read_csv("C:\\Users\\user\\2025_machine_learning\\Week_6\\regression_dataset.csv")
    X_reg = torch.tensor(df_reg[['Longitude','Latitude']].values, dtype=torch.float32).to(device)
    y_reg = torch.tensor(df_reg['Value'].values, dtype=torch.float32).to(device)
    X_reg_norm = (X_reg - X_mean) / X_std
    y_mean, y_std = y_reg.mean(), y_reg.std()

    # Filter c(x)=1
    with torch.no_grad():
        mask = (torch.sigmoid(c_model(X_reg_norm)) >= 0.5).squeeze()
    X_train_reg = X_reg_norm[mask]
    y_train_reg = y_reg[mask]

    dataset_reg = TensorDataset(X_train_reg, y_train_reg)
    train_size = int(0.7*len(dataset_reg))
    val_size = int(0.15*len(dataset_reg))
    test_size = len(dataset_reg) - train_size - val_size
    train_dataset_reg, val_dataset_reg, test_dataset_reg = random_split(dataset_reg, [train_size,val_size,test_size], generator=torch.Generator().manual_seed(42))
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=32, shuffle=False)
    test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

    r_model = RegressionModel().to(device)
    history_r = train_regression(r_model, train_loader_reg, val_loader_reg, device, epochs=500, lr=1e-4)
    y_pred, y_test_orig = evaluate_regression(r_model, test_loader_reg, device, y_mean, y_std)

    # ==== Hybrid ====
    h_pred = hybrid_model(X_reg_norm, c_model, r_model, device)
    plot_hybrid_prediction_geo(X_reg, y_reg, h_pred, device=device)