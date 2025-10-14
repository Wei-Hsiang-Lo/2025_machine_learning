import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import roc_auc_score

# =========================
# QDA 修正版
# =========================
def log_normal_pdf(x, mu, inv_cov, logdet_cov):
    diff = x - mu
    quad = torch.einsum('bi,ij,bj->b', diff, inv_cov, diff)
    d = x.shape[1]
    log_2pi = torch.log(torch.tensor(2 * torch.pi, dtype=x.dtype, device=x.device))
    return -0.5 * (d * log_2pi + logdet_cov + quad)


def qda_train(x, y, reg=1e-6):
    if x.ndim != 2 or len(torch.unique(y)) > 2:
        raise ValueError("x should be 2D tensor， y should be binary class {0, 1}")

    # 保護 phi 避免 log(0)
    phi = torch.clamp(torch.mean(y.float()), 1e-6, 1 - 1e-6).item()
    
    X0, X1 = x[y == 0], x[y == 1]
    mu0, mu1 = torch.mean(X0, dim=0), torch.mean(X1, dim=0)
    reg_matrix = reg * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)

    # 共變異矩陣（使用最大似然估計）
    S0 = (X0 - mu0).T @ (X0 - mu0) / X0.shape[0] + reg_matrix
    S1 = (X1 - mu1).T @ (X1 - mu1) / X1.shape[0] + reg_matrix

    inv0, inv1 = torch.linalg.inv(S0), torch.linalg.inv(S1)
    logdet0, logdet1 = torch.logdet(S0), torch.logdet(S1)
    
    return {
        "phi": phi,
        "mu0": mu0, "mu1": mu1,
        "inv0": inv0, "inv1": inv1,
        "logdet0": logdet0, "logdet1": logdet1
    }

def qda_predict_proba(X, params):
    log_p0 = torch.log(torch.tensor(1 - params["phi"], device=X.device)) + \
             log_normal_pdf(X, params["mu0"], params["inv0"], params["logdet0"])
    log_p1 = torch.log(torch.tensor(params["phi"], device=X.device)) + \
             log_normal_pdf(X, params["mu1"], params["inv1"], params["logdet1"])

    # log-sum-exp trick
    M = torch.maximum(log_p0, log_p1)
    den = M + torch.log(torch.exp(log_p0 - M) + torch.exp(log_p1 - M))
    return torch.exp(log_p1 - den)

def qda_predict(x_test, params, threshold=0.4):
    probs = qda_predict_proba(x_test, params)
    return (probs >= threshold).long()

def plot_qda_decision_boundary_torch(params, X, y, out_path, threshold=0.4):
    X_np, y_np = X.cpu().numpy(), y.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], color='blue', label='Class 1', alpha=0.5, s=5)
    plt.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], color='silver', label='Class 0', alpha=0.5, s=5)
    
    xmin, xmax = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    ymin, ymax = X_np[:, 1].min() - 0.7, X_np[:, 1].max() + 0.9
    gx, gy = np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200)
    GX, GY = np.meshgrid(gx, gy)

    grid_pts = torch.tensor(np.column_stack([GX.ravel(), GY.ravel()]), dtype=X.dtype, device=X.device)
    prob = qda_predict_proba(grid_pts, params).reshape(GX.shape).cpu().numpy()

    plt.contour(GX, GY, prob, levels=[threshold], colors='red', linewidths=2, linestyles='--')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('QDA Decision Boundary')
    plt.legend()
    plt.axis('equal')
    plt.savefig(out_path)
    plt.close()

# ===========================
# main function
# ===========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    # =========================
    # Load datasets
    # =========================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "classification_dataset.csv")
    print(f"Reading CSV file from: {csv_path}")
    df_class = pd.read_csv(csv_path)
    x = df_class[['Longitude', 'Latitude']].values.astype(np.float32)
    y = df_class['Class'].values.astype(np.float32)

    features = torch.tensor(x, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32)

    # construct dataset
    dataset = TensorDataset(features, labels)

    # calculate sizes
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(0.7 * dataset_size)
    val_size   = int(0.15 * dataset_size)
    test_size  = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        TensorDataset(features, labels), [train_size, val_size, test_size], generator=generator
    )

    # 取得訓練、驗證、測試集的特徵和標籤
    X_train, y_train = train_dataset.dataset[train_dataset.indices]
    X_val, y_val = val_dataset.dataset[val_dataset.indices]
    X_test, y_test = test_dataset.dataset[test_dataset.indices]

    # 將所有資料移動到指定設備
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # ===================================================================
    # QDA 模型
    # ===================================================================
    print("===================================")
    print("  Running Model 1: QDA")
    print("===================================")
    
    # QDA不需要特徵標準化，直接訓練
    # 標籤需要是 long 類型
    qda_params = qda_train(X_train, y_train.long())
    
    # 在測試集上評估
    qda_probs_test = qda_predict_proba(X_test, qda_params)
    qda_auc = roc_auc_score(y_test.cpu().numpy(), qda_probs_test.cpu().numpy())
    
    print(f"\n---- QDA Test Set AUC: {qda_auc:.4f} ----\n")
    
    # 視覺化 QDA 決策邊界
    plot_qda_decision_boundary_torch(qda_params, features.to(device), labels.to(device), "qda_boundary.png")
    print("QDA decision boundary plot saved to qda_boundary.png")

    print("\n===================================")
    print("         Comparison Summary")
    print("===================================")
    print(f"QDA Test AUC: {qda_auc:.4f}")
    print("===================================")