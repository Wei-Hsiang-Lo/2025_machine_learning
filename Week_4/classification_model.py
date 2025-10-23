import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score

# =========================
# Define the model of classification
# =========================
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
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
    def forward(self, x):
        return self.network(x)

# =========================
# Train and evaluate the model
# =========================
def train_and_evaluate(model, train_loader, val_loader, epochs, learning_rate):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc':[]}
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features, labels
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # validation
        model.eval()
        val_running_loss = 0.0
        all_val_labels, all_val_preds = [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features, labels
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * features.size(0)
                all_val_labels.append(labels)
                all_val_preds.append(outputs)

        val_labels_tensor = torch.cat(all_val_labels)
        val_preds_tensor = torch.cat(all_val_preds)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_auc = roc_auc_score(val_labels_tensor.cpu(), val_preds_tensor.cpu())

        history['val_loss'].append(epoch_val_loss)
        history['val_auc'].append(epoch_val_auc)
        
        # 調整 learning rate
        scheduler.step(epoch_val_loss)

        # early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

        print (f"Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val AUC: {epoch_val_auc:.4f}")
    return history

# Plotting history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history['train_loss'], 'o-', label='Train Loss', markersize=2)
    ax1.plot(history['val_loss'], 'o-', label='Validation Loss', markersize=2)
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot AUC
    ax2.plot(history['val_auc'], 'o-', label='Validation AUC', color='MediumTurquoise', markersize=2)
    ax2.set_title('Validation AUC over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.legend()

    plt.show()

# Evaluate on test set
def evaluate_model_test_set(model, test_loader):
    model.eval()
    all_test_labels = []
    all_test_preds = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features).squeeze()
            all_test_labels.extend(labels.numpy())
            all_test_preds.extend(outputs.numpy())

    test_auc = roc_auc_score(all_test_labels, all_test_preds)
    print(f" ---- Test Set AUC: {test_auc:.4f} ----")
    return test_auc

if __name__ == "__main__":
    # =========================
    # Load datasets
    # =========================
    df_class = pd.read_csv("C:\\Users\\user\\2025_machine_learning\\Week_4\\classification_dataset.csv")
    x = df_class[['Longitude', 'Latitude']].values.astype(np.float32)
    y = df_class['Class'].values.astype(np.float32)

    X_orig = features = torch.tensor(x, dtype=torch.float32)
    Y_orig = labels = torch.tensor(y, dtype=torch.float32)

    # =========================
    # 標準化 features (經緯度)
    # =========================
    mu = features.mean(dim=0)
    sigma = features.std(dim=0)
    features = (features - mu) / sigma

    # 建立 dataset
    dataset = TensorDataset(features, labels)

    # 計算切分大小
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size   = int(0.15 * dataset_size)
    test_size  = dataset_size - train_size - val_size

    # 隨機切分成 train/val/test
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)   # 為了 reproducibility
    )

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ClassificationModel()
    history = train_and_evaluate(model, train_loader, val_loader, 50, 0.001)
    plot_history(history)
    evaluate_model_test_set(model, test_loader)

    # 轉成 numpy
    X = features.numpy()
    y = labels.numpy()

    X_np, y_np = X_orig.cpu().numpy(), Y_orig.cpu().numpy()
    plt.figure(figsize=(8, 6))
    
    # 散點圖
    plt.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], color='CadetBlue', label='Valid (1)', alpha=0.6, s=5)
    plt.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], color='silver', label='Invalid (0)', alpha=0.6, s=5)
    
    # 建立原始經緯度網格
    xmin, xmax = X_np[:,0].min() - 0.5, X_np[:,0].max() + 0.5
    ymin, ymax = X_np[:,1].min() - 0.5, X_np[:,1].max() + 0.5
    gx, gy = np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200)
    GX, GY = np.meshgrid(gx, gy)
    
    # 將網格標準化後送入模型
    grid_orig = np.column_stack([GX.ravel(), GY.ravel()])
    grid_tensor = torch.tensor((grid_orig - mu.numpy()) / sigma.numpy(), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(grid_tensor)).numpy().reshape(GX.shape)
    
    # 畫 decision boundary
    plt.contour(GX, GY, probs, levels=[0.4], colors='black', linewidths=2, linestyles='dashed')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Valid vs Invalid Classification with Decision Boundary')
    plt.legend()
    plt.show()