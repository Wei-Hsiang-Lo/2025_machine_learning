import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import griddata

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)
    
def train_and_evaluate(model, train_loader, val_loader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(epochs):
        # switch to training mode, updating parameters
        model.train()
        running_loss = 0.0
        for features, values in train_loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # switch to evaluation mode, not updating parameters
        model.eval()
        val_running_loss = 0.0
        all_val_values = []
        all_val_preds = []
        for features, values in val_loader:
            with torch.no_grad():
                outputs = model(features).squeeze()
                loss = criterion(outputs, values)
                val_running_loss += loss.item() * features.size(0)
                all_val_values.extend(values.numpy())
                all_val_preds.extend(outputs.numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_mae = mean_absolute_error(all_val_values, all_val_preds)

        history['val_loss'].append(epoch_val_loss)
        history['val_mae'].append(epoch_val_mae)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:02d}/{epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val RMSE: {np.sqrt(epoch_val_loss):.4f} | Val MAE: {epoch_val_mae:.4f}')

    return history

def plot_history(history, value_std):
    val_mae_original = np.array(history['val_mae']) * value_std
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    ax1.plot(history['train_loss'], 'o-', label='Train Loss', markersize=2)
    ax1.plot(history['val_loss'], 'o-', label='Val Loss', markersize=2)
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()

    ax2.plot(val_mae_original, 'o-', label='Val MAE', markersize=2, color='MediumTurquoise')
    ax2.set_title('MAE over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model_test_set(model, test_loader, value_mean, value_std):
    model.eval()
    all_test_values = []
    all_test_preds = []

    with torch.no_grad():
        for features, values in test_loader:
            outputs = model(features).squeeze()
            values_orig = values.numpy() * value_std + value_mean
            pred_values_orig = outputs.numpy() * value_std + value_mean
            all_test_values.extend(values_orig)
            all_test_preds.extend(pred_values_orig)

    test_mae = mean_absolute_error(all_test_values, all_test_preds)
    test_rmse = np.sqrt(mean_squared_error(all_test_values, all_test_preds))

    print(f"Test Set MAE: {test_mae:.4f} °C")
    print(f"Test Set RMSE: {test_rmse:.4f} °C")
    idx = np.random.randint(0, len(all_test_values))
    print(f"Example Prediction {idx}:")
    print(f"Predicted Temperature: {all_test_preds[idx]:.2f} °C")
    print(f"Actual Temperature: {all_test_values[idx]:.2f} °C")
    print(f"Error: {abs(all_test_preds[idx] - all_test_values[idx]):.2f} °C")

    return np.array(all_test_preds), np.array(all_test_values)

def plot_prediction_error(X_test, y_test, y_pred, n_grid=100):
    x_min, x_max = X_test[:,0].min(), X_test[:,0].max()
    y_min, y_max = X_test[:,1].min(), X_test[:,1].max()
    X_grid = np.linspace(x_min, x_max, n_grid)
    Y_grid = np.linspace(y_min, y_max, n_grid)
    X0, Y0 = np.meshgrid(X_grid, Y_grid)

    S = griddata(X_test, y_pred, (X0, Y0), method='cubic')
    TT = griddata(X_test, y_test, (X0, Y0), method='cubic')
    Error = TT - S  # 絕對誤差

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.pcolormesh(X0, Y0, S, cmap="turbo", shading='auto')
    plt.colorbar(pad=0.03)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Model Prediction")
    plt.axis("square")

    plt.subplot(1,2,2)
    plt.pcolormesh(X0, Y0, Error, cmap="turbo", shading='auto')
    plt.colorbar(pad=0.03)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Absolute Error")
    plt.axis("square")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # =========================
    # Load datasets
    # =========================
    df_class = pd.read_csv("regression_dataset.csv")
    x = df_class[['Longitude', 'Latitude']].values.astype(np.float32)
    y = df_class['Value'].values.astype(np.float32)

    features = torch.tensor(x, dtype=torch.float32)
    values = torch.tensor(y, dtype=torch.float32)

    # =========================
    # Normalize features (經緯度)
    # =========================
    features_mu = features.mean(dim=0)
    features_sigma = features.std(dim=0)
    features = (features - features_mu) / features_sigma
    # =========================
    # Normalize values (溫度)
    # =========================
    values_mu = values.mean()
    values_sigma = values.std()
    values = (values - values_mu) / values_sigma

    # 建立 dataset
    dataset = TensorDataset(features, values)

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
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = RegressionModel()
    history = train_and_evaluate(model, train_loader, val_loader, 50, 0.001)
    plot_history(history, values_sigma.item())
    evaluate_model_test_set(model, test_loader, values_mu.item(), values_sigma.item())

    # Evaluate
    y_pred_test, y_test_orig = evaluate_model_test_set(model, test_loader, values_mu.item(), values_sigma.item())

    # Prepare X_test for plotting
    X_test = np.array([features[i] * features_sigma + features_mu for i in test_dataset.indices])
    plot_prediction_error(X_test, y_test_orig, y_pred_test)