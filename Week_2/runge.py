import numpy as np
import matplotlib.pyplot as plt

# Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# Activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

# Inintialize parameters
def init_params(input_dim, hidden_dim1, hidden_dim2, output_dim):
    np.random.seed(4)
    # Glorot/Xavier initialization for tanh
    W1 = np.random.randn(input_dim, hidden_dim1) * np.sqrt(1/input_dim)
    b1 = np.zeros((1, hidden_dim1))
    W2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(1/hidden_dim1)
    b2 = np.zeros((1, hidden_dim2))
    W3 = np.random.randn(hidden_dim2, output_dim) * np.sqrt(1/hidden_dim2)
    b3 = np.zeros((1, output_dim))
    return W1,b1,W2,b2,W3,b3

# Forward propagation
def forward(x, W1,b1,W2,b2,W3,b3):
    z1 = x.dot(W1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = tanh(z2)
    z3 = a2.dot(W3) + b3
    y_pred = z3  # regression output
    cache = (x, z1,a1,z2,a2,z3)
    return y_pred, cache

# Loss function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Backward propagation
def backward(y_true, y_pred, cache, W2, W3):
    x, z1,a1,z2,a2,z3 = cache
    m = y_true.shape[0]

    dz3 = (y_pred - y_true) / m
    dW3 = a2.T.dot(dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = dz3.dot(W3.T)
    dz2 = da2 * tanh_deriv(z2)
    dW2 = a1.T.dot(dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2.dot(W2.T)
    dz1 = da1 * tanh_deriv(z1)
    dW1 = x.T.dot(dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Initialize the state for Adam
def init_adam(W1,b1,W2,b2,W3,b3):
    params = [W1,b1,W2,b2,W3,b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    return m, v

# Updating parameters (Adam)
def update_params_adam(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    new_params = []
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = beta1 * m[i] + (1-beta1) * g
        v[i] = beta2 * v[i] + (1-beta2) * (g**2)
        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        new_params.append(p)
    return new_params, m, v

# Data
N = 100
x = np.linspace(-1, 1, N).reshape(-1,1)
y = runge(x)

# Split manually
idx = np.arange(N)
np.random.seed(4)
np.random.shuffle(idx)

train_idx = idx[:int(0.7*N)]
val_idx   = idx[int(0.7*N):int(0.85*N)]
test_idx  = idx[int(0.85*N):]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val     = x[val_idx], y[val_idx]
x_test, y_test   = x[test_idx], y[test_idx]

# -----------------------------
# Initialize parameters
# -----------------------------
W1,b1,W2,b2,W3,b3 = init_params(1,128,128,1)
params = [W1,b1,W2,b2,W3,b3]

# 初始化 Adam 狀態 (新增)
m, v = init_adam(*params)

train_losses = []
val_losses = []

# -----------------------------
# Training loop
# -----------------------------
epochs = 500
lr = 0.01  # 建議用較小 lr, Adam 會自動調整
for epoch in range(1, epochs+1):
    # forward
    y_pred, cache = forward(x_train, *params)
    # loss
    loss = mse(y_train, y_pred)
    # backward
    grads = backward(y_train, y_pred, cache, params[2], params[4])

    params, m, v = update_params_adam(params, grads, m, v, t=epoch, lr=lr)
    
    # validation loss
    y_val_pred, _ = forward(x_val, *params)
    val_loss = mse(y_val, y_val_pred)
    
    train_losses.append(loss)
    val_losses.append(val_loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss:.6f}, Val Loss: {val_loss:.6f}")

y_test_pred, _ = forward(x_test, *params)
test_loss = mse(y_test, y_test_pred)
print(f"Test MSE: {test_loss:.6f}")

x_dense = np.linspace(-1,1,500).reshape(-1,1)
y_dense_pred,_ = forward(x_dense, *params)

plt.plot(x_dense, runge(x_dense), label="True Runge")
plt.plot(x_dense, y_dense_pred, label="NN Approximation")
plt.scatter(x_train, y_train, s=10, c="gray", alpha=0.5)
plt.legend()
plt.show()

plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Val loss")
plt.yscale("log")
plt.legend()
plt.show()