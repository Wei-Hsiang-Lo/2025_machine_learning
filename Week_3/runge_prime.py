import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Runge function and derivative
# -----------------------------
def runge(x):
    return 1 / (1 + 25 * x**2)

def runge_prime(x):
    return -50 * x / (1 + 25 * x**2)**2

# -----------------------------
# Activation function and derivative
# -----------------------------
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

# -----------------------------
# Initialize parameters
# -----------------------------
def init_params(input_dim, hidden_dim1, hidden_dim2, output_dim):
    np.random.seed(4)
    W1 = np.random.randn(input_dim, hidden_dim1) * np.sqrt(1/input_dim)
    b1 = np.zeros((1, hidden_dim1))
    W2 = np.random.randn(hidden_dim1, hidden_dim2) * np.sqrt(1/hidden_dim1)
    b2 = np.zeros((1, hidden_dim2))
    W3 = np.random.randn(hidden_dim2, output_dim) * np.sqrt(1/hidden_dim2)
    b3 = np.zeros((1, output_dim))
    return W1,b1,W2,b2,W3,b3

# -----------------------------
# Forward propagation
# -----------------------------
def forward(x, W1,b1,W2,b2,W3,b3):
    z1 = x.dot(W1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = tanh(z2)
    z3 = a2.dot(W3) + b3
    y_pred = z3
    cache = (x, z1,a1,z2,a2,z3)
    return y_pred, cache

# -----------------------------
# Loss functions
# -----------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# -----------------------------
# Backward propagation (returns grads + derivative wrt input)
# -----------------------------
def backward(y_true, y_pred, cache, W2, W3, derivative_loss=False):
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

    # 如果要求 derivative loss，需要額外計算 dy/dx
    if derivative_loss:
        # chain rule: dy/dx = W1 * σ'(z1) * W2 * σ'(z2) * W3
        J = W1 * 0.0 + 1.0  # identity since input dim=1
        # first layer
        J = J.dot(W1.T) * tanh_deriv(z1)
        # second layer
        J = J.dot(W2.T) * tanh_deriv(z2)
        # third layer
        J = J.dot(W3)
        dy_dx_pred = J  # shape (m,1)
        return dW1, db1, dW2, db2, dW3, db3, dy_dx_pred
    else:
        return dW1, db1, dW2, db2, dW3, db3

# -----------------------------
# Adam optimizer
# -----------------------------
def init_adam(W1,b1,W2,b2,W3,b3):
    params = [W1,b1,W2,b2,W3,b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    return m, v

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

# -----------------------------
# Data
# -----------------------------
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
W1,b1,W2,b2,W3,b3 = init_params(1,64,64,1)
params = [W1,b1,W2,b2,W3,b3]
m, v = init_adam(*params)

# -----------------------------
# Training loop with derivative loss
# -----------------------------
epochs = 500
lr = 0.01
lambda_deriv = 1.0

train_losses = []
train_f_losses = []
train_d_losses = []
val_losses = []
val_f_losses = []
val_d_losses = []

for epoch in range(1, epochs+1):
    # forward
    y_pred, cache = forward(x_train, *params)
    # function loss
    f_loss = mse(y_train, y_pred)
    # backward + derivative prediction
    grads = backward(y_train, y_pred, cache, params[2], params[4], derivative_loss=True)
    dW1, db1, dW2, db2, dW3, db3, dy_dx_pred = grads
    # derivative loss
    dy_dx_true = runge_prime(x_train)
    d_loss = mse(dy_dx_true, dy_dx_pred)
    # total loss
    total_loss = f_loss + lambda_deriv * d_loss
    # update
    grads_no_deriv = [dW1, db1, dW2, db2, dW3, db3]
    params, m, v = update_params_adam(params, grads_no_deriv, m, v, t=epoch, lr=lr)

    # validation
    y_val_pred, cache_val = forward(x_val, *params)
    f_val_loss = mse(y_val, y_val_pred)
    grads_val = backward(y_val, y_val_pred, cache_val, params[2], params[4], derivative_loss=True)
    _,_,_,_,_,_, dy_dx_val_pred = grads_val
    d_val_loss = mse(runge_prime(x_val), dy_dx_val_pred)
    val_total = f_val_loss + lambda_deriv * d_val_loss

    # record
    train_losses.append(total_loss)
    train_f_losses.append(f_loss)
    train_d_losses.append(d_loss)
    val_losses.append(val_total)
    val_f_losses.append(f_val_loss)
    val_d_losses.append(d_val_loss)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {total_loss:.6f}, f_loss: {f_loss:.6f}, d_loss: {d_loss:.6f}, Val Loss: {val_total:.6f}")

# -----------------------------
# Test performance
# -----------------------------
y_test_pred, cache_test = forward(x_test, *params)
f_test_mse = mse(y_test, y_test_pred)
f_test_linf = np.max(np.abs(y_test - y_test_pred))

grads_test = backward(y_test, y_test_pred, cache_test, params[2], params[4], derivative_loss=True)
_,_,_,_,_,_, dy_dx_test_pred = grads_test
dy_dx_true_test = runge_prime(x_test)
d_test_mse = mse(dy_dx_true_test, dy_dx_test_pred)
d_test_linf = np.max(np.abs(dy_dx_true_test - dy_dx_test_pred))

# -----------------------------
# Plots
# -----------------------------
x_dense = np.linspace(-1,1,500).reshape(-1,1)
y_dense_pred, cache_dense = forward(x_dense, *params)
grads_dense = backward(runge(x_dense), y_dense_pred, cache_dense, params[2], params[4], derivative_loss=True)
_,_,_,_,_,_, dy_dx_dense_pred = grads_dense

plt.figure(figsize=(7,5))
plt.plot(x_dense, runge(x_dense), label="True Runge f(x)")
plt.plot(x_dense, y_dense_pred, label="NN prediction")
plt.scatter(x_train, y_train, s=10, c="gray", alpha=0.5, label="Training data")
plt.legend()
plt.title("Runge Function Approximation (with derivative loss)")
plt.show()

plt.figure(figsize=(7,5))
plt.plot(x_dense, runge_prime(x_dense), label="True derivative f'(x)")
plt.plot(x_dense, dy_dx_dense_pred, label="NN derivative")
plt.legend()
plt.title("Derivative Approximation")
plt.show()

plt.figure(figsize=(7,5))
plt.plot(train_losses, label="Train total")
plt.plot(train_f_losses, label="Train function")
plt.plot(train_d_losses, label="Train derivative")
plt.plot(val_losses, label="Val total")
plt.plot(val_f_losses, label="Val function")
plt.plot(val_d_losses, label="Val derivative")
plt.yscale("log")
plt.legend()
plt.title("Loss curves with derivative loss")
plt.show()

# -----------------------------
# Derivative evaluation (Train/Test)
# -----------------------------

def runge_prime(x):
    return -50 * x / (1 + 25 * x**2)**2

def compute_derivative(x, params):
    W1,b1,W2,b2,W3,b3 = params
    # Forward
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    # Compute derivative via chain rule
    J = np.ones((x.shape[0], 1))   # start from dy/dz3 = 1
    # layer 3
    J = J.dot(W3.T) * (1 - np.tanh(z2)**2)   # propagate through a2
    # layer 2
    J = J.dot(W2.T) * (1 - np.tanh(z1)**2)   # propagate through a1
    # layer 1
    J = J.dot(W1.T)                          # propagate back to x
    return J

# Training derivative error
dy_dx_train_pred = compute_derivative(x_train, params)
dy_dx_train_true = runge_prime(x_train)
train_mse = np.mean((dy_dx_train_true - dy_dx_train_pred)**2)
train_maxerr = np.max(np.abs(dy_dx_train_true - dy_dx_train_pred))

# Testing derivative error
dy_dx_test_pred = compute_derivative(x_test, params)
dy_dx_test_true = runge_prime(x_test)
test_mse = np.mean((dy_dx_test_true - dy_dx_test_pred)**2)
test_maxerr = np.max(np.abs(dy_dx_test_true - dy_dx_test_pred))

print(f"Derivative Train MSE: {train_mse:.6e}, Max|err|: {train_maxerr:.6e}")
print(f"Derivative Test  MSE: {test_mse:.6e}, Max|err|: {test_maxerr:.6e}")