import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def polynomial(x, degree=2):
    return x ** degree


def linear(x):
    return x


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class Model:
    def __init__(self, w, u, loss_function, activation_function, input_dim, output_dim, hidden_dim):
        self.w = w  # weights for the first layer
        self.u = u
        self.loss_function = loss_function
        self.activation_function = activation_function
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Apply the first layer transformation
        hidden = self.activation_function(np.dot(self.w, x))
        # Apply the second layer transformation
        output = np.dot(self.u, hidden)
        return output

    def loss(self, x, y):
        output = self.forward(x)
        return self.loss_function(output, y)

    def get_params(self):
        return self.w, self.u

    def set_params(self, w, u):
        self.w = w
        self.u = u


def find_dependent_rows(W):
    lin_independent = []
    lin_dependent = []
    matrix = W.copy()
    rank = np.linalg.matrix_rank(W)
    for i in range(matrix.shape[0]):
        matrix[i] = np.zeros_like(matrix[i])  # Zero out the i-th row to check linear independence
        if np.linalg.matrix_rank(matrix) == rank:
            lin_dependent.append(i)  # i-th row is linearly dependent
        else:
            lin_independent.append(i)  # i-th row is linearly independent
            matrix[i] = W[i]
    coefficients = np.zeros((W.shape[0], W.shape[0]))
    for idx in lin_dependent:
        coefficients[idx, :] = np.linalg.lstsq(matrix.T, W[idx])[0]
    tol = 1e-8
    coefficients[np.abs(coefficients) < tol] = 0  # Set small coefficients to zero
    return matrix, lin_independent, lin_dependent, coefficients


def set_u(W, U):
    matrix, ind, dep, coeffs = find_dependent_rows(W)
    U = U.astype(np.float64)  # Ensure U is a float64 array
    for idx in ind:
        U[:, idx] += U @ coeffs[:, idx]
    for idx in dep:
        U[:, idx] = np.zeros_like(U[:, idx])
    return U, dep


# randomised way to create a full rank matrix
def create_full_rank(w, lin_dep):
    full_rank = w.copy()
    tries = 10000
    for _ in range(tries):
        for idx in lin_dep:
            full_rank[idx] = np.random.rand(w.shape[1])
        if np.linalg.matrix_rank(full_rank) == full_rank.shape[1]:
            return full_rank
    raise ValueError("Failed to create a full rank matrix after multiple attempts.")


def train(model, x, y, epochs=1000, learning_rate=0.01, tol=1e-6):
    model.u, dep = set_u(model.w, model.u)  # Set U based on W (such that the model stays unchanged)
    model.w = create_full_rank(model.w, dep)  # Ensure W is full rank
    weight_history = []
    loss_history = []
    prev_loss = float('inf')
    for epoch in range(epochs):
        output = model.forward(x)
        loss = model.loss(x, y)
        hidden = model.activation_function(np.dot(model.w, x))
        grad_u = np.dot((output - y), hidden.T) / x.shape[1]
        model.u -= learning_rate * grad_u
        weight_history.append(model.u.flatten().copy())
        loss_history.append(loss)
        if epoch % 25 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
        # Early stopping condition
        # if abs(prev_loss - loss) < tol:
        #     print(f"Stopping early at epoch {epoch}, change in loss < {tol}")
        #     break
        # prev_loss = loss
    return model.w, model.u, weight_history, loss_history


def plot_loss_landscape(model, x, y, weight_history):
    weights = np.array(weight_history)
    deltas = np.diff(weights, axis=0)
    pca = PCA(n_components=2)
    pca.fit(deltas)
    directions = pca.components_

    final_w = weights[-1]
    coords = (weights - final_w) @ directions.T

    x_vals = np.linspace(coords[:, 0].min() - 1, coords[:, 0].max() + 1, 50)
    y_vals = np.linspace(coords[:, 1].min() - 1, coords[:, 1].max() + 1, 50)
    xx, yy = np.meshgrid(x_vals, y_vals)
    zz = np.zeros_like(xx)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            delta = xx[i, j] * directions[0] + yy[i, j] * directions[1]
            u_perturbed = final_w + delta
            model.u = u_perturbed.reshape(model.u.shape)
            zz[i, j] = model.loss(x, y)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.8, edgecolor='none')

    path_coords = coords
    path_losses = []
    for u in weights:
        # 1) assign the network’s parameters to this historical weight
        model.u = u.reshape(model.u.shape)
        # 2) compute and store the loss at that point
        path_losses.append(model.loss(x, y))
    ax.plot(path_coords[:, 0], path_coords[:, 1], path_losses, color='red', marker='o', label='Optimization Path')
    ax.scatter(
        coords[-1, 0],
        coords[-1, 1],
        path_losses[-1],
        color='green',
        marker='o',
        s=80,
        label='Optimum'
    )
    ax.set_title("3D Loss Landscape with Optimization Path")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Loss")
    ax.view_init(elev=45, azim=135)
    ax.legend()
    plt.tight_layout()
    plt.show()
