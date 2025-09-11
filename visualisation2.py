import numpy as np
import matplotlib.pyplot as plt


def polynomial(x, degree=2):
    return x ** degree


class Model:
    def __init__(self, w, u, loss_function, activation_function, input_dim, output_dim=10, hidden_dim=32):
        self.w = w  # weights for the first layer
        self.u = u
        self.loss_function = loss_function
        self.activation_function = activation_function
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Apply the first layer transformation
        hidden = self.activation_function(np.dot(x, self.W))
        # Apply the second layer transformation
        output = np.dot(hidden, self.U)
        return output

    def loss(self, x, y):
        output = self.forward(x)
        return self.loss_function(output, y)

    def get_params(self):
        return self.w, self.u

    def set_params(self, w, u):
        self.w = w
        self.u = u

    def compute_w_u(self):
        pass