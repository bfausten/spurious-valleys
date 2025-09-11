import numpy as np
from theorem8 import theorem_8 as th8
import time
from model import Monomial1HiddenNN


def main():
    # Define the data
    input_dim = 10
    n_samples = 200
    np.random.seed(0)
    x = np.random.uniform(-2, 2, size=(input_dim, n_samples))
    y = x ** 2

    np.random.seed(int(time.time()))
    net = Monomial1HiddenNN(input_dim=10, hidden_dim=15, output_dim=10, degree=1)
    # Create the model
    model = th8.Model(
        input_dim=10,
        output_dim=10,
        hidden_dim=15,
        w=np.random.randn(15, 10),  # 15 hidden neurons, 10 input features
        u=np.random.randn(10, 15),  # 10 output features, 15 hidden neurons
        activation_function=th8.linear,
        loss_function=th8.mse,

    )
    W, U, w_history, l_history = th8.train(net, x, y, epochs=10000, learning_rate=0.01, tol=1e-4)
    print("Training complete.")
    print("Final output weights (U): " + str(U))
    th8.plot_loss_landscape(model, x, y, w_history)


main()
