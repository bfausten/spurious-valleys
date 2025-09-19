import numpy as np
from theorem8 import theorem_8 as th8
from model import Monomial1HiddenNN


# for figure 2a
def main1():
    # data
    input_dim = 10
    n_samples = 100
    np.random.seed(0)
    x = np.random.uniform(-2, 2, size=(input_dim, n_samples))
    y = 3 * x

    inits = 4
    nets = {
        f"{i}": Monomial1HiddenNN(input_dim=10, hidden_dim=20, output_dim=10,
                                  degree=1, seed_W=1, seed_U=i)
        for i in range(inits)
    }
    w_histories = {}
    for key, net in nets.items():
        W, U, w_hist, l_hist = th8.train(net, x, y, epochs=1000, learning_rate=0.01, tol=1e-4)
        w_histories[key] = np.asarray(w_hist)

    th8.plot_contours_grid(nets, x, y, w_histories)


# for figure 2b
def main2():
    # data
    input_dim = 10
    n_samples = 200
    np.random.seed(0)
    x = np.random.uniform(-2, 2, size=(input_dim, n_samples))
    y = 3 * x

    nets = {
        "1": Monomial1HiddenNN(input_dim=10, hidden_dim=2, output_dim=10,
                               degree=1, seed_W=1, seed_U=1),
        "2": Monomial1HiddenNN(input_dim=10, hidden_dim=5, output_dim=10,
                               degree=1, seed_W=2, seed_U=2),
        "3": Monomial1HiddenNN(input_dim=10, hidden_dim=9, output_dim=10,
                               degree=1, seed_W=3, seed_U=3),
        "4": Monomial1HiddenNN(input_dim=10, hidden_dim=15, output_dim=10,
                               degree=1, seed_W=4, seed_U=4),
    }
    l_histories = {}
    for key, net in nets.items():
        W, U, w_hist, l_hist = th8.train(net, x, y, epochs=2000, learning_rate=0.01, tol=1e-4)
        l_histories[key] = np.asarray(l_hist)
    th8.plot_loss_curves(l_histories)
