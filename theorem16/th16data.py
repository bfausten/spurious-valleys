import numpy as np
from th16utils import sample_sphere, make_features


# create f^* (approximation via large p)
def make_teacher(d_in, p_teacher=16000, coef_bound=1.0, seed=0):
    rng = np.random.RandomState(seed)
    W_star = sample_sphere(p_teacher, d_in + 1)
    g_star = rng.uniform(-coef_bound, coef_bound, size=p_teacher) / np.sqrt(p_teacher)
    return {"W_star": W_star, "g_star": g_star}


# compute f^*(X)
def teacher_forward(X, teacher):
    return make_features(X, teacher["W_star"]) @ teacher["g_star"]


# create training and test data
def make_dataset(Ntr=8000, Nte=10000, d_in=10, seed_data=1, seed_teacher=2):
    rng = np.random.RandomState(seed_data)
    Xtr = rng.randn(Ntr, d_in) / np.sqrt(d_in)
    Xte = rng.randn(Nte, d_in) / np.sqrt(d_in)
    teacher = make_teacher(d_in, seed=seed_teacher)
    ytr = teacher_forward(Xtr, teacher)
    yte = teacher_forward(Xte, teacher)
    return Xtr, ytr, Xte, yte
