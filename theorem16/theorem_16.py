"""
Width vs. approximation error (oracle) for random-features ReLU.
- Thin gray lines: each trial across widths
- Bold blue line: median across trials
- Dashed line: 1/p reference
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from th16data import make_dataset
from th16utils import set_seed, sample_sphere, make_features, solve_u_star, mse
matplotlib.use("TkAgg")


# run one trial: sample W, fit u, evaluate approx error
def one_trial(Xtr, ytr, Xte, yte, p, reg_oracle, base_seed):
    set_seed(base_seed)
    d_in = Xtr.shape[1]
    W = sample_sphere(p, d_in + 1)  # W uniform on sphere
    Phi_tr = make_features(Xtr, W)
    Phi_te = make_features(Xte, W)
    u_oracle = solve_u_star(Phi_tr, ytr, reg=reg_oracle)  # best u for fixed W
    approx_err = mse(Phi_te @ u_oracle, yte)  # R(W) across test set
    return approx_err


# sweep across widths p, multiple trials per p
def sweep_p(Xtr, ytr, Xte, yte, p_list, trials=50, reg_oracle=1e-8):
    """
    Returns:
      med_A: (len(p_list),) medians across trials
      all_A: (trials, len(p_list)) per-trial values across widths (for line plots) (one column per p)
    """
    all_A = np.zeros((trials, len(p_list)), dtype=float)
    for j, p in enumerate(p_list):
        vals = []
        for t in range(trials):
            # base seed depends only on trial index so each gray line corresponds to one trial
            approx_err = one_trial(
                Xtr, ytr, Xte, yte,
                p=p, reg_oracle=reg_oracle, base_seed=42 + t
            )
            vals.append(approx_err)
            all_A[t, j] = approx_err
        vals = np.array(vals)
        print(f"p={p:4d} | approx(med)={np.median(vals):.3e}")
    med_A = np.median(all_A, axis=0)
    return med_A, all_A


# run experiments and plot
def main():
    # config
    p_list = [16, 32, 64, 128, 256, 512, 1024, 2048]
    trials = 100

    Xtr, ytr, Xte, yte = make_dataset(
        Ntr=8000, Nte=10000, d_in=10, seed_data=1, seed_teacher=2
    )

    medA, allA = sweep_p(
        Xtr, ytr, Xte, yte, p_list,
        trials=trials, reg_oracle=1e-8
    )

    # plot
    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=140)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # thin gray line per trial
    for t in range(1, allA.shape[0]):
        ax.plot(p_list, allA[t, :], linewidth=1.0, alpha=0.5, color="grey")
    ax.plot(p_list, allA[0, :], linewidth=1.0, alpha=0.5, color="grey", label="Trials")

    # bold median line
    ax.plot(p_list, medA, marker="o", linewidth=3.0, color="tab:blue",
            label="Median R(W)")

    # 1/p reference through the first median point
    ref = medA[0] * (np.array(p_list) / p_list[0]) ** (-1.0)
    ax.plot(p_list, ref, ls="--", linewidth=1.6, color="tab:orange", label="1/p reference")

    ax.set_xlabel("width p")
    ax.set_ylabel("approximation error  R(W)")
    ax.set_title("Width vs. approximation error for random-features")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
