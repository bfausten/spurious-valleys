import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Theta:
    W: torch.Tensor
    u: torch.Tensor


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--p", type=int, default=9)
    p.add_argument("--N", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--path_points", type=int, default=200)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--outdir", type=str, default="outputs_t13")
    return p.parse_args()


@torch.no_grad()
def sample_X(N, n, device):  # function from chatgpt; builds X like in Theorem 13
    X = torch.zeros(N, n, device=device)
    Z = torch.rand(N, device=device) < 0.5
    num1 = Z.sum().item()
    if num1 > 0:
        X[Z, : n - 1] = torch.randn(num1, n - 1, device=device)
    num0 = (~Z).sum().item()
    if num0 > 0:
        X[~Z, n - 1] = torch.randn(num0, device=device)
    return X


@torch.no_grad()
def build_targets(X, alpha, beta, p):  # builds Y like in Theorem 13 (with alpha_i = alpha and v_i = e_i)
    per_coord_relu = F.relu(X)
    g1 = alpha * per_coord_relu[:, :p].sum(dim=1)
    g2 = beta * per_coord_relu[:, -1]
    Y = g1 - g2
    return Y


def make_e(k, n, device):  # standard basis vector
    e = torch.zeros(n, device=device)
    e[k] = 1.0
    return e


def theta_bad(n, p, alpha, device):  # create local min (cancel g_1 but g_2 remains)
    assert p <= n - 1
    W = torch.stack([make_e(i, n, device) for i in range(p)], dim=0)
    u = torch.full((p,), float(alpha), device=device)
    return Theta(W, u)


def theta_good(n, p, alpha, beta, device): # create good theta
    assert p >= 2
    W_rows, u_vals = [], []
    for i in range(p - 1):
        W_rows.append(make_e(i, n, device))
        u_vals.append(float(alpha))
    W_rows.append(make_e(n - 1, n, device))
    u_vals.append(float(-beta))
    W = torch.stack(W_rows, dim=0)
    u = torch.tensor(u_vals, device=device)
    return Theta(W, u)


def random_theta_bad(n, p, alpha, device, w_eps=0.05, u_eps=0.05):  # sample different bad thetas
    assert 2 <= p <= n-1
    W = torch.zeros(p, n, device=device)
    for j in range(p):
        w = torch.zeros(n, device=device)
        w[j] = 1.0  # start at e_j (close to loc min)
        # small noise in the first n-1 coords only
        noise = torch.randn(n-1, device=device) * w_eps
        w[:n-1] += noise  # stays in subspace; no e_n component
        # normalize subspace part (chatgpt recommended)
        s = torch.norm(w[:n-1]) + 1e-8
        w[:n-1] /= s
        w[n-1] = 0.0  # explicit, just to be safe
        W[j] = w
    # positive u close to loc min (exp for > 0)
    u = alpha * torch.exp(u_eps * torch.randn(p, device=device))
    u = torch.clamp(u, min=1e-6)
    return Theta(W, u)


@torch.no_grad()
def mse(theta: Theta, X, Y):  # compute MSE loss
    pre = F.linear(X, theta.W, bias=None)
    pred = (F.relu(pre) @ theta.u).view(-1)
    return torch.mean((pred - Y) ** 2), pred


@torch.no_grad()
def loss_along_path(theta_a: Theta, theta_b: Theta, X, Y, T):  # compute losses along linear path from loc min to good
    ts = torch.linspace(0.0, 1.0, T, device=X.device)
    losses = []
    for t in ts:
        W = (1 - t) * theta_a.W + t * theta_b.W
        u = (1 - t) * theta_a.u + t * theta_b.u
        L, _ = mse(Theta(W, u), X, Y)
        losses.append(L.item())
    return ts.cpu().numpy(), np.array(losses)


def save_path_plot(ts, losses, outpath):  # plot single run
    plt.figure()
    plt.plot(ts, losses)
    plt.xlabel(r"$t$ in path ($\theta(t) = (1-t)\cdot \theta_b + t\cdot\theta_g$)")
    plt.ylabel("Loss")
    plt.title("Loss barrier along a straight path")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_path_multi_overlay(ts, losses_list, outpath):  # plot many runs
    plt.figure()
    for L in losses_list[:-1]:
        plt.plot(ts, L, linewidth=0.8, color="0.75")
    plt.plot(ts, losses_list[-1], linewidth=0.8, color="0.75", label="individual trials")
    mean_curve = np.mean(np.stack(losses_list, axis=0), axis=0)
    plt.plot(ts, mean_curve, linewidth=2.2, color="blue", label="mean")
    plt.xlabel(r"$t$ in path ($\theta(t) = (1-t)\cdot \theta_b + t\cdot\theta_g$)")
    plt.ylabel("Loss")
    plt.title("Loss barrier along straight paths")
    plt.legend()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return mean_curve


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    n, p = args.n, args.p
    assert n >= 3 and 2 <= p <= n - 1
    os.makedirs(args.outdir, exist_ok=True)

    # data + targets
    X = sample_X(args.N, n, device)
    Y = build_targets(X, args.alpha, args.beta, args.p)

    # canonical thetas
    th_bad = theta_bad(n, p, args.alpha, device)  # local min in V^+_p
    th_good = theta_good(n, p, args.alpha, args.beta, device)  # good min

    # single path plot (baseline)
    ts, losses = loss_along_path(th_bad, th_good, X, Y, args.path_points)
    save_path_plot(ts, losses, os.path.join(args.outdir, "path_loss.png"))

    # many bad inits
    if args.runs and args.runs > 0:
        print(f"Generating {args.runs} paths from different u>0 inits…")
        losses_list = []
        base_seed = args.seed
        for k in range(args.runs):
            seed_k = base_seed + 123 + k
            torch.manual_seed(seed_k)
            np.random.seed(seed_k)
            random.seed(seed_k)
            th_bad_k = random_theta_bad(n, p, args.alpha, device)
            _, losses_k = loss_along_path(th_bad_k, th_good, X, Y, args.path_points)
            losses_list.append(losses_k)
        mean_curve = save_path_multi_overlay(ts, losses_list, os.path.join(args.outdir, "path_multi_runs.png"))
        print(f"Saved multi-run png to {os.path.join(args.outdir, 'path_multi_runs.png')}")
        print(f"Mean max loss along path: {float(mean_curve.max()):.4f}")
