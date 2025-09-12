import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


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
    coefficients[np.abs(coefficients) < tol] = 0  # Set small coefficients to zero (numerical errors)
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
def create_full_rank(w, lin_dep, seed=1):
    full_rank = w.copy()
    rank = np.min([full_rank.shape[0], full_rank.shape[1]])
    tries = 10000
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        for idx in lin_dep:
            full_rank[idx] = rng.standard_normal(w.shape[1])
        if np.linalg.matrix_rank(full_rank) == rank:
            return full_rank
    raise ValueError("Failed to create a full rank matrix after multiple attempts.")


def train(model, x, y, epochs=1000, learning_rate=0.01, tol=1e-6):
    model.U, dep = set_u(model.W, model.U)  # Set U based on W (such that the model stays unchanged)
    model.W = create_full_rank(model.W, dep)  # Ensure W is full rank
    weight_history = []
    loss_history = []
    prev_loss = float('inf')
    for epoch in range(epochs):
        hidden, output = model.forward(x)
        loss = model.loss(x, y)
        grad_u = np.dot((output - y), hidden.T) / x.shape[1]
        model.U -= learning_rate * grad_u
        weight_history.append(model.U.flatten().copy())
        loss_history.append(loss)
        if epoch % 25 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
        # Early stopping condition
        # if abs(prev_loss - loss) < tol:
        #     print(f"Stopping early at epoch {epoch}, change in loss < {tol}")
        #     break
        # prev_loss = loss
    return model.W, model.U, weight_history, loss_history


def plot_loss_landscape(model, x, y, weight_history):
    # PCA on parameter deltas
    weights = np.array(weight_history)
    deltas = np.diff(weights, axis=0)
    pca = PCA(n_components=2)
    pca.fit(deltas)
    directions = pca.components_

    # center the path at the final point
    final_w = weights[-1]
    coords = (weights - final_w) @ directions.T  # CHANGED: removed 0.5*

    # build a grid symmetric around (0,0)
    # choose radii that cover the whole path; keep plot centered at origin
    margin = 1.0
    rad_x = float(np.max(np.abs(coords[:, 0]))) + margin
    rad_y = float(np.max(np.abs(coords[:, 1]))) + margin
    # tiny guards in case the path collapses to a point/line (chatgpt suggested)
    if rad_x == 0: rad_x = 1e-3
    if rad_y == 0: rad_y = 1e-3
    gx = np.linspace(-rad_x, rad_x, 150)
    gy = np.linspace(-rad_y, rad_y, 150)
    xx, yy = np.meshgrid(gx, gy)

    # evaluate loss on the plane around the final point
    zz = np.zeros_like(xx)
    U_shape = model.U.shape
    U_backup = model.U.copy()
    try:
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                delta = xx[i, j] * directions[0] + yy[i, j] * directions[1]
                u_perturbed = final_w + delta
                model.U = u_perturbed.reshape(U_shape)
                zz[i, j] = model.loss(x, y)
    finally:
        model.U = U_backup

    # 2D contour lines + optimization path
    fig, ax = plt.subplots(figsize=(8, 6))
    zmin, zmax = float(np.min(zz)), float(np.max(zz))
    levels = np.linspace(zmin, zmax, 20) if zmax > zmin else [zmin]
    CS = ax.contour(xx, yy, zz, levels=levels, colors='k', linewidths=0.9)
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.3g")

    # optimization path in the same PCA plane (final at (0,0))
    ax.plot(coords[:, 0], coords[:, 1], color='red', lw=1.8, marker='o',
            markersize=3, label='Optimization Path')
    ax.scatter(0.0, 0.0, s=70, color='green', label='Final (center)')

    ax.set_title("Loss contours (PCA plane) with optimization path")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(frameon=False)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def plot_loss(l_history):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(l_history[:100])), l_history[:100], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.yscale('log')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss_contour_on_ax(ax, model, x, y, weight_history, levels=18, grid=140, limits=None):
    """
    draw 2D loss contours in the run's PCA plane, centered at final U.
    """
    # PCA directions from step vectors
    Whist = np.asarray(weight_history)
    deltas = np.diff(Whist, axis=0)
    pca = PCA(n_components=2).fit(deltas)
    dirs = pca.components_
    var = pca.explained_variance_ratio_
    pc1_pct = 100.0 * (var[0] if var.size > 0 else 0.0)
    pc2_pct = 100.0 * (var[1] if var.size > 1 else 0.0)

    # project path so the final point is (0,0)
    final_w = Whist[-1]
    coords = (Whist - final_w) @ dirs.T

    # symmetric grid limits
    if limits is None:
        rx = float(np.abs(coords[:, 0]).max()) or 1e-3
        ry = float(np.abs(coords[:, 1]).max()) or 1e-3
    else:
        rx, ry = limits

    gx = np.linspace(-rx, rx, grid)
    gy = np.linspace(-ry, ry, grid)
    xx, yy = np.meshgrid(gx, gy)

    # evaluate loss (vary U only; keep W fixed)
    zz = np.zeros_like(xx)
    U_backup, U_shape = model.U.copy(), model.U.shape
    try:
        for i in range(xx.shape[0]):
            row_deltas = xx[i, :, None] * dirs[0] + yy[i, :, None] * dirs[1]
            for j in range(xx.shape[1]):
                model.U = (final_w + row_deltas[j]).reshape(U_shape)
                zz[i, j] = model.loss(x, y)
    finally:
        model.U = U_backup

    # contours + path
    zmin, zmax = float(zz.min()), float(zz.max())
    levels_arr = np.linspace(zmin, zmax, levels) if zmax > zmin else [zmin]
    CS = ax.contour(xx, yy, zz, levels=levels_arr, cmap='viridis', linewidths=0.9)
    ax.clabel(CS, inline=True, fontsize=7, fmt="%.3g")

    ax.plot(coords[:, 0], coords[:, 1], color='blue', lw=1.4, marker='o', markersize=2)
    ax.scatter(0.0, 0.0, s=30, color='green')  # final = center

    ax.set_xlim(-rx, rx);
    ax.set_ylim(-ry, ry)  # enforce identical view
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f"1st PCA component: {pc1_pct:.2f} %")
    ax.set_ylabel(f"2nd PCA component: {pc2_pct:.2f} %")
    ax.tick_params(labelsize=8)


def plot_contours_grid(nets, x, y, weight_histories, filename="loss_contours_2x2.png"):
    """
    Make a 2×2 grid (4 runs), all panels with identical limits and tight spacing.
    Saves to the current directory.
    """
    # pick up to 4 runs in a deterministic order
    run_ids = sorted(weight_histories.keys())[:4]

    # do per run PCA to find common plot limits
    radii = []
    for rid in run_ids:
        Whist = np.asarray(weight_histories[rid])
        pca = PCA(n_components=2).fit(np.diff(Whist, axis=0))
        dirs = pca.components_
        final_w = Whist[-1]
        coords = (Whist - final_w) @ dirs.T
        rx = float(np.abs(coords[:, 0]).max()) or 1e-3
        ry = float(np.abs(coords[:, 1]).max()) or 1e-3
        radii.append((rx, ry))

    rx_max = max(r[0] for r in radii)
    ry_max = max(r[1] for r in radii)
    common_limits = (rx_max, ry_max)

    # uild grid figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.9, bottom=0.1,
                        wspace=0.08, hspace=0.25)  # <-- reduced row gap

    axes = axes.ravel()
    for i, rid in enumerate(run_ids):
        ax = axes[i]
        plot_loss_contour_on_ax(ax, nets[rid], x, y, weight_histories[rid],
                                levels=18, grid=140, limits=common_limits)
        r = int(rid) + 1
        ax.set_title(f"Initialization {r}", fontsize=10, pad=3)

    # hide any unused panels (chat)
    for j in range(len(run_ids), 4):
        axes[j].axis('off')

    fig.suptitle("Loss contours and optimization paths in the overparametrized regime (W fixed across runs)",
                 fontsize=12)
    fig.savefig(filename, dpi=300)
    print(f"Saved {filename}")


def plot_loss_curves(loss_histories, filename="loss_vs_epoch_stacked.png"):
    """
    loss_histories: dict[key -> 1D array-like of losses per epoch]
                    The over-parameterized run is assumed to be the largest key,
                    unless you pass over_key explicitly.
    """
    # pick which key is "over"
    keys = list(loss_histories.keys())
    over_key = max(keys)
    under_keys = [k for k in keys if k != over_key]

    # Make figure (stacked) with shared x-axis
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3, 6.2), sharex=True)

    # top: underparameterized runs
    global_min = np.inf
    p_list = [2, 5, 9]
    for i, k in enumerate(under_keys, 1):
        losses = np.asarray(loss_histories[k], dtype=float)
        ax_top.plot(np.arange(len(losses)), losses, lw=1.7, label=f"p = {p_list[i - 1]}")
        m = np.nanmin(losses)
        if m < global_min:
            global_min = m

    # Dotted line at the minimum under loss
    if np.isfinite(global_min):
        ax_top.axhline(global_min, linestyle=":", lw=1.6, color="k",
                       label=f"min = {global_min:.3g}")
    ax_top.set_yscale('log')
    ax_top.set_ylabel("Training loss (log scale)")
    ax_top.set_title("Underparameterized (multiple runs)", fontsize=10)
    ax_top.legend(frameon=False, fontsize=9)
    ax_top.tick_params(labelsize=8)

    # bottom: overparameterized run
    over_losses = np.asarray(loss_histories[over_key], dtype=float)
    # set loss under tolerance to 0
    over_losses[over_losses < 1e-8] = 0.0  # only for better visualization
    min = np.nanmin(over_losses)
    ax_bot.plot(np.arange(len(over_losses)), over_losses, lw=2.2, label="p = 15")
    ax_bot.axhline(min, linestyle=":", lw=1.6, color="k", label=f"min = {min:.3g}")
    ax_bot.set_xlabel("Epoch")
    ax_bot.set_ylabel("Training loss (standard scale)")
    ax_bot.set_title("Overparameterized", fontsize=10)
    ax_bot.legend(frameon=False, fontsize=9)
    ax_bot.tick_params(labelsize=8)

    # tight layout for latex
    fig.suptitle("Loss curves in under- and overparameterized regimes")
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.1, top=0.9, hspace=0.2)
    fig.savefig(filename, dpi=400, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved {filename}")
