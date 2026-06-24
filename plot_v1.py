import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join('.', 'results', 'v1')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


def plot_loss_history(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df = pd.read_csv(os.path.join(results_dir, 'loss_history.csv'))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(df['epoch'], df['data_loss'], label='Data loss')
    ax.semilogy(df['epoch'], df['physics_loss'], label='Physics loss')
    ax.semilogy(df['epoch'], df['total_loss'], label='Total loss', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Per-Epoch Loss: Allen–Cahn PINN v1')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'loss_history.png'), dpi=150)
    plt.close(fig)


def plot_pointwise_error(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df = pd.read_csv(os.path.join(results_dir, 'pointwise_error.csv'))
    error_cols = [c for c in df.columns if c.startswith('abs_error_t')]
    n = len(error_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, col in enumerate(error_cols):
        r, c = divmod(idx, ncols)
        ax = axs[r][c]
        sc = ax.scatter(df['x'], df['y'], c=df[col], cmap='hot_r', s=5)
        t_label = col.replace('abs_error_t', 't = ')
        ax.set_title(f'Pointwise Abs Error ({t_label})')
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('|c_pred − c_true|')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r][c].set_visible(False)

    fig.suptitle('Pointwise Absolute Error: Allen–Cahn PINN v1', y=1.02)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'pointwise_error.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_loss_history()
    plot_pointwise_error()
