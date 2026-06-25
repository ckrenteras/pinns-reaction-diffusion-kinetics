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
    ax.set_ylabel('Loss')
    #ax.set_title('Per-Epoch Loss: Allen–Cahn PINN v1')
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
        sc = ax.scatter(df['x'], df['y'], c=df[col], cmap='hot_r', s=20)
        t_label = col.replace('abs_error_t', 't = ')
        #ax.set_title(f'Pointwise Abs Error ({t_label})')
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('|c_pred − c_true|')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r][c].set_visible(False)

    #fig.suptitle('Pointwise Absolute Error: Allen–Cahn PINN v1', y=1.02)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'pointwise_error.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

def plot_pred(field, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    if field not in ('c', 'k', 'r'):
        raise ValueError('Invalid field requested')
    df = pd.read_csv(os.path.join(results_dir, f'pred_{field}.csv'))
    field_cols = [c for c in df.columns if c.startswith('pred_')]
    n = len(field_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, col in enumerate(field_cols):
        r, c = divmod(idx, ncols)
        ax = axs[r][c]
        sc = ax.scatter(df['x'], df['y'], c=df[col], cmap='viridis', s=20)
        t_label = col.replace(f'pred_{field}', 't = ')
        ax.set_title(f'Predicted {field}(x, y) ({t_label})')
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f'{field}(x, y)')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r][c].set_visible(False)

    #fig.suptitle(f'Predicted {field}: Allen–Cahn PINN v1', y=1.02)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f'pred_{field}.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

def plot_gt_c(plots_dir=PLOTS_DIR):
    data_path = os.path.join('.', 'data', 'Ihuaenyi_concentration_data.csv')
    df = pd.read_csv(data_path)
    times = [0, 16, 32, 43, 63, 71]
    x = df.iloc[:,0]
    y = df.iloc[:,1]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for i in range(2):
        for j in range(3):
            z = df.iloc[:, i*3+2+j]
            t = times[i*3 + j]
            sc = axs[i][j].scatter(x, y, c=z, cmap='viridis', s=20)
            axs[i][j].set_title(f'Ground Truth c(x, y) (t = {t})')
            axs[i][j].set_xlabel('X [μm]')
            axs[i][j].set_ylabel('Y [μm]')
            cbar = fig.colorbar(sc, ax=axs[i][j])
            cbar.set_label('c(x, y)')

    #fig.suptitle('Ground Truth c: Allen–Cahn PINN v1', y=1.02)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'true_c.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    


if __name__ == '__main__':
    plot_loss_history()
    plot_pointwise_error()
    plot_gt_c()
    plot_pred('c')
    plot_pred('k')
