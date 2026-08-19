import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join('.', 'results', 'v4')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
DATA_PATH = os.path.join('.', 'data', 'Ihuaenyi_concentration_data.csv')
STRAIN_DATA_PATH = os.path.join('.', 'data', 'Ihuaenyi_strain_data.csv')
IDENTIFIED_J0_PATH = os.path.join('.', 'data', 'identified_j0_curve.csv')

TIMES = [0, 16, 30, 43, 63, 71]


LOSS_TERM_LABELS = {
    'c_loss': 'Data loss (c)',
    'e11_loss': 'Data loss (e_11)',
    'e22_loss': 'Data loss (e_22)',
    'e12_loss': 'Data loss (e_12)',
    'allen_cahn_loss': 'Physics loss (Allen-Cahn)',
    'force_balance_loss': 'Physics loss (force balance)',
    'k_reg_loss': 'k regularizer (ln k)',
    'interp_c_loss': 'Interp. pseudo-data loss (c)',
    'interp_e11_loss': 'Interp. pseudo-data loss (e_11)',
    'interp_e22_loss': 'Interp. pseudo-data loss (e_22)',
    'interp_e12_loss': 'Interp. pseudo-data loss (e_12)',
}


def plot_loss_history(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df = pd.read_csv(os.path.join(results_dir, 'loss_history.csv'))
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label in LOSS_TERM_LABELS.items():
        if col in df.columns:
            ax.semilogy(df['step'], df[col], label=label)
    ax.semilogy(df['step'], df['total_loss'], label='Total loss', linestyle='--', color='black')
    ax.set_xlabel('Step (1 pt/epoch, Adam only)')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'loss_history.png'), dpi=150)
    plt.close(fig)


TRAIN_TEST_TERMS = ['c', 'e11', 'e12', 'e22']


def plot_train_test_loss(term, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    if term not in TRAIN_TEST_TERMS:
        raise ValueError('Invalid term requested')
    df = pd.read_csv(os.path.join(results_dir, 'loss_history.csv'))
    train_col = f'{term}_loss'
    test_col = f'test_{term}_loss'
    if test_col not in df.columns:
        print(f'{test_col} not in loss_history.csv -- skipping train/test plot for {term}')
        return
    label = LOSS_TERM_LABELS.get(train_col, term)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(df['step'], df[train_col], label=f'{label} (train)')
    ax.semilogy(df['step'], df[test_col], label=f'{label} (test)', linestyle='--')
    ax.set_xlabel('Step (1 pt/epoch, Adam only)')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f'{term}_train_test_loss.png'), dpi=150)
    plt.close(fig)


def plot_lambda_history(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df = pd.read_csv(os.path.join(results_dir, 'lambda_history.csv'))
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in df.columns:
        if col.endswith('_lambda'):
            ax.plot(df['step'], df[col], label=col.replace('_lambda', ''))
    ax.set_xlabel('Step (1 pt/epoch, Adam only)')
    ax.set_ylabel('Lambda (loss weight)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'lambda_history.png'), dpi=150)
    plt.close(fig)


def plot_pointwise_error(plot_val, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    if plot_val not in ['c', 'e11', 'e12', 'e22']:
        raise ValueError('Invalid plot value requested')
    error_file = 'pointwise_c_error.csv' if plot_val == 'c' else 'pointwise_e_error.csv'
    df = pd.read_csv(os.path.join(results_dir, error_file))
    prefix = f'{plot_val}_abs_error_t'
    error_cols = [c for c in df.columns if c.startswith(prefix)]
    n = len(error_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, col in enumerate(error_cols):
        r, c = divmod(idx, ncols)
        ax = axs[r][c]
        sc = ax.scatter(df['x'], df['y'], c=df[col], cmap='hot_r', s=20)
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f'|{plot_val}_pred − {plot_val}_true|')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r][c].set_visible(False)

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f'pointwise_{plot_val}_error.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

def plot_pointwise_error_combined(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df_c = pd.read_csv(os.path.join(results_dir, 'pointwise_c_error.csv'))
    df_e = pd.read_csv(os.path.join(results_dir, 'pointwise_e_error.csv'))

    panels = [
        ('c', df_c, 'c_abs_error_t71', '|c_pred − c_true|'),
        ('e11', df_e, 'e11_abs_error_t71', '|e11_pred − e11_true|'),
        ('e12', df_e, 'e12_abs_error_t71', '|e12_pred − e12_true|'),
        ('e22', df_e, 'e22_abs_error_t71', '|e22_pred − e22_true|'),
    ]

    fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
    for ax, (name, df, col, cbar_label) in zip(axs[0], panels):
        sc = ax.scatter(df['x'], df['y'], c=df[col], cmap='hot_r', s=20)
        ax.set_title(f'Pointwise Abs Error: {name} (t = 71)')
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'pointwise_error_combined.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_pred(field, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    if field not in ('c', 'e_11', 'e_12', 'e_22', 'k', 'j_0'):
        raise ValueError('Invalid field requested')
    df = pd.read_csv(os.path.join(results_dir, f'pred_{field}.csv'))
    field_cols = [c for c in df.columns if c.startswith('pred_')]
    n = len(field_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    vmin = df[field_cols].to_numpy().min()
    vmax = df[field_cols].to_numpy().max()

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, col in enumerate(field_cols):
        r, c = divmod(idx, ncols)
        ax = axs[r][c]
        sc = ax.scatter(df['x'], df['y'], c=df[col], cmap='viridis', s=20,
                         vmin=vmin, vmax=vmax)
        t_label = col.replace(f'pred_{field}', 't = ')
        ax.set_title(f'Predicted {field}(x, y) ({t_label})')
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f'{field}(x, y)')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r][c].set_visible(False)

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f'pred_{field}.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

def plot_gt_c(plots_dir=PLOTS_DIR):
    data_path = os.path.join('.', 'data', 'Ihuaenyi_concentration_data.csv')
    df = pd.read_csv(data_path)
    times = [0, 16, 30, 43, 63, 71]
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    c_cols = df.iloc[:, 2:2+len(times)]
    vmin = c_cols.to_numpy().min()
    vmax = c_cols.to_numpy().max()

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for i in range(2):
        for j in range(3):
            z = df.iloc[:, i*3+2+j]
            t = times[i*3 + j]
            sc = axs[i][j].scatter(x, y, c=z, cmap='viridis', s=20,
                                    vmin=vmin, vmax=vmax)
            axs[i][j].set_title(f'Ground Truth c(x, y) (t = {t})')
            axs[i][j].set_xlabel('X [μm]')
            axs[i][j].set_ylabel('Y [μm]')
            cbar = fig.colorbar(sc, ax=axs[i][j])
            cbar.set_label('c(x, y)')

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'true_c.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def load_strain_gt(data_path=STRAIN_DATA_PATH):
    df = pd.read_csv(data_path, skiprows=2, header=None)
    cols = ['x', 'y', '_sep0']
    for comp in ('e11', 'e22', 'e12'):
        cols += [f'{comp}_t{t}' for t in TIMES]
        cols.append(f'_sep_{comp}')
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    return df


def _plot_comparison_grid(x_gt, y_gt, gt_cols_data, x_pred, y_pred, pred_cols_data,
                           times, cbar_label, save_path):
    vmin = min(gt_cols_data.min().min(), pred_cols_data.min().min())
    vmax = max(gt_cols_data.max().max(), pred_cols_data.max().max())

    n = len(times)
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
    for j, t in enumerate(times):
        sc = axs[0][j].scatter(x_gt, y_gt, c=gt_cols_data.iloc[:, j],
                                cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        axs[1][j].scatter(x_pred, y_pred, c=pred_cols_data.iloc[:, j],
                           cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        axs[0][j].set_title(f't = {t}')
        for ax in (axs[0][j], axs[1][j]):
            ax.set_xlabel('X [μm]')
    axs[0][0].set_ylabel('Ground Truth\nY [μm]')
    axs[1][0].set_ylabel('Predicted\nY [μm]')

    fig.tight_layout(rect=(0, 0, 0.95, 1))
    cbar_ax = fig.add_axes((0.96, 0.15, 0.015, 0.7))
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(cbar_label)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_c_comparison(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR, data_path=DATA_PATH):
    df_gt = pd.read_csv(data_path)
    df_pred = pd.read_csv(os.path.join(results_dir, 'pred_c.csv'))

    gt_cols = [f'C [t={t}]' for t in TIMES]
    pred_cols = [f'pred_c{t}' for t in TIMES]

    _plot_comparison_grid(
        df_gt.iloc[:, 0], df_gt.iloc[:, 1], df_gt[gt_cols],
        df_pred['x'], df_pred['y'], df_pred[pred_cols],
        TIMES, 'c(x, y)', os.path.join(plots_dir, 'c_comparison.png'),
    )


def plot_e_comparison(component, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR,
                       data_path=STRAIN_DATA_PATH):
    if component not in ('11', '12', '22'):
        raise ValueError('Invalid component requested')
    df_gt = load_strain_gt(data_path)
    df_pred = pd.read_csv(os.path.join(results_dir, f'pred_e_{component}.csv'))

    gt_cols = [f'e{component}_t{t}' for t in TIMES]
    pred_cols = [f'pred_e_{component}{t}' for t in TIMES]

    _plot_comparison_grid(
        df_gt['x'], df_gt['y'], df_gt[gt_cols],
        df_pred['x'], df_pred['y'], df_pred[pred_cols],
        TIMES, f'e_{component}(x, y)', os.path.join(plots_dir, f'e_{component}_comparison.png'),
    )


def plot_identified_j0_curve(data_path=IDENTIFIED_J0_PATH, plots_dir=PLOTS_DIR):
    df = pd.read_csv(data_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(df['conc'], df['j0'], s=3)
    axs[0].set_xlabel('Concentration c')
    axs[0].set_ylabel('j0(c)')
    axs[0].grid(True, which='both', alpha=0.3)

    axs[1].scatter(df['conc'], df['normalized_j0'], s=3, color='tab:orange')
    axs[1].set_xlabel('Concentration c')
    axs[1].set_ylabel('Normalized j0(c)')
    axs[1].grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'identified_j0_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_k_spatial(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df = pd.read_csv(os.path.join(results_dir, 'pred_k.csv'))
    field_cols = [c for c in df.columns if c.startswith('pred_k')]

    z = df[field_cols[0]]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df['x'], df['y'], c=z, cmap='viridis', s=25)
    ax.set_title('Predicted k(x, y)')
    ax.set_xlabel('X [μm]')
    ax.set_ylabel('Y [μm]')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('k(x, y)')

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'pred_k_spatial.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_over_c(field, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    if field not in ('k', 'j_0'):
        raise ValueError('Invalid field requested')
    df_c = pd.read_csv(DATA_PATH)
    df_field = pd.read_csv(os.path.join(results_dir, f'pred_{field}.csv'))
    slices = [
        pd.DataFrame({
            'concentration': df_c[f'C [t={time}]'],
            f'{field}(c)': df_field[f'pred_{field}{time}'],
        })
        for time in TIMES
    ]
    df_over_c = pd.concat(slices, ignore_index=True)

    df_over_c.to_csv(os.path.join(results_dir, f'{field}_over_c.csv'), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(df_over_c['concentration'], df_over_c[f'{field}(c)'], s=10, alpha=0.4)
    ax.set_xlabel('Concentration')
    ax.set_ylabel(f'{field}(c)')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f'{field}_over_c.png'), dpi=150)
    plt.close(fig)



if __name__ == '__main__':
    plot_loss_history()
    plot_lambda_history()
    plot_train_test_loss('c')
    plot_train_test_loss('e11')
    plot_train_test_loss('e12')
    plot_train_test_loss('e22')
    plot_pointwise_error('c')
    plot_pointwise_error('e11')
    plot_pointwise_error('e12')
    plot_pointwise_error('e22')
    plot_gt_c()
    plot_pred('c')
    plot_pred('e_11')
    plot_pred('e_12')
    plot_pred('e_22')
    plot_pred('k')
    plot_pred('j_0')
    plot_over_c('j_0')
    plot_c_comparison()
    plot_e_comparison('11')
    plot_e_comparison('12')
    plot_e_comparison('22')
    plot_pointwise_error_combined()
    plot_identified_j0_curve()
    plot_k_spatial()
