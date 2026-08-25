import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join('.', 'results', 'v4')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
DATA_PATH = os.path.join('.', 'data', 'Ihuaenyi_concentration_data.csv')
STRAIN_DATA_PATH = os.path.join('.', 'data', 'Ihuaenyi_strain_data.csv')
IDENTIFIED_J0_PATH = os.path.join('.', 'data', 'identified_j0_curve.csv')
CHEM_DATA_PATH = os.path.join('.', 'data', 'chem.csv')

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


def plot_loss_history(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR, exclude_prefixes=(),
                       save_name='loss_history.png'):
    df = pd.read_csv(os.path.join(results_dir, 'loss_history.csv'))
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label in LOSS_TERM_LABELS.items():
        if col in df.columns and not col.startswith(exclude_prefixes):
            ax.semilogy(df['step'], df[col], label=label)
    ax.semilogy(df['step'], df['total_loss'], label='Total loss', linestyle='--', color='black')
    ax.set_xlabel('Step (1 pt/epoch, Adam only)')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, save_name), dpi=150)
    plt.close(fig)


def plot_loss_history_no_interp(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    """same as plot_loss_history but omits the interp_* pseudo-data terms, which are
    always the smallest/densest cluster of lines and mostly just add clutter"""
    plot_loss_history(results_dir, plots_dir, exclude_prefixes=('interp_',),
                       save_name='loss_history_no_interp.png')


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

    # gt_cols_data/pred_cols_data share the same underlying point grid and column
    # (time) order, so a straight elementwise diff gives the pointwise error
    abs_error = np.abs(pred_cols_data.to_numpy() - gt_cols_data.to_numpy())
    err_vmax = abs_error.max()

    n = len(times)
    fig, axs = plt.subplots(3, n, figsize=(4 * n, 12), squeeze=False)
    for j, t in enumerate(times):
        sc = axs[0][j].scatter(x_gt, y_gt, c=gt_cols_data.iloc[:, j],
                                cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        axs[1][j].scatter(x_pred, y_pred, c=pred_cols_data.iloc[:, j],
                           cmap='viridis', s=20, vmin=vmin, vmax=vmax)
        err_sc = axs[2][j].scatter(x_pred, y_pred, c=abs_error[:, j],
                                    cmap='hot_r', s=20, vmin=0, vmax=err_vmax)
        axs[0][j].set_title(f't = {t}')
        for ax in (axs[0][j], axs[1][j], axs[2][j]):
            ax.set_xlabel('X [μm]')
    axs[0][0].set_ylabel('Ground Truth\nY [μm]')
    axs[1][0].set_ylabel('Predicted\nY [μm]')
    axs[2][0].set_ylabel('Abs Error\nY [μm]')

    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cbar_ax = fig.add_axes((0.93, 0.38, 0.015, 0.6))
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(cbar_label)

    err_cbar_ax = fig.add_axes((0.93, 0.08, 0.015, 0.22))
    err_cbar = fig.colorbar(err_sc, cax=err_cbar_ax)
    err_cbar.set_label(f'|pred − true| {cbar_label}')

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


def pointwise_error_summary(results_dir=RESULTS_DIR, data_path=DATA_PATH,
                             strain_data_path=STRAIN_DATA_PATH):
    """mean/RMS/max pointwise abs error for c and each strain component, split into
    train slices (all but the last time slice) vs. the held-out test slice. the
    same train/test split used during training (num_test_slices=1, last slice held out)"""
    train_times, test_time = TIMES[:-1], TIMES[-1]

    df_c_gt = pd.read_csv(data_path)
    df_c_pred = pd.read_csv(os.path.join(results_dir, 'pred_c.csv'))
    df_strain_gt = load_strain_gt(strain_data_path)

    fields = {
        'c': (
            df_c_gt[[f'C [t={t}]' for t in TIMES]].to_numpy(),
            df_c_pred[[f'pred_c{t}' for t in TIMES]].to_numpy(),
        ),
    }
    for comp in ('e11', 'e22', 'e12'):
        df_pred = pd.read_csv(os.path.join(results_dir, f'pred_{comp[0]}_{comp[1:]}.csv'))
        fields[comp] = (
            df_strain_gt[[f'{comp}_t{t}' for t in TIMES]].to_numpy(),
            df_pred[[f'pred_{comp[0]}_{comp[1:]}{t}' for t in TIMES]].to_numpy(),
        )

    rows = []
    for name, (gt, pred) in fields.items():
        abs_err = np.abs(pred - gt)
        train_err, train_gt = abs_err[:, :-1], gt[:, :-1]
        test_err, test_gt = abs_err[:, -1], gt[:, -1]
        # omit RMSE % for strain cuz small absolute vals causes percentages to explode
        gt_range = 1.0 if name == 'c' else gt.max() - gt.min()
        row = {
            'field': name,
            'train_mean_abs_error': train_err.mean(),
            'train_rms_error': np.sqrt((train_err ** 2).mean()),
            'train_max_abs_error': train_err.max(),
            'test_mean_abs_error': test_err.mean(),
            'test_rms_error': np.sqrt((test_err ** 2).mean()),
            'test_max_abs_error': test_err.max(),
            'train_nrmse_pct': 100 * np.sqrt((train_err ** 2).mean()) / gt_range,
            'test_nrmse_pct': 100 * np.sqrt((test_err ** 2).mean()) / gt_range,
        }
        # plain pointwise percent error only makes sense for c it's bounded well away
        # from zero (min ~0.013); the strain fields cross zero, so left as NaN for those
        if name == 'c':
            row['train_mean_pct_error'] = 100 * (train_err / train_gt).mean()
            row['test_mean_pct_error'] = 100 * (test_err / test_gt).mean()
        else:
            row['train_mean_pct_error'] = np.nan
            row['test_mean_pct_error'] = np.nan
        rows.append(row)
    df_summary = pd.DataFrame(rows).set_index('field')

    os.makedirs(results_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(results_dir, 'pointwise_error_summary.csv'))
    print(f'\nPointwise error summary ({results_dir}) -- '
          f'train slices t={train_times}, test slice t={test_time}:')
    print(df_summary.to_string(float_format=lambda v: f'{v:.5f}'))
    return df_summary


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


def plot_k_comparison(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df = pd.read_csv(os.path.join(results_dir, 'k_grf_comparison.csv'))
    vmin = min(df['grf_interpolated'].min(), df['pred_k'].min())
    vmax = max(df['grf_interpolated'].max(), df['pred_k'].max())

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].scatter(df['x'], df['y'], c=df['grf_interpolated'], cmap='viridis',
                   s=20, vmin=vmin, vmax=vmax)
    axs[0].set_title('Ground Truth k(x, y)')
    sc = axs[1].scatter(df['x'], df['y'], c=df['pred_k'], cmap='viridis',
                         s=20, vmin=vmin, vmax=vmax)
    axs[1].set_title('Predicted k(x, y)')
    for ax in axs:
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')

    fig.tight_layout(rect=(0, 0, 0.95, 1))
    cbar_ax = fig.add_axes((0.96, 0.15, 0.015, 0.7))
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('k(x, y)')

    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'k_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_k_pointwise_error(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    df = pd.read_csv(os.path.join(results_dir, 'k_grf_comparison.csv'))
    abs_error = (df['pred_k'] - df['grf_interpolated']).abs()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df['x'], df['y'], c=abs_error, cmap='hot_r', s=20)
    ax.set_title('Pointwise Abs Error: k(x, y)')
    ax.set_xlabel('X [μm]')
    ax.set_ylabel('Y [μm]')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('|k_pred − k_true|')

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'pointwise_k_error.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_k_pred_grf_grid(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    """standalone predicted k(x, y), self-scaled to the prediction's own range,
    evaluated on the GRF's own (denser) grid -- companion to plot_pred('k'),
    which does the same thing but on the training/prediction mesh"""
    df = pd.read_csv(os.path.join(results_dir, 'k_grf_comparison.csv'))

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df['x'], df['y'], c=df['pred_k'], cmap='viridis', s=10)
    ax.set_title('Predicted k(x, y) (GRF grid)')
    ax.set_xlabel('X [μm]')
    ax.set_ylabel('Y [μm]')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('k(x, y)')

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'pred_k_grf_grid.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mu_h_gt(data_path=CHEM_DATA_PATH, plots_dir=PLOTS_DIR):
    df = pd.read_csv(data_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['concentration'], df['chemical_potential'])
    ax.set_xlabel('Concentration c')
    ax.set_ylabel('Ground Truth mu_h(c)')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'mu_h_gt.png'), dpi=150)
    plt.close(fig)


def _pool_at_gt_c(results_dir, field):
    """long-format concentration/field(c) pairs pooled across all time slices, using
    the model's evaluation at the GT concentration values (see save_mu_h_j0_at_gt_c
    in the training script) rather than the model's own predicted c"""
    df = pd.read_csv(os.path.join(results_dir, 'mu_h_j0_at_gt_c.csv'))
    slices = [
        pd.DataFrame({
            'concentration': df[f'gt_c{t}'],
            field: df[f'{field}_at_gt_c{t}'],
        })
        for t in TIMES
    ]
    return pd.concat(slices, ignore_index=True)


def plot_mu_h_comparison(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR, data_path=CHEM_DATA_PATH):
    pooled = _pool_at_gt_c(results_dir, 'mu_h')
    gt = pd.read_csv(data_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(pooled['concentration'], pooled['mu_h'], s=8, alpha=0.3, label='Approximated (at GT c)')
    ax.plot(gt['concentration'], gt['chemical_potential'], color='black', linewidth=2, label='Ground truth')
    ax.set_xlim(pooled['concentration'].min() - 0.02, pooled['concentration'].max() + 0.02)
    ax.set_xlabel('Normalized concentration, c')
    ax.set_ylabel('mu_h(c)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'mu_h_comparison.png'), dpi=150)
    plt.close(fig)


def plot_j0_comparison(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR, data_path=IDENTIFIED_J0_PATH):
    pooled = _pool_at_gt_c(results_dir, 'j0')
    gt = pd.read_csv(data_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(pooled['concentration'], pooled['j0'], s=8, alpha=0.3, label='Predicted (at GT c)')
    ax.plot(gt['conc'], gt['j0'], color='black', linewidth=2, label='Identified (ground truth)')
    ax.set_xlabel('Normalized concentration, c')
    ax.set_ylabel('j0(c)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, 'j0_comparison.png'), dpi=150)
    plt.close(fig)


def _pointwise_error_over_time(results_dir, plots_dir, field, gt_conc, gt_field, cbar_label, save_name):
    df = pd.read_csv(os.path.join(results_dir, 'mu_h_j0_at_gt_c.csv'))
    n = len(TIMES)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, t in enumerate(TIMES):
        r, c = divmod(idx, ncols)
        ax = axs[r][c]
        pred = df[f'{field}_at_gt_c{t}'].to_numpy()
        conc = df[f'gt_c{t}'].to_numpy()
        gt_at_conc = np.interp(conc, gt_conc, gt_field)
        abs_error = np.abs(pred - gt_at_conc)
        sc = ax.scatter(df['x'], df['y'], c=abs_error, cmap='hot_r', s=20)
        ax.set_title(f't = {t}')
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axs[r][c].set_visible(False)

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mu_h_pointwise_error(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR, data_path=CHEM_DATA_PATH):
    gt = pd.read_csv(data_path).sort_values('concentration')
    _pointwise_error_over_time(
        results_dir, plots_dir, 'mu_h', gt['concentration'].to_numpy(), gt['chemical_potential'].to_numpy(),
        '|mu_h_pred − mu_h_true|', 'pointwise_mu_h_error.png',
    )


def plot_j0_pointwise_error(results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR, data_path=IDENTIFIED_J0_PATH):
    gt = pd.read_csv(data_path).sort_values('conc')
    _pointwise_error_over_time(
        results_dir, plots_dir, 'j0', gt['conc'].to_numpy(), gt['j0'].to_numpy(),
        '|j0_pred − j0_true|', 'pointwise_j0_error.png',
    )


def plot_over_c(field, results_dir=RESULTS_DIR, plots_dir=PLOTS_DIR):
    if field not in ('k', 'j_0', 'mu_h'):
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
    pointwise_error_summary()
    plot_loss_history()
    plot_loss_history_no_interp()
    plot_lambda_history()
    plot_train_test_loss('c')
    plot_train_test_loss('e11')
    plot_train_test_loss('e12')
    plot_train_test_loss('e22')
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
    plot_identified_j0_curve()
    plot_k_comparison()
    plot_k_pointwise_error()
    plot_k_pred_grf_grid()
    plot_over_c('mu_h')
    plot_mu_h_gt()
    plot_mu_h_comparison()
    plot_j0_comparison()
    plot_mu_h_pointwise_error()
    plot_j0_pointwise_error()
