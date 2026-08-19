import os

import plot_v4 as pv

RESULTS_DIR = os.path.join('.', 'results', 'v4_dropout')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


if __name__ == '__main__':
    pv.plot_loss_history(RESULTS_DIR, PLOTS_DIR)
    pv.plot_lambda_history(RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('c', RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('e11', RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('e12', RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('e22', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pointwise_error('c', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pointwise_error('e11', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pointwise_error('e12', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pointwise_error('e22', RESULTS_DIR, PLOTS_DIR)
    pv.plot_gt_c(PLOTS_DIR)
    pv.plot_pred('c', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pred('e_11', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pred('e_12', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pred('e_22', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pred('k', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pred('j_0', RESULTS_DIR, PLOTS_DIR)
    pv.plot_over_c('j_0', RESULTS_DIR, PLOTS_DIR)
    pv.plot_c_comparison(RESULTS_DIR, PLOTS_DIR)
    pv.plot_e_comparison('11', RESULTS_DIR, PLOTS_DIR)
    pv.plot_e_comparison('12', RESULTS_DIR, PLOTS_DIR)
    pv.plot_e_comparison('22', RESULTS_DIR, PLOTS_DIR)
    pv.plot_pointwise_error_combined(RESULTS_DIR, PLOTS_DIR)
    pv.plot_identified_j0_curve(pv.IDENTIFIED_J0_PATH, PLOTS_DIR)
    pv.plot_k_spatial(RESULTS_DIR, PLOTS_DIR)
