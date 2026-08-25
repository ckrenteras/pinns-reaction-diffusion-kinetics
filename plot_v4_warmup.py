import os

import plot_v4 as pv

RESULTS_DIR = os.path.join('.', 'results', 'v4_warmup')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


if __name__ == '__main__':
    pv.pointwise_error_summary(RESULTS_DIR)
    pv.plot_loss_history(RESULTS_DIR, PLOTS_DIR)
    pv.plot_loss_history_no_interp(RESULTS_DIR, PLOTS_DIR)
    pv.plot_lambda_history(RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('c', RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('e11', RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('e12', RESULTS_DIR, PLOTS_DIR)
    pv.plot_train_test_loss('e22', RESULTS_DIR, PLOTS_DIR)
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
    pv.plot_identified_j0_curve(pv.IDENTIFIED_J0_PATH, PLOTS_DIR)
    pv.plot_k_comparison(RESULTS_DIR, PLOTS_DIR)
    pv.plot_k_pointwise_error(RESULTS_DIR, PLOTS_DIR)
    pv.plot_k_pred_grf_grid(RESULTS_DIR, PLOTS_DIR)
    pv.plot_over_c('mu_h', RESULTS_DIR, PLOTS_DIR)
    pv.plot_mu_h_gt(pv.CHEM_DATA_PATH, PLOTS_DIR)
    pv.plot_mu_h_comparison(RESULTS_DIR, PLOTS_DIR, pv.CHEM_DATA_PATH)
    pv.plot_j0_comparison(RESULTS_DIR, PLOTS_DIR, pv.IDENTIFIED_J0_PATH)
    pv.plot_mu_h_pointwise_error(RESULTS_DIR, PLOTS_DIR, pv.CHEM_DATA_PATH)
    pv.plot_j0_pointwise_error(RESULTS_DIR, PLOTS_DIR, pv.IDENTIFIED_J0_PATH)
