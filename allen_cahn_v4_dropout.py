import os
import numpy as np
import torch
import torch.nn as nn

import allen_cahn_v4 as v4

# little test script to see if dropout masking helps
# only apply dropout to netA, 10% p on inner layers
# Adam stage trains with dropout on; SSBroyden QN stage follows with dropout
# zeroed out (see main()) since the QN line search needs a deterministic closure

MODEL_DIR = os.path.join('.', 'models', 'v4_dropout')
RESULTS_DIR = os.path.join('.', 'results', 'v4_dropout')

DROPOUT_P = 0.01


class PINNDropout(nn.Module):
    def __init__(self, in_channels, out_channels, width=v4.WIDTH, depth=v4.DEPTH, dropout_p=DROPOUT_P):
        super().__init__()
        layers = [nn.Linear(in_channels, width), nn.Tanh(), nn.Dropout(dropout_p)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh(), nn.Dropout(dropout_p)]
        layers.append(nn.Linear(width, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, tau):
        return self.net(tau)


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    adam_train_loader, test_loader = v4.load_data()
    interp_train_loader = v4.interp_loader()

    netA = PINNDropout(in_channels=v4.IN_CH_ONE, out_channels=v4.OUT_CH_ONE, dropout_p=DROPOUT_P)
    netB = v4.PINN(in_channels=v4.IN_CH_TWO, out_channels=v4.OUT_CH_TWO)
    netC = v4.PINN(in_channels=v4.IN_CH_THREE, out_channels=v4.OUT_CH_THREE)

    pinn = v4.AllenCahnPINN(netA, netB, netC).to(v4.device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (best_loss, per_step_total, per_step_terms, per_step_lambdas, per_step_test_total,
     per_step_test_terms, adam_epochs_run, per_step_k_stats) = v4.train_pinn_adam_only(
        model=pinn, adam_loader=adam_train_loader, interp_data_loader=None,
        lambdas=v4.DEFAULT_LAMBDAS, adam_epochs=v4.NEPOCHS_ADAM,
        lr=v4.LR, model_dir=MODEL_DIR, test_loader=test_loader,
        disable_early_stop=True,
    )

    # hand the QN phase Adam's best checkpoint (by the fixed-lambda ckpt_loss)
    # not necessarily its last epoch
    pinn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    adam_final_lambdas = {name: float(per_step_lambdas[name][-1]) for name in v4.ALL_TERM_NAMES}

    # SSBroyden's line search assumes a deterministic closure across its internal
    # re-evaluations; netA's Dropout would make each call stochastic, so zero it
    # out for this stage instead of toggling eval() (train_pinn_qn calls
    # model.train() again after every test-loader pass, which would silently
    # re-enable masking mid-stage if we relied on eval() alone)
    for m in pinn.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    qn_full_loader = v4.full_batch_loader()
    qn_inputs, qn_labels = next(iter(qn_full_loader))
    interp_inputs, interp_labels = next(iter(interp_train_loader))
    x_vals_np, y_vals_np = v4.load_domain_xy()

    (qn_best_loss, qn_per_iter_total, qn_per_iter_terms, qn_per_iter_lambdas,
     qn_per_iter_test_total, qn_per_iter_test_terms, qn_iters_run,
     qn_per_iter_k_stats) = v4.train_pinn_qn(
        model=pinn, qn_inputs=qn_inputs, qn_labels=qn_labels,
        interp_inputs=interp_inputs, interp_labels=interp_labels,
        lambdas=adam_final_lambdas, x_vals_np=x_vals_np, y_vals_np=y_vals_np,
        model_dir=MODEL_DIR, test_loader=test_loader,
    )

    combined_total = np.concatenate([per_step_total, qn_per_iter_total])
    combined_terms = {name: np.concatenate([per_step_terms[name], qn_per_iter_terms[name]])
                       for name in v4.ALL_TERM_NAMES}
    combined_lambdas = {name: np.concatenate([per_step_lambdas[name], qn_per_iter_lambdas[name]])
                         for name in v4.ALL_TERM_NAMES}
    combined_test_total = np.concatenate([per_step_test_total, qn_per_iter_test_total])
    combined_test_terms = {name: np.concatenate([per_step_test_terms[name], qn_per_iter_test_terms[name]])
                            for name in v4.ALL_TERM_NAMES}
    combined_k_stats = {stat: np.concatenate([per_step_k_stats[stat], qn_per_iter_k_stats[stat]])
                         for stat in ('std', 'min', 'max')}

    v4.save_loss_csv(combined_total, combined_terms, combined_lambdas, results_dir=RESULTS_DIR,
                      adam_epochs=adam_epochs_run, test_total=combined_test_total,
                      test_terms=combined_test_terms, k_stats=combined_k_stats)

    pinn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    v4.eval_pinn(pinn, test_loader, v4.DEFAULT_LAMBDAS)

    for i in range(6):
        v4.save_preds(pinn, i, results_dir=RESULTS_DIR)
    v4.save_mu_h_preds(pinn, results_dir=RESULTS_DIR)
    v4.save_k_at_grf_points(pinn, results_dir=RESULTS_DIR)
    v4.save_mu_h_j0_at_gt_c(pinn, results_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()
