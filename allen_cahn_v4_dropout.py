import os
import torch
import torch.nn as nn

import allen_cahn_v4 as v4

# little test script to see if dropout masking helps
# only apply dropout to netA, 10% p on inner layers

MODEL_DIR = os.path.join('.', 'models', 'v4_dropout')
RESULTS_DIR = os.path.join('.', 'results', 'v4_dropout')

DROPOUT_P = 0.1


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

    adam_train_loader, test_loader = v4.load_data()

    netA = PINNDropout(in_channels=v4.IN_CH_ONE, out_channels=v4.OUT_CH_ONE, dropout_p=DROPOUT_P)
    netB = v4.PINN(in_channels=v4.IN_CH_TWO, out_channels=v4.OUT_CH_TWO)
    netC = v4.PINN(in_channels=v4.IN_CH_THREE, out_channels=v4.OUT_CH_THREE)

    pinn = v4.AllenCahnPINN(netA, netB, netC).to(v4.device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (best_loss, per_step_total, per_step_terms, per_step_lambdas, per_step_test_total,
     per_step_test_terms, adam_epochs_run) = v4.train_pinn_adam_only(
        model=pinn, adam_loader=adam_train_loader, interp_data_loader=None,
        lambdas=v4.DEFAULT_LAMBDAS, adam_epochs=v4.NEPOCHS_ADAM,
        lr=v4.LR, model_dir=MODEL_DIR, test_loader=test_loader,
        disable_early_stop=True,
    )

    v4.save_loss_csv(per_step_total, per_step_terms, per_step_lambdas, results_dir=RESULTS_DIR,
                      adam_epochs=adam_epochs_run, test_total=per_step_test_total,
                      test_terms=per_step_test_terms)

    pinn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    v4.eval_pinn(pinn, test_loader, v4.DEFAULT_LAMBDAS)

    v4.save_pointwise_c_error(pinn, results_dir=RESULTS_DIR)
    v4.save_pointwise_e_error(pinn, results_dir=RESULTS_DIR)
    for i in range(6):
        v4.save_preds(pinn, i, results_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()
