import os
import numpy as np
import torch

import allen_cahn_v4 as v4

# v4 but warm-up stage added with only data loss for first half of adam training

MODEL_DIR = os.path.join('.', 'models', 'v4_warmup')
RESULTS_DIR = os.path.join('.', 'results', 'v4_warmup')

WARMUP_FRACTION = 0.5

WARMUP_LAMBDAS = dict(v4.DEFAULT_LAMBDAS)
for _name in ['allen_cahn', 'force_balance', 'k_reg'] + v4.INTERP_TERM_NAMES:
    WARMUP_LAMBDAS[_name] = 0.0


def train_pinn_adam_only_warmup(model, adam_loader, lambdas=v4.DEFAULT_LAMBDAS,
                                 warmup_lambdas=WARMUP_LAMBDAS, adam_epochs=v4.NEPOCHS_ADAM,
                                 warmup_fraction=WARMUP_FRACTION, lr=v4.LR, model_dir=MODEL_DIR,
                                 test_loader=None):
    """Data los only for first half of adam training. Can't use RAD collocation 
    because there's no physics loss term"""
    lambdas = dict(lambdas)
    warmup_epochs = int(adam_epochs * warmup_fraction)
    x_vals_np, y_vals_np = v4.load_domain_xy()

    per_epoch_total = np.zeros(adam_epochs)
    per_epoch_terms = {name: np.zeros(adam_epochs) for name in v4.ALL_TERM_NAMES}
    per_epoch_lambdas = {name: np.zeros(adam_epochs) for name in v4.ALL_TERM_NAMES}
    per_epoch_test_total = np.zeros(adam_epochs)
    per_epoch_test_terms = {name: np.zeros(adam_epochs) for name in v4.ALL_TERM_NAMES}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(adam_epochs, 1), eta_min=v4.COSINE_ETA_MIN)

    best_loss = float('inf')
    colloc_pool_np = None
    step_lambdas = warmup_lambdas

    for epoch in range(adam_epochs):
        in_warmup = epoch < warmup_epochs
        iter_total_loss = 0.0
        iter_term_sums = {name: 0.0 for name in v4.ALL_TERM_NAMES}
        n = 0

        if not in_warmup:
            epochs_since_warmup = epoch - warmup_epochs
            if epoch == warmup_epochs or epochs_since_warmup % v4.RAD_RESAMPLE_EVERY == 0:
                colloc_pool_np = v4.adaptive_rad_sample(
                    model, x_vals_np, y_vals_np, v4.RAD_POOL_SIZE, v4.T_DIFF)

        for batch_idx, (inputs, labels) in enumerate(adam_loader):
            inputs, labels = inputs.to(v4.device), labels.to(v4.device)

            if in_warmup:
                collocation_data = None
                step_lambdas = warmup_lambdas
            else:
                colloc_idx = np.random.randint(0, len(colloc_pool_np), size=inputs.shape[0])
                collocation_data = torch.from_numpy(colloc_pool_np[colloc_idx]).to(v4.device)

                if batch_idx == 0 and (epoch == warmup_epochs
                                        or epochs_since_warmup % v4.GRAD_NORM_REBALANCE_EVERY == 0):
                    grad_norms = v4.compute_term_grad_norms(model, inputs, labels, v4.T_DIFF, collocation_data)
                    lambdas = v4.rebalance_lambdas(lambdas, grad_norms)
                step_lambdas = lambdas

            optimizer.zero_grad()
            loss, loss_terms = v4.pinn_loss(
                model,
                in_data=inputs,
                labels=labels,
                t_diff=v4.T_DIFF,
                lambdas=step_lambdas,
                collocation_data=collocation_data)

            loss.backward()
            optimizer.step()
            n += 1
            iter_total_loss += loss.item()
            for name in v4.ALL_TERM_NAMES:
                iter_term_sums[name] += loss_terms[name].item()

        scheduler.step()

        iter_total_loss /= n
        for name in v4.ALL_TERM_NAMES:
            iter_term_sums[name] /= n

        per_epoch_total[epoch] = iter_total_loss
        for name in v4.ALL_TERM_NAMES:
            per_epoch_terms[name][epoch] = iter_term_sums[name]
            per_epoch_lambdas[name][epoch] = step_lambdas[name]

        if test_loader is not None:
            test_total_loss, test_term_sums = v4.eval_pinn(model, test_loader, step_lambdas)
            model.train()
            per_epoch_test_total[epoch] = test_total_loss
            for name in v4.ALL_TERM_NAMES:
                per_epoch_test_terms[name][epoch] = test_term_sums[name]
            ckpt_loss = sum(v4.DEFAULT_LAMBDAS[name] * test_term_sums[name] for name in v4.ALL_TERM_NAMES)
        else:
            ckpt_loss = sum(v4.DEFAULT_LAMBDAS[name] * iter_term_sums[name] for name in v4.ALL_TERM_NAMES)

        if ckpt_loss < best_loss:
            best_loss = ckpt_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

        if epoch % 100 == 0:
            phase = 'warmup (data-only)' if in_warmup else 'joint'
            terms_str = '  | '.join(f'{name}: {iter_term_sums[name]}' for name in v4.ALL_TERM_NAMES)
            lambdas_str = '  | '.join(f'lam_{name}: {step_lambdas[name]:.4g}' for name in v4.ALL_TERM_NAMES)
            print(f'EPOCH: {epoch} [{phase}]  | {terms_str}  | total loss: {iter_total_loss}  '
                  f'| lr: {scheduler.get_last_lr()[0]:.3e}')
            print(f'         {lambdas_str}')
            if test_loader is not None:
                print(f'         test total loss: {per_epoch_test_total[epoch]}')

    return (best_loss, per_epoch_total, per_epoch_terms, per_epoch_lambdas, per_epoch_test_total,
            per_epoch_test_terms, adam_epochs)


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    adam_train_loader, test_loader = v4.load_data()

    netA = v4.PINN(in_channels=v4.IN_CH_ONE, out_channels=v4.OUT_CH_ONE)
    netB = v4.PINN(in_channels=v4.IN_CH_TWO, out_channels=v4.OUT_CH_TWO)
    netC = v4.PINN(in_channels=v4.IN_CH_THREE, out_channels=v4.OUT_CH_THREE)

    pinn = v4.AllenCahnPINN(netA, netB, netC).to(v4.device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (best_loss, per_step_total, per_step_terms, per_step_lambdas, per_step_test_total,
     per_step_test_terms, adam_epochs_run) = train_pinn_adam_only_warmup(
        model=pinn, adam_loader=adam_train_loader,
        lambdas=v4.DEFAULT_LAMBDAS, warmup_lambdas=WARMUP_LAMBDAS,
        adam_epochs=v4.NEPOCHS_ADAM, warmup_fraction=WARMUP_FRACTION,
        lr=v4.LR, model_dir=MODEL_DIR, test_loader=test_loader,
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
