import sys, os, io, contextlib
sys.path.insert(0, '/users/ckrenter/Desktop/CRUNCH-summer-2026/pinns-reaction-diffusion-kinetics')
os.chdir('/users/ckrenter/Desktop/CRUNCH-summer-2026/pinns-reaction-diffusion-kinetics')

import torch
import numpy as np
import allen_cahn_v4 as m

OLD_LAMBDAS = {
    'c': 1.0, 'e11': 1.0, 'e22': 1.0, 'e12': 1.0,
    'allen_cahn': 100.0, 'force_balance': 100.0, 'k_reg': 1e-3,
    'interp_c': 0.1, 'interp_e11': 0.1, 'interp_e22': 0.1, 'interp_e12': 0.1,
}
N_EPOCHS = 100
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagnose_collapse_results.txt')

_out_fh = open(OUT_PATH, 'w')


def log(msg=''):
    print(msg)
    _out_fh.write(str(msg) + '\n')
    _out_fh.flush()


def make_model():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    netA = m.PINN(in_channels=m.IN_CH_ONE, out_channels=m.OUT_CH_ONE)
    netB = m.PINN(in_channels=m.IN_CH_TWO, out_channels=m.OUT_CH_TWO)
    netC = m.PINN(in_channels=m.IN_CH_THREE, out_channels=m.OUT_CH_THREE)
    return m.AllenCahnPINN(netA, netB, netC).to(m.device)


def check_output_std(model):
    x_vals_np, y_vals_np = m.load_domain_xy()
    in_data, _, _, _ = m.extract_c_data(m.CONCENTRATION_CSV_PATH, num_test_slices=0)
    slice_in = in_data[:m.NUM_POS]
    inputs = m.normalize_model_input(torch.from_numpy(slice_in).to(m.device))
    model.eval()
    with torch.no_grad():
        out = model(inputs)
    c_std = out[:, 0].std().item()
    e_std = out[:, 1:4].std().item()
    k_std = out[:, 4].std().item()
    return c_std, e_std, k_std


ORIGINAL_REBALANCED_TERM_NAMES = list(m.REBALANCED_TERM_NAMES)
OLD_REBALANCED_TERM_NAMES = ORIGINAL_REBALANCED_TERM_NAMES + ['k_reg']


def run_test(label, lambdas, freeze_mu_res, revert_k_reg_rebalance=False):
    log(f'\n=== {label} ===')
    model = make_model()
    if freeze_mu_res:
        # keep requires_grad=True (compute_term_grad_norms calls autograd.grad over
        # all model.parameters() and errors if one doesn't require grad) but zero its
        # gradient every backward so Adam never actually moves it away from 0
        model.mu_res_raw.register_hook(lambda grad: torch.zeros_like(grad))

    # REBALANCED_TERM_NAMES is a module-level global that compute_term_grad_norms/
    # rebalance_lambdas look up dynamically each call, so reassigning m.REBALANCED_
    # TERM_NAMES from here changes what those functions see on their next call
    m.REBALANCED_TERM_NAMES = (
        OLD_REBALANCED_TERM_NAMES if revert_k_reg_rebalance else ORIGINAL_REBALANCED_TERM_NAMES)

    adam_loader, test_loader = m.load_data()
    interp_loader = m.interp_loader()

    c0, e0, k0 = check_output_std(model)
    log(f'epoch 0 (init) std: c={c0:.5f} e={e0:.5f} k={k0:.5f}')

    # get_mu() prints a debug line on every single call (many per batch); silence
    # stdout during the actual training call so that spam can't blow past any
    # output-size limit, then keep only the lines that matter
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.train_pinn_adam_only(
            model=model, adam_loader=adam_loader, interp_data_loader=interp_loader,
            lambdas=lambdas, adam_epochs=N_EPOCHS, lr=m.LR, model_dir='/tmp/diag_model',
            test_loader=test_loader, disable_early_stop=True,
        )
    for line in buf.getvalue().splitlines():
        if 'units check' not in line:
            log(line)

    c1, e1, k1 = check_output_std(model)
    log(f'epoch {N_EPOCHS} std: c={c1:.5f} e={e1:.5f} k={k1:.5f}')
    log(f'mu_res_raw final: {model.mu_res_raw.item():.6g}')


os.makedirs('/tmp/diag_model', exist_ok=True)

TINY_FORCE_BALANCE_LAMBDAS = dict(OLD_LAMBDAS)
TINY_FORCE_BALANCE_LAMBDAS['force_balance'] = 1e-4

log(f'Writing results to {OUT_PATH}')
run_test('E: force_balance lambda slashed to 1e-4 (everything else current/default)',
         TINY_FORCE_BALANCE_LAMBDAS, freeze_mu_res=False)

_out_fh.close()
