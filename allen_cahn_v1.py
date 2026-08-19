import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import scipy
from scipy.optimize import minimize

import pandas as pd

# Global constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_POS = 3130
NUM_T_STATES = 6
TIME_SLICES = [0, 16, 30, 43, 63, 71]
T_DIFF = 71
TAU_SLICES = np.array(TIME_SLICES) / T_DIFF
CONCENTRATION_CSV_PATH = os.path.join('.', 'data', 'Ihuaenyi_concentration_data.csv')
MODEL_DIR = os.path.join('.', 'models')
RESULTS_DIR = os.path.join('.', 'results', 'v1')

PRED_COLS = ['c', 'k', 'j_0']

IN_CH_ONE = 3
IN_CH_TWO = 1
OUT_CH_ONE = 2
OUT_CH_TWO = 1
WIDTH = 32
DEPTH=4

NEPOCHS_ADAM = 1000
NEPOCHS_BFGS = 500
NEPOCHS = NEPOCHS_ADAM + NEPOCHS_BFGS
LR=1e-4

METHOD_BFGS = "SSBroyden1"   
METHOD = "BFGS"   
NCHANGE = 200
INITIAL_SCALE = False
BFGS_BATCH = 15650


n = 1 # TO BE CHANGED!!!
a = 0.5

# datasets and dataloaders

def extract_data(data_path=CONCENTRATION_CSV_PATH, num_test_slices=1):
    '''given path to csv data file extracts concentration data over pos and time
        returns input data and label numpy arrays'''
    df = pd.read_csv(data_path)
    x_vals = df.iloc[:,0]
    y_vals = df.iloc[:,1]
    x_vals_np = x_vals.to_numpy().astype(np.float32)
    y_vals_np = y_vals.to_numpy().astype(np.float32)
    in_data_np = None
    out_data_np = None
    for i in range(NUM_T_STATES - num_test_slices):
        c_vals = df.iloc[:,2+i]
        c_vals_np = c_vals.to_numpy().astype(np.float32)
        t_col = np.zeros_like(x_vals_np)
        t_col[:] = TAU_SLICES[i]
        time_slice_in = np.stack([x_vals_np, y_vals_np, t_col], axis=1)
        time_slice_out = c_vals_np.reshape(-1, 1)
        if in_data_np is not None: 
            in_data_np = np.concatenate([in_data_np, time_slice_in], axis=0)
            out_data_np = np.concatenate([out_data_np, time_slice_out], axis=0)
        else: 
            in_data_np = time_slice_in
            out_data_np = time_slice_out

    test_in_np = None
    test_out_np = None
    for i in range(NUM_T_STATES - num_test_slices, NUM_T_STATES):
        c_vals = df.iloc[:,2+i]
        c_vals_np = c_vals.to_numpy().astype(np.float32)
        t_col = np.zeros_like(x_vals_np)
        t_col[:] = TAU_SLICES[i]
        time_slice_in = np.stack([x_vals_np, y_vals_np, t_col], axis=1)
        time_slice_out = c_vals_np.reshape(-1, 1)
        if test_in_np is not None: 
            test_in_np = np.concatenate([test_in_np, time_slice_in], axis=0)
            test_out_np = np.concatenate([test_out_np, time_slice_out], axis=0)
        else: 
            test_in_np = time_slice_in
            test_out_np = time_slice_out

    return in_data_np, out_data_np, test_in_np, test_out_np

class ConcentrationData(Dataset):
    def __init__(self, data_path=CONCENTRATION_CSV_PATH, split='train', num_test_slices=1):
        train_data, train_labels, test_data, test_labels = extract_data(data_path, num_test_slices)
        self.train_X = torch.from_numpy(train_data)
        self.train_Y = torch.from_numpy(train_labels)
        self.test_X = torch.from_numpy(test_data)
        self.test_Y = torch.from_numpy(test_labels)
        self.split = split
        
    def __len__(self):
        return len(self.train_X) if self.split == 'train' else len(self.test_X)
        
    def __getitem__(self, idx):
        # Retrieve a single sample and its label by index
        if self.split=='train':
            return self.train_X[idx], self.train_Y[idx]
        elif self.split=='test':
            return self.test_X[idx], self.test_Y[idx] 
        else:
            raise ValueError("Split must be either 'train' or 'test'")


def load_concentration_data(data_path=CONCENTRATION_CSV_PATH, num_test_slices=1, batch_size=512):
    # this is currently reading csv twice—fine for now but should be changed later
    pin = device.type == 'cuda'
    train_data = ConcentrationData(data_path, 'train',  num_test_slices)
    test_data = ConcentrationData(data_path, 'test',  num_test_slices)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=pin
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=32, 
        shuffle=False,
    )

    return train_loader, test_loader

def bfgs_loader(data_path=CONCENTRATION_CSV_PATH, num_test_slices=1, batch_size=BFGS_BATCH):
    # this is currently reading csv twice—fine for now but should be changed later
    pin = device.type == 'cuda'
    train_data = ConcentrationData(data_path, 'train',  num_test_slices)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=pin
    )

    return train_loader

# Residuals

def compute_time_derivs(component, tau):
    return torch.autograd.grad(
        component,
        tau,
        grad_outputs=torch.ones_like(component),
        create_graph=True,
        retain_graph=True,
    )[0]

def get_residual(c, k, j_0, tau, t_diff):
    dc_dtau = compute_time_derivs(c, tau)

    # have to divide by t1 - t1 by chain rule to recover time derivs
    dc_dt = dc_dtau / t_diff

    rhs = k * j_0 *(np.exp(-a * n) - np.exp((1-a)*n))
    return dc_dt - rhs

def pinn_loss(
        model,
        in_data,
        c_true,
        t_diff,
        lambda_data,
        lambda_phys,
    ):
    pos = in_data[:, 0:2].clone().detach()
    tau = in_data[:, 2:3].clone().detach().requires_grad_(True)
    model_input = torch.cat([pos, tau], dim=1)
    z_hat = model(model_input)
    c_pred = z_hat[:, 0:1]
    k_pred = z_hat[:, 1:2]
    j_0_pred = z_hat[:, 2:3]
    data_loss = torch.mean((c_pred - c_true) ** 2)
    res = get_residual(c_pred, k_pred, j_0_pred, tau, t_diff)
    physics_loss = torch.mean(res ** 2)


    total_loss = (
        lambda_data * data_loss
        + lambda_phys * physics_loss
    )
    return total_loss, data_loss, physics_loss


def loss_grad_np(
        weights,
        model,
        in_data,
        c_true,
        t_diff,
        lambda_data,
        lambda_phys):
    # for BFGS and other 2nd order optimizers
    first_param = next(model.parameters())
    dtype = first_param.dtype
    w_tensor = torch.as_tensor(weights, dtype=dtype, device=device)

    with torch.no_grad():
        vector_to_parameters(w_tensor, model.parameters())

    total_loss, data_loss, physics_loss = pinn_loss(model,
                                                    in_data,
                                                    c_true,
                                                    t_diff,
                                                    lambda_data,
                                                    lambda_phys)
    
    params = list(model.parameters())
    gradsN = torch.autograd.grad(
        total_loss, params,
        create_graph=False, retain_graph=False, allow_unused=False)
    
    grads_flat = torch.cat([g.reshape(-1) for g in gradsN])
    return float(total_loss.detach().cpu().item()), grads_flat.detach().cpu().numpy()

# PINNs

class PINN(nn.Module):
    def __init__(self, in_channels, out_channels, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_channels, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, tau):
        return self.net(tau)
    
class AllenCahnPINN(nn.Module):
    def __init__(self, netA, netB):
        super().__init__()
        self.netA = netA
        self.netB = netB
    
    def forward(self, data):
        c_k = self.netA(data)
        c = c_k[:, 0:1]
        j_0 = self.netB(c)
        out = torch.cat([c_k, j_0], dim=1)
        return out

 # eval

def eval_pinn(model, data_loader):
    n = 0
    model.eval()
    total_loss = 0.0
    phys_loss = 0.0
    data_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)            
        iter_total_loss, iter_phys_loss, iter_data_loss = pinn_loss(
            model, inputs, labels, T_DIFF, 
            lambda_data=1, lambda_phys=100)
        total_loss += iter_total_loss
        phys_loss += iter_phys_loss
        data_loss += iter_data_loss            
        n += 1       
    total_loss /= n
    phys_loss /= n
    data_loss /= n
    return total_loss.item(), phys_loss.item(), data_loss.item()

# Training


def train_pinn(model, adam_loader, bfgs_loader, lam_data, lam_phys, adam_epochs=NEPOCHS_ADAM,
               bfgs_epochs=NEPOCHS_BFGS, lr=LR, model_dir=MODEL_DIR):
    per_epoch_total = np.zeros(adam_epochs + bfgs_epochs)
    per_epoch_data = np.zeros(adam_epochs + bfgs_epochs)
    per_epoch_phys = np.zeros(adam_epochs + bfgs_epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    for epoch in range(adam_epochs):
        iter_total_loss = 0.0
        iter_data_loss = 0.0
        iter_phys_loss = 0.0
        n = 0
        
        for inputs, labels in adam_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, data_loss, physics_loss = pinn_loss(
                model,
                in_data=inputs,
                c_true=labels,
                t_diff=T_DIFF,
                lambda_data=lam_data,
                lambda_phys=lam_phys
            )
            loss.backward()
            optimizer.step()
            n += 1
            iter_total_loss += loss.item()
            iter_data_loss += data_loss.item()
            iter_phys_loss += physics_loss.item()

        iter_total_loss /= n
        iter_data_loss /= n
        iter_phys_loss /= n

        per_epoch_total[epoch] = iter_total_loss
        per_epoch_data[epoch] = iter_data_loss
        per_epoch_phys[epoch] = iter_phys_loss

        if iter_total_loss < best_loss:
            best_loss = iter_total_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

        if epoch % 100 == 0:
            print(f'EPOCH: {epoch}  | data loss: {iter_data_loss}  | physics loss: {iter_phys_loss}  | total loss: {iter_total_loss}')

    initial_weights = parameters_to_vector(model.parameters()).detach().cpu().numpy()

    # one continuous minimize() call instead of restarting every NCHANGE iterations
    # repeated restarts only symmetrized (not positive-definite always(?)) that carried-over inverse Hessian,
    # setting off  scipy's positive-definite check
    bfgs_inputs, bfgs_labels = next(iter(bfgs_loader))
    bfgs_inputs, bfgs_labels = bfgs_inputs.to(device), bfgs_labels.to(device)
    dtype = next(model.parameters()).dtype

    state = {'best_loss': best_loss, 'iters': 0}

    def callback(intermediate_result):
        state['iters'] += 1
        if state['iters'] % NCHANGE != 0:
            return
        epoch = state['iters'] // NCHANGE - 1

        with torch.no_grad():
            vector_to_parameters(
                torch.as_tensor(intermediate_result.x, dtype=dtype, device=device),
                model.parameters())
        __, iter_phys_loss, iter_data_loss = eval_pinn(model, bfgs_loader)
        iter_total_loss = float(intermediate_result.fun)

        per_epoch_data[NEPOCHS_ADAM + epoch] = iter_data_loss
        per_epoch_phys[NEPOCHS_ADAM + epoch] = iter_phys_loss
        per_epoch_total[NEPOCHS_ADAM + epoch] = iter_total_loss

        if iter_total_loss < state['best_loss']:
            state['best_loss'] = iter_total_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        if epoch % 10 == 0:
            print(f'EPOCH: {NEPOCHS_ADAM + epoch}  | data loss: {iter_data_loss}  | physics loss: {iter_phys_loss}  | total loss: {iter_total_loss}')

    result = minimize(
        loss_grad_np,
        initial_weights,
        args=(model, bfgs_inputs, bfgs_labels, T_DIFF, lam_data, lam_phys),
        method=METHOD,
        jac=True,
        callback=callback,
        options={
            'maxiter': bfgs_epochs * NCHANGE,
            'gtol': 1e-8,
            'method_bfgs': METHOD_BFGS,
            'initial_scale': INITIAL_SCALE})

    print(f'BFGS finished: success={result.success}  message={result.message}  '
          f'iters={result.nit}/{bfgs_epochs * NCHANGE}')

    with torch.no_grad():
        vector_to_parameters(
            torch.as_tensor(result.x, dtype=dtype, device=device),
            model.parameters())
    best_loss = state['best_loss']

    return best_loss, per_epoch_total, per_epoch_data, per_epoch_phys


# plotting

def save_loss_csv(per_epoch_total, per_epoch_data, per_epoch_phys, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    epochs = np.arange(len(per_epoch_total))
    df = pd.DataFrame({
        'epoch': epochs,
        'data_loss': per_epoch_data,
        'physics_loss': per_epoch_phys,
        'total_loss': per_epoch_total,
    })
    df.to_csv(os.path.join(results_dir, 'loss_history.csv'), index=False)


def save_pointwise_error(model, num_test_slices=1, data_path=CONCENTRATION_CSV_PATH,
                         results_dir=RESULTS_DIR):
    _, _, test_in_np, test_out_np = extract_data(data_path, num_test_slices)
    x = test_in_np[:NUM_POS, 0]
    y = test_in_np[:NUM_POS, 1]

    model.eval()
    cols = {'x': x, 'y': y}
    with torch.no_grad():
        for i in range(num_test_slices):
            slice_in = test_in_np[i * NUM_POS:(i + 1) * NUM_POS]
            slice_true = test_out_np[i * NUM_POS:(i + 1) * NUM_POS].squeeze()
            inputs = torch.from_numpy(slice_in).to(device)
            c_pred = model(inputs)[:, 0].cpu().numpy()
            t_val = TIME_SLICES[NUM_T_STATES - num_test_slices + i]
            cols[f'abs_error_t{t_val}'] = np.abs(c_pred - slice_true)

    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(cols).to_csv(os.path.join(results_dir, 'pointwise_error.csv'), index=False)


def save_preds(model, field_idx, data_path=CONCENTRATION_CSV_PATH,
                         results_dir=RESULTS_DIR):
    if field_idx not in (0, 1, 2):
        raise ValueError('Invalid field index')
    in_data, _, _, _ = extract_data(data_path, num_test_slices=0)
    x = in_data[:NUM_POS, 0]
    y = in_data[:NUM_POS, 1]
    pred_field = PRED_COLS[field_idx]

    model.eval()
    cols = {'x': x, 'y': y}
    with torch.no_grad():
        for i in range(NUM_T_STATES):
            slice_in = in_data[i * NUM_POS:(i + 1) * NUM_POS]
            inputs = torch.from_numpy(slice_in).to(device)
            pred = model(inputs)[:, field_idx].cpu().numpy()
            t_val = TIME_SLICES[i]
            cols[f'pred_{pred_field}{t_val}'] = pred

    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(cols).to_csv(os.path.join(results_dir, f'pred_{pred_field}.csv'), index=False)

def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    adam_train_loader, test_loader = load_concentration_data()
    bfgs_train_loader = bfgs_loader()

    netA = PINN(in_channels=IN_CH_ONE, out_channels=OUT_CH_ONE)
    netB = PINN(in_channels=IN_CH_TWO, out_channels=OUT_CH_TWO)

    pinn = AllenCahnPINN(netA, netB).to(device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_loss, per_epoch_total, per_epoch_data, per_epoch_phys = train_pinn(
        model=pinn, adam_loader=adam_train_loader, bfgs_loader=bfgs_train_loader,
        lam_data=1, lam_phys=100, adam_epochs=NEPOCHS_ADAM, bfgs_epochs=NEPOCHS_BFGS,
        lr=LR, model_dir=MODEL_DIR,
    )

    save_loss_csv(per_epoch_total, per_epoch_data, per_epoch_phys)

    pinn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    total_eval_loss, __, __, = eval_pinn(pinn, test_loader)

    save_pointwise_error(pinn)
    save_preds(pinn, 0)
    save_preds(pinn, 1)
    save_preds(pinn, 2)


if __name__ == "__main__":
    main()