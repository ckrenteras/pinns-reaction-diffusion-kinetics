import os
# must be set before the CUDA context is created for deterministic CuBLAS matmuls
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector

from scipy.interpolate import CubicSpline

from ssbfgs_pytorch import SSBFGS, SSBROYDEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#====== metadata constants =======

NUM_POS = 3130
NUM_T_STATES = 6
TIME_SLICES = [0, 16, 30, 43, 63, 71]
T_DIFF = 71
TAU_SLICES = np.array(TIME_SLICES) / T_DIFF
X_MIN = 0.6037047
X_MAX = 22.005236
Y_MIN = 3.1534662
Y_MAX = 40.003933
X_DIFF = X_MAX - X_MIN
Y_DIFF = Y_MAX - Y_MIN
POS_MIN = torch.tensor([X_MIN, Y_MIN], dtype=torch.float32).to(device)
POS_DIFF = torch.tensor([X_DIFF, Y_DIFF], dtype=torch.float32).to(device)

EPS_11_SCALE = 0.026053918
EPS_22_SCALE = 0.006760982
EPS_12_SCALE = 0.016504412
EPS_SCALE = torch.tensor([EPS_11_SCALE, EPS_22_SCALE, EPS_12_SCALE], dtype=torch.float32).to(device)

EPS_11_LEARN_SCALE = 100.0
EPS_22_LEARN_SCALE = 1000.0
EPS_12_LEARN_SCALE = 1000.0
EPS_LEARN_SCALE = torch.tensor(
    [EPS_11_LEARN_SCALE, EPS_22_LEARN_SCALE, EPS_12_LEARN_SCALE], dtype=torch.float32).to(device)
CONCENTRATION_CSV_PATH = os.path.join('.', 'data', 'Ihuaenyi_concentration_data.csv')
STRAIN_CSV_PATH = os.path.join('.', 'data', 'Ihuaenyi_strain_data.csv')
GRF_K_CSV_PATH = os.path.join('.', 'data', 'grf_interpolated.csv')
MODEL_DIR = os.path.join('.', 'models', 'v4')
RESULTS_DIR = os.path.join('.', 'results', 'v4')
EPS_IDX = [11, 22, 12]
PRED_COLS = ['c', 'e_11', 'e_22', 'e_12', 'k', 'j_0']

# ====== model constants ========

IN_CH_ONE = 3
IN_CH_TWO = 2
IN_CH_THREE = 1
OUT_CH_ONE = 4
OUT_CH_TWO = 1
OUT_CH_THREE = 1
WIDTH = 32
DEPTH = 5

# netC (j_0(c)) deliberately kept much smaller than netA
# an overparameterized netC can fit an arbitrarily complex j_0(c) 
# curve that absorbs spatial variation  through c's own spatial dependence
#causing k collapse to near-flat
# netC's should fit whatever spatial pattern the residual
# demands onto k instead
NETC_WIDTH = 16
NETC_DEPTH = 3

# ======= training constants ========
NEPOCHS_ADAM = 1000
LR = 1e-4
COSINE_ETA_MIN = LR * 1e-2

GRAD_NORM_REBALANCE_EVERY = 100
GRAD_NORM_EMA_ALPHA = 0.1

EARLY_STOP_PATIENCE_ADAM = 3 * GRAD_NORM_REBALANCE_EVERY
TAU_MAX = 1.0

# RAD resample consts
RAD_K1 = 1.0
RAD_K2 = 1.0
RAD_NSAMPLING = 20000
RAD_POOL_SIZE = 4096
RAD_RESAMPLE_EVERY = 100

# interpolation consts
N_INTERP_BETWEEN = 10
INTERP_BATCH_SIZE = 512


LOSS_TERM_NAMES = ['c', 'e11', 'e22', 'e12', 'allen_cahn', 'force_balance', 'k_reg']
INTERP_TERM_NAMES = ['interp_c', 'interp_e11', 'interp_e22', 'interp_e12']

# k_reg is a weak prior that k is of order 0 so it is not included in lambda rebalancing
# force_balance was excluded because its raw magnitude/gradient norm was orders of
# magnitude larger than the data terms under the old, un-non-dimensionalized residual
REBALANCED_TERM_NAMES = ['c', 'e11', 'e22', 'e12', 'allen_cahn', 'force_balance']

ALL_TERM_NAMES = LOSS_TERM_NAMES + INTERP_TERM_NAMES

# default weights set constant throughout training to compare test losses per epoch for best model selection
# these make each term roughly the same magnitude
DEFAULT_LAMBDAS = {
    'c': 8.9,
    'e11': 2.1,
    'e22': 0.48,
    'e12': 0.076,
    'allen_cahn': 100.0,
    'force_balance': 0.01,
    'k_reg': 1e-3,
    'interp_c': 0.1,
    'interp_e11': 0.1,
    'interp_e22': 0.1,
    'interp_e12': 0.1,
}

# ===== Broyden phase =====
QN_METHOD = 'SSBROYDEN2' 
QN_CHANGE_EVERY = 200  # how often rebalance lambdas and refresh the RAD collocation pool
QN_N_CHANGES = 10
QN_MAX_ITERS = QN_CHANGE_EVERY * QN_N_CHANGES
QN_PRINT_EVERY = 50
QN_LR = 1.0
QN_MAX_LINE_SEARCH = 25
QN_CURVATURE_EPS = 1e-16
QN_TOLERANCE_GRAD = 0.0
QN_TOLERANCE_CHANGE = 0.0
QN_INITIAL_SCALE = False

EARLY_STOP_PATIENCE_QN = 2 * QN_CHANGE_EVERY
BFGS_BATCH = NUM_POS * (NUM_T_STATES - 1)  # full training set in one batch (num_test_slices=1)

# ====== physical constants ======

# mu_res (reservoir chemical potential) is now a learnable scalar on AllenCahnPINN
# rather than a fixed constant here
a = 0.5
BETA_a = 2.33e-6
BETA_b = 1.52e-6
BETA_c = -0.88e-6

C_11 = 175.9
C_22 = 153.6
C_33 = 135.0
C_44 = 38.8
C_55 = 47.5
C_66 = 55.6
C_12 = 29.6
C_13 = 54.0
C_23 = 19.6

Q_aa = C_11 - (C_12**2 / C_22)
Q_ac = C_13 - ((C_12*C_23) / C_22)
Q_cc = C_33 - (C_23**2 / C_22)

MPA_TO_PA = 1e6  # stiffness constants above are given in MPa
C_EQV = torch.tensor([[Q_aa, Q_ac, 0],
                 [Q_ac, Q_cc, 0],
                 [0, 0, C_55]]).to(device) * MPA_TO_PA

# characteristic stress for non-dimensionalizing the force-balance residual (see
# force_balance()); Q_aa is a representative in-plane stiffness for this material
FORCE_BALANCE_SIGMA_CHAR = Q_aa * MPA_TO_PA

KBT = 4.1143 * 1e-21
KAPPA = 5.02 * 1e-10
C_MAX = 2.29 * 1e4
N_A = 6.02214076e23  # Avogadro's number converts molar (BETA, C_MAX) quantities to per-particle for dimensional consistency
UM_TO_M = 1e-6  # position data (x, y) is recorded in micrometers but KAPPA is given in J/m

# ==== data extraction + loaders =======

def extract_c_data(data_path=CONCENTRATION_CSV_PATH, num_test_slices=1):
    '''given path to csv data file extracts concentration data over pos and time
        returns input data and label numpy arrays'''
    df = pd.read_csv(data_path)
    x_vals = df.iloc[:, 0]
    y_vals = df.iloc[:, 1]
    x_vals_np = x_vals.to_numpy().astype(np.float32)
    y_vals_np = y_vals.to_numpy().astype(np.float32)
    in_data_np = None
    out_data_np = None
    for i in range(NUM_T_STATES - num_test_slices):
        c_vals = df.iloc[:, 2 + i]
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
        c_vals = df.iloc[:, 2 + i]
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


def extract_strain_data(data_path=STRAIN_CSV_PATH, num_test_slices=1):
    '''given path to csv data file extracts concentration data over pos and time
        returns input data and label numpy arrays'''
    df = pd.read_csv(data_path, header=1)
    x_vals = df.iloc[:, 0]
    y_vals = df.iloc[:, 1]
    x_vals_np = x_vals.to_numpy().astype(np.float32)
    y_vals_np = y_vals.to_numpy().astype(np.float32)
    in_data_np = None
    out_data_np = None
    EPS_11_START_COL = 3
    EPS_22_START_COL = 10
    EPS_12_START_COL = 17
    for i in range(NUM_T_STATES - num_test_slices):
        eps_11_np = df.iloc[:, EPS_11_START_COL + i].to_numpy().astype(np.float32)
        eps_22_np = df.iloc[:, EPS_22_START_COL + i].to_numpy().astype(np.float32)
        eps_12_np = df.iloc[:, EPS_12_START_COL + i].to_numpy().astype(np.float32)
        t_col = np.zeros_like(x_vals_np)
        t_col[:] = TAU_SLICES[i]
        time_slice_in = np.stack([x_vals_np, y_vals_np, t_col], axis=1)
        eps11_slice_out = eps_11_np.reshape(-1, 1)
        eps22_slice_out = eps_22_np.reshape(-1, 1)
        eps12_slice_out = eps_12_np.reshape(-1, 1)
        time_slice_out = np.concatenate([eps11_slice_out, eps22_slice_out, eps12_slice_out], axis=1)
        if in_data_np is not None:
            in_data_np = np.concatenate([in_data_np, time_slice_in], axis=0)
            out_data_np = np.concatenate([out_data_np, time_slice_out], axis=0)
        else:
            in_data_np = time_slice_in
            out_data_np = time_slice_out

    test_in_np = None
    test_out_np = None
    for i in range(NUM_T_STATES - num_test_slices, NUM_T_STATES):
        eps_11_np = df.iloc[:, EPS_11_START_COL + i].to_numpy().astype(np.float32)
        eps_22_np = df.iloc[:, EPS_22_START_COL + i].to_numpy().astype(np.float32)
        eps_12_np = df.iloc[:, EPS_12_START_COL + i].to_numpy().astype(np.float32)
        t_col = np.zeros_like(x_vals_np)
        t_col[:] = TAU_SLICES[i]
        time_slice_in = np.stack([x_vals_np, y_vals_np, t_col], axis=1)
        eps11_slice_out = eps_11_np.reshape(-1, 1)
        eps22_slice_out = eps_22_np.reshape(-1, 1)
        eps12_slice_out = eps_12_np.reshape(-1, 1)
        time_slice_out = np.concatenate([eps11_slice_out, eps22_slice_out, eps12_slice_out], axis=1)
        if test_in_np is not None:
            test_in_np = np.concatenate([test_in_np, time_slice_in], axis=0)
            test_out_np = np.concatenate([test_out_np, time_slice_out], axis=0)
        else:
            test_in_np = time_slice_in
            test_out_np = time_slice_out

    return in_data_np, out_data_np, test_in_np, test_out_np


class IhuaenyiData(Dataset):
    def __init__(self, concentration_path=CONCENTRATION_CSV_PATH,
                 strain_path=STRAIN_CSV_PATH, split='train', num_test_slices=1):
        train_data, train_labels, test_data, test_labels = extract_c_data(concentration_path, num_test_slices)
        __, train_epsilon, __, test_epsilon, = extract_strain_data(strain_path, num_test_slices)
        train_epsilon = train_epsilon[0:, 0:3]
        test_epsilon = test_epsilon[0:, 0:3]

        self.train_X = torch.from_numpy(train_data)
        self.train_Y = torch.from_numpy(np.concatenate([train_labels, train_epsilon], axis=1))
        self.test_X = torch.from_numpy(test_data)
        self.test_Y = torch.from_numpy(np.concatenate([test_labels, test_epsilon], axis=1))
        self.split = split

    def __len__(self):
        return len(self.train_X) if self.split == 'train' else len(self.test_X)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_X[idx], self.train_Y[idx]
        elif self.split == 'test':
            return self.test_X[idx], self.test_Y[idx]
        else:
            raise ValueError("Split must be either 'train' or 'test'")

DATALOADER_SEED = 0


def _loader_generator(seed=DATALOADER_SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_data(c_path=CONCENTRATION_CSV_PATH, strain_path=STRAIN_CSV_PATH, num_test_slices=1, batch_size=512):
    pin = device.type == 'cuda'
    train_data = IhuaenyiData(concentration_path=c_path, strain_path=strain_path,
                                    split='train', num_test_slices=num_test_slices)
    test_data = IhuaenyiData(concentration_path=c_path, strain_path=strain_path, split='test',
                                  num_test_slices=num_test_slices)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
        generator=_loader_generator(),
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=4096,
        shuffle=False,
    )

    return train_loader, test_loader


def load_domain_xy(data_path=CONCENTRATION_CSV_PATH):
    df = pd.read_csv(data_path)
    x_vals_np = df.iloc[:, 0].to_numpy().astype(np.float32)
    y_vals_np = df.iloc[:, 1].to_numpy().astype(np.float32)
    return x_vals_np, y_vals_np


def sample_collocation_batch(x_vals_np, y_vals_np, n_points, tau_max=TAU_MAX):
    idx = np.random.randint(0, len(x_vals_np), size=n_points)
    x = x_vals_np[idx]
    y = y_vals_np[idx]
    tau = np.random.uniform(0.0, tau_max, size=n_points).astype(np.float32)
    return np.stack([x, y, tau], axis=1)


def build_interp_pseudo_data(concentration_path=CONCENTRATION_CSV_PATH, strain_path=STRAIN_CSV_PATH,
                              num_test_slices=1, n_between=N_INTERP_BETWEEN):
    """using cubic spline"""
    train_in_np, train_c_np, _, _ = extract_c_data(concentration_path, num_test_slices)
    _, train_eps_np, _, _ = extract_strain_data(strain_path, num_test_slices)

    n_train_slices = NUM_T_STATES - num_test_slices
    x = train_in_np[:NUM_POS, 0]
    y = train_in_np[:NUM_POS, 1]
    tau_train = TAU_SLICES[:n_train_slices]

    c_grid = train_c_np.reshape(n_train_slices, NUM_POS, 1)
    eps_grid = train_eps_np.reshape(n_train_slices, NUM_POS, 3)
    y_grid = np.concatenate([c_grid, eps_grid], axis=2)  # (n_train_slices, NUM_POS, 4)

    spline = CubicSpline(tau_train, y_grid, axis=0)

    in_chunks, out_chunks = [], []
    for i in range(n_train_slices - 1):
        taus = np.linspace(tau_train[i], tau_train[i + 1], n_between + 2)[1:-1]
        for tau_val in taus:
            vals = spline(tau_val).astype(np.float32)  # (NUM_POS, 4): [c, e11, e22, e12]
            t_col = np.full_like(x, tau_val, dtype=np.float32)
            in_chunks.append(np.stack([x, y, t_col], axis=1))
            out_chunks.append(vals)

    return np.concatenate(in_chunks, axis=0), np.concatenate(out_chunks, axis=0)


def interp_loader(concentration_path=CONCENTRATION_CSV_PATH, strain_path=STRAIN_CSV_PATH,
                   num_test_slices=1, n_between=N_INTERP_BETWEEN, batch_size=INTERP_BATCH_SIZE):
    pin = device.type == 'cuda'
    in_np, out_np = build_interp_pseudo_data(concentration_path, strain_path, num_test_slices, n_between)
    print(f'Built {len(in_np)} interpolated pseudo-data points '
          f'({n_between} between each of the {NUM_T_STATES - num_test_slices} training time slices)')
    dataset = TensorDataset(torch.from_numpy(in_np), torch.from_numpy(out_np))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin,
                       generator=_loader_generator())


def full_batch_loader(c_path=CONCENTRATION_CSV_PATH, strain_path=STRAIN_CSV_PATH,
                       num_test_slices=1, batch_size=BFGS_BATCH):
    """load full batch for per epoch test comparison"""
    pin = device.type == 'cuda'
    train_data = IhuaenyiData(concentration_path=c_path, strain_path=strain_path,
                               split='train', num_test_slices=num_test_slices)
    return DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=pin,
                       generator=_loader_generator())


# ====== mathematical operations (normalize, time/spatial deriv., etc.) =======

def normalize_model_input(xy_tau):
    return torch.cat([(xy_tau[:, 0:2] - POS_MIN) / POS_DIFF, xy_tau[:, 2:3]], dim=1)

def compute_time_derivs(component, tau):
    return torch.autograd.grad(
        component,
        tau,
        grad_outputs=torch.ones_like(component),
        create_graph=True,
        retain_graph=True,
    )[0]

def compute_spatial_grad(component, pos):
    return torch.autograd.grad(
        component,
        pos,
        grad_outputs=torch.ones_like(component),
        create_graph=True,
        retain_graph=True,
    )[0]

def laplacian(component, pos, pos_diff=None):
    "in 2D"
    grad = compute_spatial_grad(component, pos)
    d_dx = grad[:, 0:1]
    d_dy = grad[:, 1:2]

    d2_dx = torch.autograd.grad(d_dx, pos, grad_outputs=torch.ones_like(d_dx),
                                create_graph=True, retain_graph=True)[0]
    d2_dy = torch.autograd.grad(d_dy, pos, grad_outputs=torch.ones_like(d_dy),
                                    create_graph=True, retain_graph=True)[0]
    d2_dx2 = d2_dx[:, 0:1]
    d2_dy2 = d2_dy[:, 1:2]
    if pos_diff is not None:
        return d2_dx2 / pos_diff[0]**2 + d2_dy2 / pos_diff[1]**2
    return d2_dx2 + d2_dy2


# ======== model ==========

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
    def __init__(self, netA, netB, netC):
        super().__init__()
        self.netA = netA
        self.netB = netB
        self.netC = netC
        # raw param kept near O(1) for Adam; mu itself is ~KBT scale (~1e-21), so an
        # unscaled parameter blows up eta = (mu - mu_res)/KBT after a single optimizer
        # step, since Adam's step size is ~lr regardless of the parameter's natural scale
        self.mu_res_raw = nn.Parameter(torch.zeros(1))

    @property
    def mu_res(self):
        return self.mu_res_raw * KBT

    def forward(self, c_eps_data):
        c_eps = self.netA(c_eps_data)
        pos_data = c_eps_data[:, 0:2]
        c = torch.sigmoid(c_eps[:, 0:1])
        eps_scaled = c_eps[:, 1:4]
        log_k = self.netB(pos_data)
        k = torch.exp(log_k)
        log_f = self.netC(c)
        f = torch.exp(log_f)
        j_0 = c*(1-c)*f
        out = torch.cat([c, eps_scaled, k, j_0], dim=1)
        return out


def descale_eps(eps_scaled):
    return eps_scaled / EPS_LEARN_SCALE

# ===== physics =======

def get_eps_ch(c, c_0):
    """returns eps_ch,11, eps_ch,22, eps_ch12; c, c_0 are dimensionless (c_bar),
    so beta must be nondimensionalized by c_max first: beta_tilde = beta * c_max"""
    eps_ch_11 = BETA_a * C_MAX * (c - c_0)
    eps_ch_22 = BETA_c * C_MAX * (c - c_0)
    eps_ch_12 = torch.zeros_like(eps_ch_11)
    return torch.cat([eps_ch_11, eps_ch_22, eps_ch_12], dim=1)

def epsilon_voigt_vector(epsilon):
    """expects epsilon like [eps_11, eps_22, eps_12]"""
    eps_11, eps_22, eps_12 = epsilon[:, 0:1], epsilon[:, 1:2], epsilon[:, 2:3]
    zeros = torch.zeros_like(eps_11)
    return torch.cat([eps_11,
                        eps_22,
                        2 * eps_12], dim=1)

def stress_strain_eq(c, c_0, epsilon):
    """returns stress voigt vector given, c, c_0,
    and epsilon like [eps_11, eps_22, eps_12]"""
    eps_ch = get_eps_ch(c, c_0)
    elastic_strain = epsilon - eps_ch
    elastic_strain_voigt = epsilon_voigt_vector(elastic_strain)
    return  (C_EQV @ elastic_strain_voigt.T).T

def force_balance(sigma, pos):
    """expects input sigma in voigt notation. non-dimensionalized by Q_aa so the residual sits
    at the same O(1) order as the other loss terms """
    sigma_nd = sigma / FORCE_BALANCE_SIGMA_CHAR
    sigma_11 = sigma_nd[:, 0:1]
    sigma_22 = sigma_nd[:, 1:2]
    sigma_12 = sigma_nd[:, 2:3]

    dsigma_11 = compute_spatial_grad(sigma_11, pos)
    dsigma_22 = compute_spatial_grad(sigma_22, pos)
    dsigma_12 = compute_spatial_grad(sigma_12, pos)

    div_sigma_x = dsigma_11[:, 0:1] + dsigma_12[:, 1:2]
    div_sigma_y = dsigma_12[:, 0:1] + dsigma_22[:, 1:2]
    return torch.cat([div_sigma_x, div_sigma_y], dim=1)


# ====== allen-cahn eq ========
def get_mu_h(c):
    # Approximation given by Royal
    # Note: checked 2026-08-24 against data/chem.csv over the actual measured 
    # concentration range (~0.01-0.99) and mean abs gap is ~0.064
    # about 0.13 near the concentration boundarie
    # inherent limitation to this approximation, should be addressed
    return torch.log(c / (1 - c)) + 3.311*(1 - 2 *c)

def get_mu(mu_h, c, sigma, pos, pos_diff):
    """expects sigma as a voigt vector; mu = mu_h(c) - k*laplacian(c) - beta:sigma,
    all terms combined on a per-lithium-site (particle) energy basis to match KBT*mu_h"""
    sigma_11 = sigma[:, 0:1]
    sigma_22 = sigma[:, 1:2]
    # BETA is a molar eigenstrain coefficient (m^3/mol); BETA*sigma is J/mol, so divide by
    # N_A to get a per-particle energy matching KBT*mu_h
    stress_term = (BETA_a * sigma_11 + BETA_c * sigma_22) / N_A
    # pos_diff is in micrometers but KAPPA is given in J/m, so convert to meters for the
    # Laplacian
    # KAPPA*laplacian(c) is then an energy density (J/m^3), so divide by the
    # lithium-site number density (C_MAX * N_A, sites/m^3) to get a per-particle energy
    pos_diff_m = pos_diff * UM_TO_M
    grad_energy_term = (KAPPA * laplacian(c, pos, pos_diff_m)) / (C_MAX * N_A)
    homogeneous_term = KBT * mu_h
    return homogeneous_term - grad_energy_term - stress_term # per royal's note unnormalize mu_h by * KBT

def get_eta(mu, mu_res):
    """mu is already dimensionless (mu_h is nondimensionalized by KBT per the paper,
    and the gradient/stress corrections are small perturbations on that same scale)"""
    return (mu - mu_res) / KBT

def get_allen_cahn_res(c, tau, eta, t_diff, k, j_0):
    dc_dtau = compute_time_derivs(c, tau)
    dc_dt = dc_dtau / t_diff
    rhs_1 = k * j_0 *(torch.exp(-a * eta) - torch.exp((1-a)*eta))
    return  dc_dt - rhs_1


def get_residuals(c, c_0, epsilon, k, j_0, tau, t_diff, pos, pos_diff, mu_res):
    sigma = stress_strain_eq(c, c_0, epsilon)
    force_balance_res = force_balance(sigma, pos)

    mu_h = get_mu_h(c)
    mu = get_mu(mu_h, c, sigma, pos, pos_diff)
    eta = get_eta(mu, mu_res)
    allen_cahn_res = get_allen_cahn_res(c, tau, eta, t_diff, k, j_0)

    return allen_cahn_res, force_balance_res


def collocation_residual_magnitude(model, xy_tau, t_diff):
    pos = ((xy_tau[:, 0:2] - POS_MIN) / POS_DIFF).clone().detach().requires_grad_(True)
    tau = xy_tau[:, 2:3].clone().detach().requires_grad_(True)
    z_hat = model(torch.cat([pos, tau], dim=1))
    c_pred = z_hat[:, 0:1]
    eps_pred = descale_eps(z_hat[:, 1:4])
    k_pred = z_hat[:, 4:5]
    j_0_pred = z_hat[:, 5:6]

    tau0 = torch.zeros_like(tau, requires_grad=False)
    c_0 = model(torch.cat([pos, tau0], dim=1))[:, 0:1]

    allen_cahn_res, force_balance_res = get_residuals(
        c_pred, c_0, eps_pred, k_pred, j_0_pred, tau, t_diff, pos, POS_DIFF, model.mu_res)

    combined = torch.cat([allen_cahn_res, force_balance_res], dim=1)
    return combined.pow(2).sum(dim=1).sqrt().detach()


def adaptive_rad_sample(model, x_vals_np, y_vals_np, n_points, t_diff, tau_max=TAU_MAX,
                         k1=RAD_K1, k2=RAD_K2, n_sampling=RAD_NSAMPLING):
    """draw n_sampling candidates uniformly over the domain, weight  by
    the residual magnitude, and sample n_points from distribution without replacement"""
    param_dtype = next(model.parameters()).dtype
    candidates_np = sample_collocation_batch(x_vals_np, y_vals_np, n_sampling, tau_max)
    candidates = torch.from_numpy(candidates_np).to(device=device, dtype=param_dtype)

    y = collocation_residual_magnitude(model, candidates, t_diff)
    w = y ** k1
    err_eq = w / w.mean() + k2
    p = (err_eq / err_eq.sum()).clamp_min(1e-12)
    idx = torch.multinomial(p, num_samples=min(n_points, n_sampling), replacement=False)
    return candidates[idx].detach().cpu().numpy()

# ======= loss =======

def get_data_loss_terms(c_pred, c_true, eps_pred_scaled, eps_true):
    """using plain epsilon scaling: compares the model's already-scaled eps
    prediction against eps_true * EPS_LEARN_SCALE"""
    data_loss_c = torch.mean((c_pred - c_true) ** 2)

    eps_true_scaled = eps_true * EPS_LEARN_SCALE
    data_loss_e11 = torch.mean((eps_pred_scaled[:, 0:1] - eps_true_scaled[:, 0:1]) ** 2)
    data_loss_e22 = torch.mean((eps_pred_scaled[:, 1:2] - eps_true_scaled[:, 1:2]) ** 2)
    data_loss_e12 = torch.mean((eps_pred_scaled[:, 2:3] - eps_true_scaled[:, 2:3]) ** 2)

    return data_loss_c, data_loss_e11, data_loss_e22, data_loss_e12


def get_k_reg_loss(k_pred):
    """regularizer on ln(k): pushes ln(k) toward 0, i.e. k toward O(1)"""
    return torch.mean(torch.log(k_pred) ** 2)


def pinn_loss(
        model,
        in_data,
        labels,
        t_diff,
        lambdas=DEFAULT_LAMBDAS,
        collocation_data=None,
        interp_data=None,
        interp_labels=None):
    pos_data = ((in_data[:, 0:2].clone().detach() - POS_MIN) / POS_DIFF).requires_grad_(True)
    tau = in_data[:, 2:3].clone().detach().requires_grad_(True)
    z_hat = model(torch.cat([pos_data, tau], dim=1))
    c_pred = z_hat[:, 0:1]
    eps_pred_scaled = z_hat[:, 1:4]
    eps_pred = descale_eps(eps_pred_scaled)
    k_pred = z_hat[:, 4:5]
    j_0_pred = z_hat[:, 5:6]

    tau0 = torch.zeros_like(tau, requires_grad=False)
    model_input_0 = torch.cat([pos_data, tau0], dim=1)
    z_hat_0 = model(model_input_0)
    c_0 = z_hat_0[:, 0:1]

    c_true = labels[:, 0:1]
    eps_true = labels[:, 1:4]
    data_loss_c, data_loss_e11, data_loss_e22, data_loss_e12 = get_data_loss_terms(
        c_pred, c_true, eps_pred_scaled, eps_true)

    allen_cahn_res, force_balance_res = get_residuals(c_pred, c_0, eps_pred, k_pred, j_0_pred,
                                                      tau, t_diff, pos_data, POS_DIFF, model.mu_res)

    # physics residual is also enforced on RAD collocation points sampled
    # across the full space-time domain (physics-only, no labels)
    if collocation_data is not None:
        colloc_pos = ((collocation_data[:, 0:2].clone().detach() - POS_MIN) / POS_DIFF).requires_grad_(True)
        colloc_tau = collocation_data[:, 2:3].clone().detach().requires_grad_(True)
        z_hat_c = model(torch.cat([colloc_pos, colloc_tau], dim=1))
        c_pred_c = z_hat_c[:, 0:1]
        eps_pred_c = descale_eps(z_hat_c[:, 1:4])
        k_pred_c = z_hat_c[:, 4:5]
        j_0_pred_c = z_hat_c[:, 5:6]

        colloc_tau0 = torch.zeros_like(colloc_tau, requires_grad=False)
        z_hat_c0 = model(torch.cat([colloc_pos, colloc_tau0], dim=1))
        c_0_c = z_hat_c0[:, 0:1]

        allen_cahn_res_c, force_balance_res_c = get_residuals(
            c_pred_c, c_0_c, eps_pred_c, k_pred_c, j_0_pred_c,
            colloc_tau, t_diff, colloc_pos, POS_DIFF, model.mu_res)

        allen_cahn_res = torch.cat([allen_cahn_res, allen_cahn_res_c], dim=0)
        force_balance_res = torch.cat([force_balance_res, force_balance_res_c], dim=0)
        k_pred = torch.cat([k_pred, k_pred_c], dim=0)

    physics_loss_allen_cahn = torch.mean(allen_cahn_res ** 2)
    physics_loss_force_balance = torch.mean(force_balance_res ** 2)

    k_reg_loss = get_k_reg_loss(k_pred)

    loss_terms = {
        'c': data_loss_c,
        'e11': data_loss_e11,
        'e22': data_loss_e22,
        'e12': data_loss_e12,
        'allen_cahn': physics_loss_allen_cahn,
        'force_balance': physics_loss_force_balance,
        'k_reg': k_reg_loss,
    }

    # soft-data supervision from cubic-spline-interpolated pseudo-labels
    # no physics residual is enforced here as RAD points are used for that
    # as such dont track grads of these terms
    if interp_data is not None:
        interp_pos = (interp_data[:, 0:2] - POS_MIN) / POS_DIFF
        interp_tau = interp_data[:, 2:3]
        z_hat_i = model(torch.cat([interp_pos, interp_tau], dim=1))
        c_pred_i = z_hat_i[:, 0:1]
        eps_pred_i_scaled = z_hat_i[:, 1:4]

        c_true_i = interp_labels[:, 0:1]
        eps_true_i = interp_labels[:, 1:4]
        interp_loss_c, interp_loss_e11, interp_loss_e22, interp_loss_e12 = get_data_loss_terms(
            c_pred_i, c_true_i, eps_pred_i_scaled, eps_true_i)
    else:
        zero = torch.zeros((), device=in_data.device, dtype=in_data.dtype)
        interp_loss_c = interp_loss_e11 = interp_loss_e22 = interp_loss_e12 = zero

    loss_terms.update({
        'interp_c': interp_loss_c,
        'interp_e11': interp_loss_e11,
        'interp_e22': interp_loss_e22,
        'interp_e12': interp_loss_e12,
    })

    total_loss = sum(lambdas[name] * loss_terms[name] for name in ALL_TERM_NAMES)
    return total_loss, loss_terms


def compute_term_grad_norms(model, in_data, labels, t_diff, collocation_data=None):
    """L2 norm of each (unweighted) loss term's gradient w.r.t. model params;
    used for Wang et al. (2021) esque adaptive loss balancing. Restricted to
    REBALANCED_TERM_NAMES"""
    unit_lambdas = {name: 1.0 for name in ALL_TERM_NAMES}
    __, loss_terms = pinn_loss(model, in_data, labels, t_diff, unit_lambdas, collocation_data)
    params = list(model.parameters())
    grad_norms = {}
    for name in REBALANCED_TERM_NAMES:
        grads = torch.autograd.grad(
            loss_terms[name], params, retain_graph=True, allow_unused=True)
        sq_norms = [g.pow(2).sum() for g in grads if g is not None]
        grad_norms[name] = torch.sqrt(sum(sq_norms)).item() if sq_norms else 0.0
    return grad_norms


def rebalance_lambdas(lambdas, grad_norms, alpha=GRAD_NORM_EMA_ALPHA, eps=1e-8, lambda_max=1e3):
    reference = max(grad_norms[name] for name in ('c', 'e11', 'e22', 'e12'))
    new_lambdas = dict(lambdas)
    for name in REBALANCED_TERM_NAMES:
        target = min(reference / (grad_norms[name] + eps), lambda_max)
        new_lambdas[name] = (1 - alpha) * lambdas[name] + alpha * target
    return new_lambdas

# ======= train + eval ======

K_STATS_EVERY = 10


def k_output_stats(model, x_vals_np, y_vals_np):
    """std/min/max of the model's predicted k(x, y) over the full domain grid, so
    checkpoint selection's blindness to k quality (it only sees a 1e-3-weighted
    k_reg term, not k's actual spread) can be diagnosed from logged history instead
    of guessed at after the fact; k is tau-independent (netB only sees position) so
    tau is fixed to 0"""
    tau = np.zeros_like(x_vals_np)
    xy_tau = np.stack([x_vals_np, y_vals_np, tau], axis=1)
    was_training = model.training
    param_dtype = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        inputs = normalize_model_input(
            torch.from_numpy(xy_tau).to(device=device, dtype=param_dtype))
        k_pred = model(inputs)[:, 4]
    if was_training:
        model.train()
    return k_pred.std().item(), k_pred.min().item(), k_pred.max().item()


def eval_pinn(model, data_loader, lambdas=DEFAULT_LAMBDAS, collocation_data=None):
    n = 0
    model.eval()
    total_loss = 0.0
    term_sums = {name: 0.0 for name in ALL_TERM_NAMES}
    param_dtype = next(model.parameters()).dtype
    for in_data, labels in data_loader:
        in_data, labels = in_data.to(device=device, dtype=param_dtype), labels.to(device=device, dtype=param_dtype)
        iter_total_loss, iter_loss_terms = pinn_loss(
            model, in_data, labels, T_DIFF, lambdas, collocation_data)

        total_loss += iter_total_loss
        for name in ALL_TERM_NAMES:
            term_sums[name] += iter_loss_terms[name]
        n += 1
    total_loss /= n
    for name in ALL_TERM_NAMES:
        term_sums[name] = (term_sums[name] / n).item()
    return total_loss.item(), term_sums


def _next_batch(loader, it):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return batch, it


def train_pinn_adam_only(model, adam_loader, interp_data_loader, lambdas=DEFAULT_LAMBDAS,
                          adam_epochs=NEPOCHS_ADAM, lr=LR, model_dir=MODEL_DIR, test_loader=None,
                          disable_early_stop=False):
    lambdas = dict(lambdas)
    x_vals_np, y_vals_np = load_domain_xy()

    per_epoch_total = np.zeros(adam_epochs)
    per_epoch_terms = {name: np.zeros(adam_epochs) for name in ALL_TERM_NAMES}
    per_epoch_lambdas = {name: np.zeros(adam_epochs) for name in ALL_TERM_NAMES}
    per_epoch_test_total = np.zeros(adam_epochs)
    per_epoch_test_terms = {name: np.zeros(adam_epochs) for name in ALL_TERM_NAMES}
    per_epoch_k_std = np.zeros(adam_epochs)
    per_epoch_k_min = np.zeros(adam_epochs)
    per_epoch_k_max = np.zeros(adam_epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(adam_epochs, 1), eta_min=COSINE_ETA_MIN)

    best_loss = float('inf')
    epochs_since_improvement = 0

    colloc_pool_np = None
    interp_iter = iter(interp_data_loader) if interp_data_loader is not None else None

    for epoch in range(adam_epochs):

        iter_total_loss = torch.zeros((), device=device)
        iter_term_sums = {name: torch.zeros((), device=device) for name in ALL_TERM_NAMES}
        n = 0

        if epoch == 0 or epoch % RAD_RESAMPLE_EVERY == 0:
            colloc_pool_np = adaptive_rad_sample(model, x_vals_np, y_vals_np, RAD_POOL_SIZE, T_DIFF)

        for batch_idx, (inputs, labels) in enumerate(adam_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            colloc_idx = np.random.randint(0, len(colloc_pool_np), size=inputs.shape[0])
            collocation_data = torch.from_numpy(colloc_pool_np[colloc_idx]).to(device)

            if interp_data_loader is not None:
                (interp_inputs, interp_labels), interp_iter = _next_batch(interp_data_loader, interp_iter)
                interp_inputs = interp_inputs.to(device)
                interp_labels = interp_labels.to(device)
            else:
                interp_inputs = None
                interp_labels = None

            if batch_idx == 0 and (epoch == 0 or epoch % GRAD_NORM_REBALANCE_EVERY == 0):
                grad_norms = compute_term_grad_norms(model, inputs, labels, T_DIFF, collocation_data)
                lambdas = rebalance_lambdas(lambdas, grad_norms)

            step_lambdas = lambdas

            optimizer.zero_grad()
            loss, loss_terms = pinn_loss(
                model,
                in_data=inputs,
                labels=labels,
                t_diff=T_DIFF,
                lambdas=step_lambdas,
                collocation_data=collocation_data,
                interp_data=interp_inputs,
                interp_labels=interp_labels)

            loss.backward()
            optimizer.step()
            n += 1
            iter_total_loss += loss.detach()
            for name in ALL_TERM_NAMES:
                iter_term_sums[name] += loss_terms[name].detach()

        scheduler.step()

        iter_total_loss = (iter_total_loss / n).item()
        for name in ALL_TERM_NAMES:
            iter_term_sums[name] = (iter_term_sums[name] / n).item()

        per_epoch_total[epoch] = iter_total_loss
        for name in ALL_TERM_NAMES:
            per_epoch_terms[name][epoch] = iter_term_sums[name]
            per_epoch_lambdas[name][epoch] = step_lambdas[name]
        if epoch % K_STATS_EVERY == 0:
            per_epoch_k_std[epoch], per_epoch_k_min[epoch], per_epoch_k_max[epoch] = (
                k_output_stats(model, x_vals_np, y_vals_np))
        else:
            per_epoch_k_std[epoch] = per_epoch_k_std[epoch - 1]
            per_epoch_k_min[epoch] = per_epoch_k_min[epoch - 1]
            per_epoch_k_max[epoch] = per_epoch_k_max[epoch - 1]

        if test_loader is not None:
            test_total_loss, test_term_sums = eval_pinn(model, test_loader, step_lambdas)
            model.train()
            per_epoch_test_total[epoch] = test_total_loss
            for name in ALL_TERM_NAMES:
                per_epoch_test_terms[name][epoch] = test_term_sums[name]
            # fixed lambdas here for the sake of good comparison
            ckpt_loss = sum(DEFAULT_LAMBDAS[name] * test_term_sums[name] for name in ALL_TERM_NAMES)
        else:
            ckpt_loss = sum(DEFAULT_LAMBDAS[name] * iter_term_sums[name] for name in ALL_TERM_NAMES)

        # a checkpoint only becomes eligible for "best" once the lambdas have been
        # rebalanced at least once; before that the network is close to random init,
        # and a near-constant output can score deceptively well against the single
        # held-out (extrapolation) test slice purely by chance, not genuine fit
        if epoch >= GRAD_NORM_REBALANCE_EVERY and ckpt_loss < best_loss:
            best_loss = ckpt_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        else:
            epochs_since_improvement += 1

        if epoch % 100 == 0:
            terms_str = '  | '.join(f'{name}: {iter_term_sums[name]}' for name in ALL_TERM_NAMES)
            lambdas_str = '  | '.join(f'lam_{name}: {step_lambdas[name]:.4g}' for name in ALL_TERM_NAMES)
            print(f'EPOCH: {epoch}  | {terms_str}  | total loss: {iter_total_loss}  '
                  f'| lr: {scheduler.get_last_lr()[0]:.3e}')
            print(f'         {lambdas_str}')
            if test_loader is not None:
                print(f'         test total loss: {per_epoch_test_total[epoch]}')

        if not disable_early_stop and epochs_since_improvement >= EARLY_STOP_PATIENCE_ADAM:
            print(f'Adam early stop at epoch {epoch}: no {"test" if test_loader is not None else "train"} '
                  f'loss improvement for {EARLY_STOP_PATIENCE_ADAM} epochs (best: {best_loss})')
            adam_epochs_run = epoch + 1
            break
    else:
        adam_epochs_run = adam_epochs

    if best_loss == float('inf'):
        # adam_epochs was too short to clear the GRAD_NORM_REBALANCE_EVERY floor above,
        # so no checkpoint was ever saved this run; fall back to the final weights so
        # best_model.pt always reflects this run instead of silently reusing a stale
        # file left over from a previous experiment
        torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

    per_epoch_total = per_epoch_total[:adam_epochs_run]
    per_epoch_terms = {name: arr[:adam_epochs_run] for name, arr in per_epoch_terms.items()}
    per_epoch_lambdas = {name: arr[:adam_epochs_run] for name, arr in per_epoch_lambdas.items()}
    per_epoch_test_total = per_epoch_test_total[:adam_epochs_run]
    per_epoch_test_terms = {name: arr[:adam_epochs_run] for name, arr in per_epoch_test_terms.items()}
    per_epoch_k_stats = {
        'std': per_epoch_k_std[:adam_epochs_run],
        'min': per_epoch_k_min[:adam_epochs_run],
        'max': per_epoch_k_max[:adam_epochs_run],
    }

    return (best_loss, per_epoch_total, per_epoch_terms, per_epoch_lambdas, per_epoch_test_total,
            per_epoch_test_terms, adam_epochs_run, per_epoch_k_stats)


def train_pinn_qn(model, qn_inputs, qn_labels, interp_inputs, interp_labels, lambdas,
                   x_vals_np, y_vals_np, model_dir, test_loader=None, qn_method=QN_METHOD,
                   max_iters=QN_MAX_ITERS, change_every=QN_CHANGE_EVERY,
                   patience=EARLY_STOP_PATIENCE_QN):
    lambdas = dict(lambdas)

    global POS_MIN, POS_DIFF, EPS_LEARN_SCALE, C_EQV
    POS_MIN, POS_DIFF = POS_MIN.double(), POS_DIFF.double()
    EPS_LEARN_SCALE, C_EQV = EPS_LEARN_SCALE.double(), C_EQV.double()
    model.double()

    qn_inputs = qn_inputs.to(device=device, dtype=torch.float64)
    qn_labels = qn_labels.to(device=device, dtype=torch.float64)
    if interp_inputs is not None:
        interp_inputs = interp_inputs.to(device=device, dtype=torch.float64)
        interp_labels = interp_labels.to(device=device, dtype=torch.float64)

    colloc_pool_np = adaptive_rad_sample(model, x_vals_np, y_vals_np, BFGS_BATCH, T_DIFF)
    collocation_data = torch.from_numpy(colloc_pool_np).to(device=device, dtype=torch.float64)

  
    grad_norms = compute_term_grad_norms(model, qn_inputs, qn_labels, T_DIFF, collocation_data)
    lambdas = rebalance_lambdas(lambdas, grad_norms)

    method_key = qn_method.upper()
    common_kwargs = dict(
        lr=QN_LR, max_iter=1, tolerance_grad=QN_TOLERANCE_GRAD,
        tolerance_change=QN_TOLERANCE_CHANGE, max_line_search=QN_MAX_LINE_SEARCH,
        curvature_eps=QN_CURVATURE_EPS, initial_scale=QN_INITIAL_SCALE)
    if method_key == 'SSBFGS_OL':
        qn_optimizer = SSBFGS(model.parameters(), variant='OL', **common_kwargs)
    elif method_key == 'SSBFGS_AB':
        qn_optimizer = SSBFGS(model.parameters(), variant='AB', **common_kwargs)
    elif method_key in {'SSBROYDEN1', 'SSBROYDEN_1'}:
        qn_optimizer = SSBROYDEN(model.parameters(), variant=1, **common_kwargs)
    elif method_key in {'SSBROYDEN2', 'SSBROYDEN_2'}:
        qn_optimizer = SSBROYDEN(model.parameters(), variant=2, **common_kwargs)
    else:
        raise ValueError('qn_method must be SSBFGS_OL, SSBFGS_AB, SSBROYDEN1, or SSBROYDEN2')
    qn_state = qn_optimizer.state[next(iter(model.parameters()))]

    per_iter_total = np.zeros(max_iters)
    per_iter_terms = {name: np.zeros(max_iters) for name in ALL_TERM_NAMES}
    per_iter_lambdas = {name: np.zeros(max_iters) for name in ALL_TERM_NAMES}
    per_iter_test_total = np.zeros(max_iters)
    per_iter_test_terms = {name: np.zeros(max_iters) for name in ALL_TERM_NAMES}
    per_iter_k_std = np.zeros(max_iters)
    per_iter_k_min = np.zeros(max_iters)
    per_iter_k_max = np.zeros(max_iters)

    term_sink = {}

    def qn_closure():
        qn_optimizer.zero_grad(set_to_none=True)
        total_loss, loss_terms = pinn_loss(
            model, in_data=qn_inputs, labels=qn_labels, t_diff=T_DIFF, lambdas=lambdas,
            collocation_data=collocation_data, interp_data=interp_inputs, interp_labels=interp_labels)
        total_loss.backward()
        term_sink.clear()
        term_sink.update({name: loss_terms[name].item() for name in ALL_TERM_NAMES})
        return total_loss

    best_loss = float('inf')
    epochs_since_improvement = 0
    n_iters_run = 0

    for it in range(max_iters):
        try:
            qn_loss = qn_optimizer.step(qn_closure)
        except RuntimeError as exc:
            if 'strong-Wolfe line search failed' not in str(exc):
                raise

            print(f'QN iter {it}: strong-Wolfe search exhausted; resampling collocation and continuing.')
            colloc_pool_np = adaptive_rad_sample(model, x_vals_np, y_vals_np, BFGS_BATCH, T_DIFF)
            collocation_data = torch.from_numpy(colloc_pool_np).to(device=device, dtype=torch.float64)
            qn_state['old_old_loss'] = None
            continue

        n_iters_run = it + 1
        per_iter_total[it] = qn_loss.item()
        for name in ALL_TERM_NAMES:
            per_iter_terms[name][it] = term_sink[name]
            per_iter_lambdas[name][it] = lambdas[name]
        if it % K_STATS_EVERY == 0:
            per_iter_k_std[it], per_iter_k_min[it], per_iter_k_max[it] = (
                k_output_stats(model, x_vals_np, y_vals_np))
        else:
            per_iter_k_std[it] = per_iter_k_std[it - 1]
            per_iter_k_min[it] = per_iter_k_min[it - 1]
            per_iter_k_max[it] = per_iter_k_max[it - 1]

        if test_loader is not None:
            test_total_loss, test_term_sums = eval_pinn(model, test_loader, lambdas)
            model.train()
            per_iter_test_total[it] = test_total_loss
            for name in ALL_TERM_NAMES:
                per_iter_test_terms[name][it] = test_term_sums[name]
            # fixed DEFAULT_LAMBDAS weighting for checkpoint selection
            ckpt_loss = sum(DEFAULT_LAMBDAS[name] * test_term_sums[name] for name in ALL_TERM_NAMES)
        else:
            ckpt_loss = sum(DEFAULT_LAMBDAS[name] * term_sink[name] for name in ALL_TERM_NAMES)

        if ckpt_loss < best_loss:
            best_loss = ckpt_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        else:
            epochs_since_improvement += 1

        if it % QN_PRINT_EVERY == 0:
            terms_str = '  | '.join(f'{name}: {term_sink[name]}' for name in ALL_TERM_NAMES)
            lambdas_str = '  | '.join(f'lam_{name}: {lambdas[name]:.4g}' for name in ALL_TERM_NAMES)
            print(f'QN ITER: {it}  | {terms_str}  | total loss: {qn_loss.item()}')
            print(f'         {lambdas_str}')
            if test_loader is not None:
                print(f'         test total loss: {per_iter_test_total[it]}')

        if epochs_since_improvement >= patience:
            print(f'QN early stop at iter {it}: no test loss improvement for {patience} iters (best: {best_loss})')
            break

        if (it + 1) % change_every == 0 and it + 1 < max_iters:
            grad_norms = compute_term_grad_norms(model, qn_inputs, qn_labels, T_DIFF, collocation_data)
            lambdas = rebalance_lambdas(lambdas, grad_norms)
            colloc_pool_np = adaptive_rad_sample(model, x_vals_np, y_vals_np, BFGS_BATCH, T_DIFF)
            collocation_data = torch.from_numpy(colloc_pool_np).to(device=device, dtype=torch.float64)
            lambdas_str = '  | '.join(f'lam_{name}: {lambdas[name]:.4g}' for name in ALL_TERM_NAMES)
            print(f'QN chunk boundary at iter {it + 1}: {lambdas_str}')

    per_iter_total = per_iter_total[:n_iters_run]
    per_iter_terms = {name: arr[:n_iters_run] for name, arr in per_iter_terms.items()}
    per_iter_lambdas = {name: arr[:n_iters_run] for name, arr in per_iter_lambdas.items()}
    per_iter_test_total = per_iter_test_total[:n_iters_run]
    per_iter_test_terms = {name: arr[:n_iters_run] for name, arr in per_iter_test_terms.items()}
    per_iter_k_stats = {
        'std': per_iter_k_std[:n_iters_run],
        'min': per_iter_k_min[:n_iters_run],
        'max': per_iter_k_max[:n_iters_run],
    }

    model.float()
    POS_MIN, POS_DIFF = POS_MIN.float(), POS_DIFF.float()
    EPS_LEARN_SCALE, C_EQV = EPS_LEARN_SCALE.float(), C_EQV.float()

    return (best_loss, per_iter_total, per_iter_terms, per_iter_lambdas, per_iter_test_total,
            per_iter_test_terms, n_iters_run, per_iter_k_stats)

# ====== for plotting ========


def save_loss_csv(per_step_total, per_step_terms, per_step_lambdas, results_dir=RESULTS_DIR,
                   adam_epochs=NEPOCHS_ADAM, test_total=None, test_terms=None, k_stats=None):
    os.makedirs(results_dir, exist_ok=True)
    steps = np.arange(len(per_step_total))
    phase = np.where(steps < adam_epochs, 'adam', 'bfgs')
    data = {'step': steps, 'phase': phase}
    for name in ALL_TERM_NAMES:
        data[f'{name}_loss'] = per_step_terms[name]
    data['total_loss'] = per_step_total
    if test_total is not None:
        data['test_total_loss'] = test_total
        for name in ALL_TERM_NAMES:
            data[f'test_{name}_loss'] = test_terms[name]
    if k_stats is not None:
        data['k_std'] = k_stats['std']
        data['k_min'] = k_stats['min']
        data['k_max'] = k_stats['max']
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(results_dir, 'loss_history.csv'), index=False)

    lambda_data = {'step': steps, 'phase': phase}
    for name in ALL_TERM_NAMES:
        lambda_data[f'{name}_lambda'] = per_step_lambdas[name]
    pd.DataFrame(lambda_data).to_csv(os.path.join(results_dir, 'lambda_history.csv'), index=False)


def save_preds(model, field_idx, data_path=CONCENTRATION_CSV_PATH,
                         results_dir=RESULTS_DIR):
    if field_idx not in range(6):
        raise ValueError('Invalid field index')
    in_data, _, _, _ = extract_c_data(data_path, num_test_slices=0)
    x = in_data[:NUM_POS, 0]
    y = in_data[:NUM_POS, 1]
    pred_field = PRED_COLS[field_idx]

    model.eval()
    cols = {'x': x, 'y': y}
    with torch.no_grad():
        for i in range(NUM_T_STATES):
            slice_in = in_data[i * NUM_POS:(i + 1) * NUM_POS]
            inputs = normalize_model_input(torch.from_numpy(slice_in).to(device))
            out = model(inputs)[:, field_idx]
            if field_idx in (1, 2, 3):
                out = out / EPS_LEARN_SCALE[field_idx - 1]
            pred = out.cpu().numpy()
            t_val = TIME_SLICES[i]
            cols[f'pred_{pred_field}{t_val}'] = pred

    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(cols).to_csv(os.path.join(results_dir, f'pred_{pred_field}.csv'), index=False)


def save_mu_h_preds(model, data_path=CONCENTRATION_CSV_PATH, results_dir=RESULTS_DIR):
    """mu_h is not a network output; it's get_mu_h applied to the model's own c
    prediction, same as it's used inside the Allen-Cahn residual during training"""
    in_data, _, _, _ = extract_c_data(data_path, num_test_slices=0)
    x = in_data[:NUM_POS, 0]
    y = in_data[:NUM_POS, 1]

    model.eval()
    cols = {'x': x, 'y': y}
    with torch.no_grad():
        for i in range(NUM_T_STATES):
            slice_in = in_data[i * NUM_POS:(i + 1) * NUM_POS]
            inputs = normalize_model_input(torch.from_numpy(slice_in).to(device))
            c_pred = model(inputs)[:, 0:1]
            mu_h_pred = get_mu_h(c_pred).squeeze(1).cpu().numpy()
            t_val = TIME_SLICES[i]
            cols[f'pred_mu_h{t_val}'] = mu_h_pred

    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(cols).to_csv(os.path.join(results_dir, 'pred_mu_h.csv'), index=False)


def save_k_at_grf_points(model, grf_path=GRF_K_CSV_PATH, results_dir=RESULTS_DIR):
    """evaluates the model's k(x, y) at the GT GRF field's own (x, y) points (a
    different, denser grid than the training positions) for a pointwise comparison"""
    df = pd.read_csv(grf_path)
    x = df['x'].to_numpy().astype(np.float32)
    y = df['y'].to_numpy().astype(np.float32)
    grf_k = df['grf_interpolated'].to_numpy().astype(np.float32)
    tau = np.zeros_like(x)  # k(x, y) is tau-independent (netB only sees position)
    xy_tau = np.stack([x, y, tau], axis=1)

    model.eval()
    with torch.no_grad():
        inputs = normalize_model_input(torch.from_numpy(xy_tau).to(device))
        k_pred = model(inputs)[:, 4].cpu().numpy()

    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame({'x': x, 'y': y, 'grf_interpolated': grf_k, 'pred_k': k_pred}).to_csv(
        os.path.join(results_dir, 'k_grf_comparison.csv'), index=False)


def save_mu_h_j0_at_gt_c(model, data_path=CONCENTRATION_CSV_PATH, results_dir=RESULTS_DIR):
    """evaluates mu_h (a closed-form function of c) and j_0 (via netC, which takes only
    c as input) directly at the GT concentration values instead of the model's own
    predicted c, isolating the learned/assumed c-dependent relationships from any noise
    in the model's own concentration fit -- for a cleaner comparison against chem.csv's
    and identified_j0_curve.csv's reference curves"""
    in_data, out_data, _, _ = extract_c_data(data_path, num_test_slices=0)
    x = in_data[:NUM_POS, 0]
    y = in_data[:NUM_POS, 1]

    model.eval()
    cols = {'x': x, 'y': y}
    with torch.no_grad():
        for i in range(NUM_T_STATES):
            c_true = torch.from_numpy(out_data[i * NUM_POS:(i + 1) * NUM_POS]).to(device)
            mu_h_at_gt_c = get_mu_h(c_true).squeeze(1).cpu().numpy()
            log_f = model.netC(c_true)
            j0_at_gt_c = (c_true * (1 - c_true) * torch.exp(log_f)).squeeze(1).cpu().numpy()
            t_val = TIME_SLICES[i]
            cols[f'gt_c{t_val}'] = out_data[i * NUM_POS:(i + 1) * NUM_POS].squeeze(1)
            cols[f'mu_h_at_gt_c{t_val}'] = mu_h_at_gt_c
            cols[f'j0_at_gt_c{t_val}'] = j0_at_gt_c

    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(cols).to_csv(os.path.join(results_dir, 'mu_h_j0_at_gt_c.csv'), index=False)

# ======= main =======


def main():
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    adam_train_loader, test_loader = load_data()
    interp_train_loader = interp_loader()

    netA = PINN(in_channels=IN_CH_ONE, out_channels=OUT_CH_ONE)
    netB = PINN(in_channels=IN_CH_TWO, out_channels=OUT_CH_TWO)
    netC = PINN(in_channels=IN_CH_THREE, out_channels=OUT_CH_THREE, width=NETC_WIDTH, depth=NETC_DEPTH)

    pinn = AllenCahnPINN(netA, netB, netC).to(device)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (best_loss, per_step_total, per_step_terms, per_step_lambdas, per_step_test_total,
     per_step_test_terms, adam_epochs_run, per_step_k_stats) = train_pinn_adam_only(
        model=pinn, adam_loader=adam_train_loader, interp_data_loader=interp_train_loader,
        lambdas=DEFAULT_LAMBDAS, adam_epochs=NEPOCHS_ADAM,
        lr=LR, model_dir=MODEL_DIR, test_loader=test_loader,
        disable_early_stop=True,
    )

    # hand the QN phase Adam's best checkpoint (by the fixed-lambda ckpt_loss)
    # not necessarily its last epoch 
    pinn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    adam_final_lambdas = {name: float(per_step_lambdas[name][-1]) for name in ALL_TERM_NAMES}

    qn_full_loader = full_batch_loader()
    qn_inputs, qn_labels = next(iter(qn_full_loader))
    interp_inputs, interp_labels = next(iter(interp_train_loader))
    x_vals_np, y_vals_np = load_domain_xy()

    (qn_best_loss, qn_per_iter_total, qn_per_iter_terms, qn_per_iter_lambdas,
     qn_per_iter_test_total, qn_per_iter_test_terms, qn_iters_run, qn_per_iter_k_stats) = train_pinn_qn(
        model=pinn, qn_inputs=qn_inputs, qn_labels=qn_labels,
        interp_inputs=interp_inputs, interp_labels=interp_labels,
        lambdas=adam_final_lambdas, x_vals_np=x_vals_np, y_vals_np=y_vals_np,
        model_dir=MODEL_DIR, test_loader=test_loader,
        # patience >= max_iters makes epochs_since_improvement unreachable within
        # the loop, i.e. no early stop -- run the full QN budget regardless
        patience=QN_MAX_ITERS,
    )

    combined_total = np.concatenate([per_step_total, qn_per_iter_total])
    combined_terms = {name: np.concatenate([per_step_terms[name], qn_per_iter_terms[name]])
                       for name in ALL_TERM_NAMES}
    combined_lambdas = {name: np.concatenate([per_step_lambdas[name], qn_per_iter_lambdas[name]])
                         for name in ALL_TERM_NAMES}
    combined_test_total = np.concatenate([per_step_test_total, qn_per_iter_test_total])
    combined_test_terms = {name: np.concatenate([per_step_test_terms[name], qn_per_iter_test_terms[name]])
                            for name in ALL_TERM_NAMES}
    combined_k_stats = {stat: np.concatenate([per_step_k_stats[stat], qn_per_iter_k_stats[stat]])
                         for stat in ('std', 'min', 'max')}

    save_loss_csv(combined_total, combined_terms, combined_lambdas, adam_epochs=adam_epochs_run,
                  test_total=combined_test_total, test_terms=combined_test_terms,
                  k_stats=combined_k_stats)

    pinn.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    total_eval_loss, __ = eval_pinn(pinn, test_loader, DEFAULT_LAMBDAS)

    for i in range(6):
        save_preds(pinn, i)
    save_mu_h_preds(pinn)
    save_k_at_grf_points(pinn)
    save_mu_h_j0_at_gt_c(pinn)


if __name__ == "__main__":
    main()
