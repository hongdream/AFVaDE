import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


warnings.filterwarnings('ignore', message="KMeans is known to have a memory leak on Windows with MKL")
warnings.filterwarnings('ignore', category=RuntimeWarning, message="covariance is not symmetric positive-semidefinite")
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message="Could not find the number of physical cores.*")


GMM_LOG_VARIANCE_MIN = -20.0
GMM_LOG_VARIANCE_MAX = 20.0
GMM_DIAGNOSTIC_COLUMNS = (
    "gmm_numeric_issue",
    "gmm_sampling_calls",
    "gmm_repair_calls",
    "gmm_repaired_values",
    "gmm_invalid_components",
    "gmm_all_invalid_fallback_calls",
    "gmm_weight_fallback_calls",
)


@dataclass
class GMMSamplingDiagnostics:

    sampling_calls: int = 0
    repair_calls: int = 0
    repaired_values: int = 0
    invalid_components: int = 0
    all_invalid_fallback_calls: int = 0
    weight_fallback_calls: int = 0

    def csv_summary(self):
        return {
            "gmm_numeric_issue": int(self.repair_calls > 0),
            "gmm_sampling_calls": self.sampling_calls,
            "gmm_repair_calls": self.repair_calls,
            "gmm_repaired_values": self.repaired_values,
            "gmm_invalid_components": self.invalid_components,
            "gmm_all_invalid_fallback_calls": (
                self.all_invalid_fallback_calls
            ),
            "gmm_weight_fallback_calls": self.weight_fallback_calls,
        }


_ACTIVE_GMM_DIAGNOSTICS = ContextVar(
    "active_gmm_sampling_diagnostics",
    default=None,
)


@contextmanager
def collect_gmm_sampling_diagnostics():

    diagnostics = GMMSamplingDiagnostics()
    token = _ACTIVE_GMM_DIAGNOSTICS.set(diagnostics)
    try:
        yield diagnostics
    finally:
        _ACTIVE_GMM_DIAGNOSTICS.reset(token)


def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / Y_pred.size

    return acc, w


def vae_avg(state_dicts, weights):
    weights = weights / weights.sum()
    server_dict = {}
    skip_keys = {'rho_', 'mu_c', 'log_sigma2_c'}

    for key in state_dicts[0].keys():
        if key in skip_keys:
            continue

        stacked = torch.stack([sd[key].detach().cpu() for sd in state_dicts], dim=0)
        weighted_avg = torch.sum(stacked * weights.view(-1, *([1] * (stacked.dim() - 1))), dim=0)
        server_dict[key] = weighted_avg


    return server_dict


def gmm_sampling(state_dicts, weights, nums=10000, random_state=None):

    diagnostics = _ACTIVE_GMM_DIAGNOSTICS.get()
    if diagnostics is not None:
        diagnostics.sampling_calls += 1

    if not state_dicts:
        raise ValueError("gmm_sampling requires at least one GMM state")
    if int(nums) != nums or nums < 0:
        raise ValueError(f"nums must be a non-negative integer, got {nums!r}")
    nums = int(nums)

    client_weights = (
        torch.as_tensor(weights, dtype=torch.float64)
        .detach()
        .cpu()
        .flatten()
        .clone()
    )
    if client_weights.numel() != len(state_dicts):
        raise ValueError(
            "GMM state/weight length mismatch: "
            f"states={len(state_dicts)}, weights={client_weights.numel()}"
        )

    repair_count = 0
    invalid_weights = ~torch.isfinite(client_weights) | (client_weights < 0)
    repair_count += int(invalid_weights.sum().item())
    client_weights[invalid_weights] = 0.0

    pi_list = []
    mu_list = []
    log_sigma_list = []
    lengths = []
    valid_clients = []
    invalid_component_count = 0

    for client_index, state in enumerate(state_dicts):
        missing_keys = {
            "rho_", "mu_c", "log_sigma2_c"
        } - set(state)
        if missing_keys:
            raise KeyError(
                f"GMM state {client_index} is missing keys: "
                f"{sorted(missing_keys)}"
            )

        rho = (
            torch.as_tensor(state["rho_"], dtype=torch.float64)
            .detach()
            .cpu()
            .flatten()
            .clone()
        )
        mu = (
            torch.as_tensor(state["mu_c"], dtype=torch.float64)
            .detach()
            .cpu()
            .clone()
        )
        log_sigma = (
            torch.as_tensor(state["log_sigma2_c"], dtype=torch.float64)
            .detach()
            .cpu()
            .clone()
        )
        if rho.numel() == 0:
            raise ValueError(f"GMM state {client_index} has no components")
        if mu.ndim != 2 or log_sigma.ndim != 2:
            raise ValueError(
                f"GMM state {client_index} requires 2-D mu/log-variance tensors"
            )
        if mu.shape != log_sigma.shape or mu.shape[0] != rho.numel():
            raise ValueError(
                f"GMM state {client_index} has incompatible shapes: "
                f"rho={tuple(rho.shape)}, mu={tuple(mu.shape)}, "
                f"log_sigma2={tuple(log_sigma.shape)}"
            )
        if mu.shape[1] == 0:
            raise ValueError(f"GMM state {client_index} has zero latent dimension")

        valid_components = (
            torch.isfinite(rho)
            & torch.isfinite(mu).all(dim=1)
            & torch.isfinite(log_sigma).all(dim=1)
        )
        invalid_component_count += int((~valid_components).sum().item())
        invalid_value_count = (
            int((~torch.isfinite(rho)).sum().item())
            + int((~torch.isfinite(mu)).sum().item())
            + int((~torch.isfinite(log_sigma)).sum().item())
        )
        repair_count += invalid_value_count


        rho = torch.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        log_sigma = torch.nan_to_num(
            log_sigma,
            nan=0.0,
            posinf=GMM_LOG_VARIANCE_MAX,
            neginf=GMM_LOG_VARIANCE_MIN,
        )
        outside_log_variance_range = (
            (log_sigma < GMM_LOG_VARIANCE_MIN)
            | (log_sigma > GMM_LOG_VARIANCE_MAX)
        )
        repair_count += int(outside_log_variance_range.sum().item())
        log_sigma.clamp_(GMM_LOG_VARIANCE_MIN, GMM_LOG_VARIANCE_MAX)

        pi = torch.softmax(rho, dim=0)
        pi[~valid_components] = 0.0
        valid_mass = float(pi.sum().item())
        client_is_valid = bool(valid_mass > 0.0 and np.isfinite(valid_mass))
        if client_is_valid:
            pi /= valid_mass
        else:
            client_weights[client_index] = 0.0

        pi_list.append(pi)
        mu_list.append(mu)
        log_sigma_list.append(log_sigma)
        lengths.append(int(rho.numel()))
        valid_clients.append(client_is_valid)

    if len({int(mu.shape[1]) for mu in mu_list}) != 1:
        raise ValueError("All client GMM states must use the same latent dimension")

    all_invalid_fallback = not any(valid_clients)
    weight_fallback = False
    if all_invalid_fallback:


        client_weights.fill_(1.0)
        for index, pi in enumerate(pi_list):
            pi.copy_(torch.softmax(
                torch.nan_to_num(
                    torch.as_tensor(
                        state_dicts[index]["rho_"], dtype=torch.float64
                    ).detach().cpu().flatten(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dim=0,
            ))
    elif float(client_weights.sum().item()) <= 0.0:

        client_weights.copy_(torch.as_tensor(valid_clients, dtype=torch.float64))
        weight_fallback = True

    client_weights /= client_weights.sum()
    pi_cat = torch.cat(pi_list, dim=0)
    mu_cat = torch.cat(mu_list, dim=0)
    log_sigma_cat = torch.cat(log_sigma_list, dim=0)
    expanded_weights = torch.repeat_interleave(
        client_weights,
        torch.as_tensor(lengths, dtype=torch.long),
    )
    global_pi = (pi_cat * expanded_weights).numpy()
    global_pi = np.nan_to_num(global_pi, nan=0.0, posinf=0.0, neginf=0.0)
    global_pi = np.clip(global_pi, 0.0, None)
    total_prob = float(global_pi.sum())
    if not np.isfinite(total_prob) or total_prob <= 0.0:
        raise RuntimeError(
            "GMM probability repair failed to produce positive finite mass"
        )
    global_pi /= total_prob

    if repair_count or all_invalid_fallback or weight_fallback:
        if diagnostics is not None:
            diagnostics.repair_calls += 1
            diagnostics.repaired_values += repair_count
            diagnostics.invalid_components += invalid_component_count
            diagnostics.all_invalid_fallback_calls += int(all_invalid_fallback)
            diagnostics.weight_fallback_calls += int(weight_fallback)
        warnings.warn(
            "gmm_sampling repaired invalid GMM input "
            f"(values={repair_count}, all_invalid_fallback="
            f"{all_invalid_fallback}, weight_fallback={weight_fallback})",
            RuntimeWarning,
            stacklevel=2,
        )

    if random_state is not None:
        np.random.seed(random_state)

    K = len(global_pi)
    d = mu_cat.shape[1]
    y_hat = np.random.choice(K, size=nums, p=global_pi)
    Z_hat = np.zeros((nums, d))
    mu_cat_np = mu_cat.numpy()
    sigma2_cat_np = torch.exp(log_sigma_cat).numpy()

    for k in range(K):
        n_k = np.sum(y_hat == k)
        if n_k > 0:
            cov = np.diag(sigma2_cat_np[k])
            Z_hat[y_hat == k] = np.random.multivariate_normal(
                mu_cat_np[k], cov, n_k
            )

    return Z_hat, y_hat, lengths, global_pi, mu_cat_np, sigma2_cat_np


def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.ReLU(True)
    ]

    return layers


def _network_dims(args):

    dims = tuple(int(value) for value in getattr(
        args,
        "network",
        (500, 500, 2000),
    ))
    if not dims:
        raise ValueError("network must contain at least one hidden layer")
    if any(value <= 0 for value in dims):
        raise ValueError("network must contain positive integers")
    return dims


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()


        self.hidden_dims = _network_dims(args)
        layers = []
        previous_dim = args.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(block(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        self.mu_l = nn.Linear(previous_dim, args.hid_dim)
        self.log_sigma2_l = nn.Linear(previous_dim, args.hid_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.mu_l(e)
        log_sigma2 = self.log_sigma2_l(e)

        return mu, log_sigma2


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()


        self.hidden_dims = tuple(reversed(_network_dims(args)))
        layers = []
        previous_dim = args.hid_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(block(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, args.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        x_pro = self.decoder(z)

        return x_pro


class VaDE(nn.Module):
    def __init__(self, args):


        super(VaDE, self).__init__()

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)


        self.mu_c = nn.Parameter(torch.Tensor(args.nClusters, args.hid_dim))
        nn.init.xavier_normal_(self.mu_c)


        self.rho_ = nn.Parameter(torch.Tensor(args.nClusters))
        nn.init.uniform_(self.rho_, a=-1, b=1)


        self.log_sigma2_c = nn.Parameter(torch.Tensor(args.nClusters, args.hid_dim))
        nn.init.constant_(self.log_sigma2_c, 0)

        from types import SimpleNamespace
        self.args = SimpleNamespace(**vars(args))


    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        rho = self.rho_
        pi = torch.softmax(rho, dim=0)
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def ELBO_Loss(self, x, alpha, L=1):
        det = 1e-10

        L_rec = 0

        z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu

            x_pro = self.decoder(z)

            L_rec += F.mse_loss(x_pro, x)

        L_rec /= L

        Loss = L_rec * x.size(1)

        rho = self.rho_
        pi = torch.softmax(rho, dim=0)
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))

        Loss += alpha * 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                              torch.exp(
                                                                  z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(
                                                                      0)) +
                                                              (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(
                                                                  2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        Loss -= alpha * (torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + z_sigma2_log, 1)))

        return Loss

    def ELBO_Loss_server(self, x_s, x_l, alpha, L=1):
        det = 1e-10


        L_rec = 0

        z_mu, z_sigma2_log = self.encoder(x_s)
        for l in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu

            x_pro = self.decoder(z)

            L_rec += F.mse_loss(x_pro, x_s)

        L_rec /= L

        Loss = L_rec * x_s.size(1)

        rho = self.rho_
        pi = torch.softmax(rho, dim=0)
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det

        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))

        Loss += alpha * 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                              torch.exp(
                                                                  z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(
                                                                      0)) +
                                                              (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(
                                                                  2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        Loss -= alpha * (torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + z_sigma2_log, 1)))

        return Loss


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(mus.size(0)):
            G.append(self.gaussian_pdf_log(x, mus[c:c+1, :], log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G, 1)


    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))
