import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np
import warnings

warnings.filterwarnings('ignore', message="KMeans is known to have a memory leak on Windows with MKL")
warnings.filterwarnings('ignore', category=RuntimeWarning, message="covariance is not symmetric positive-semidefinite")


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

def vae_avg(server_dict, clients_update_dicts, weights):
    aggregation_dict = {}
    skip_keys = {'rho_', 'mu_c', 'log_sigma2_c'}

    for key in clients_update_dicts[0].keys():
        if key in skip_keys:
            continue

        stacked = torch.stack([sd[key].detach().cpu() for sd in clients_update_dicts], dim=0)
        participate_weight = weights.to(stacked.device)
        weighted_avg = torch.sum(stacked * participate_weight.view(-1, *([1] * (stacked.dim() - 1))), dim=0)
        aggregation_dict[key] = server_dict[key].detach().cpu() + weighted_avg

    return aggregation_dict

def gmm_sampling(state_dicts, weights, nums=10000, random_state=None):
    pi_list = []
    mu_list = []
    sigma_list = []
    lengths = []

    for sd in state_dicts:
        rho = sd['rho_'].cpu()
        pi = torch.softmax(rho, dim=0)

        pi_list.append(pi)
        mu_list.append(sd['mu_c'].cpu())
        sigma_list.append(sd['log_sigma2_c'].cpu())
        lengths.append(len(pi))

    pi_cat = torch.cat(pi_list, dim=0)
    mu_cat = torch.cat(mu_list, dim=0)
    log_sigma_cat = torch.cat(sigma_list, dim=0)

    expanded_weights = torch.repeat_interleave(weights, torch.tensor(lengths))
    global_pi = (pi_cat * expanded_weights).numpy()

    if np.any(np.isnan(global_pi)):
        global_pi = np.nan_to_num(global_pi, nan=0.0)

    total_prob = np.sum(global_pi)
    if np.isclose(total_prob, 0.0):
        global_pi = np.ones_like(global_pi) / len(global_pi)
    else:
        global_pi = global_pi / total_prob

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
            Z_hat[y_hat == k] = np.random.multivariate_normal(mu_cat_np[k], cov, n_k)

    return Z_hat, y_hat, lengths, global_pi, mu_cat_np, sigma2_cat_np


def block(in_c, out_c):
    layers = [
        nn.Linear(in_c, out_c),
        nn.ReLU(True)
    ]
    return layers


def pre_train(args, server, clients, size, pre_epoch=10):
    epsilon = 1e-10
    device = torch.device("cuda" if args.cuda else "cpu")
    Loss = nn.MSELoss()
    print('Pretraining......')
    for c in clients:
        c['optimizer'] = Adam(itertools.chain(c['vade'].encoder.parameters(), c['vade'].decoder.parameters()))

    participation_prob = np.random.rand(args.client_num)
    client_participate_times = np.zeros(args.client_num)
    commu_weight = np.ones(args.client_num)

    epoch_bar = tqdm(range(pre_epoch))
    for _ in epoch_bar:
        participate_clients_idx = []
        for idx, _ in enumerate(clients):
            if np.random.rand() < participation_prob[idx]:
                participate_clients_idx.append(idx)
        if len(participate_clients_idx) > 0:
            client_participate_times[participate_clients_idx] += 1
        total_participation = np.sum(client_participate_times)
        if total_participation > 0:
            commu_weight = args.xi / (args.xi + client_participate_times / total_participation)
        else:
            commu_weight = np.ones(args.client_num)

        participate_clients = [clients[i] for i in participate_clients_idx]

        weights = size[participate_clients_idx] / sum(size[participate_clients_idx])
        participate_weight = weights * commu_weight[participate_clients_idx]
        participate_weight = participate_weight / sum(participate_weight)
        participate_weight = torch.from_numpy(participate_weight)

        clients_vae_param = []
        clients_vae_old = []
        L = 0
        for c in participate_clients:
            old_state_snapshot = {
                key: value.clone().detach().cpu()
                for key, value in c['vade'].state_dict().items()
            }

            clients_vae_old.append(old_state_snapshot)
            for x, y in c['dataloader']:
                x = x.to(device)
                z, _ = c['vade'].encoder(x)
                x_ = c['vade'].decoder(z)
                loss = Loss(x, x_)
                L += loss.item()

                c['optimizer'].zero_grad()
                loss.backward()
                c['optimizer'].step()
            clients_vae_param.append(c['vade'].state_dict())

        clients_vae_updates = []

        for old_state, new_state in zip(clients_vae_old, clients_vae_param):
            client_delta = {}

            for key in old_state.keys():
                client_delta[key] = new_state[key].cpu() - old_state[key].cpu()

            clients_vae_updates.append(client_delta)

        server_vade_param = vae_avg(server['vade'].state_dict(), clients_vae_updates, participate_weight)
        server['vade'].load_state_dict(server_vade_param, strict=False)

        for c in clients:
            c['vade'].load_state_dict(server['vade'].state_dict())

    for c in clients:
        c['vade'].encoder.log_sigma2_l.load_state_dict(c['vade'].encoder.mu_l.state_dict())
    server['vade'].encoder.log_sigma2_l.load_state_dict(server['vade'].encoder.mu_l.state_dict())

    clients_gmm_param = []
    with torch.no_grad():
        for c in clients:
            local_mu = []
            local_y = []
            for x, y in c['dataloader']:
                x = x.to(device)
                mu, sigma = c['vade'].encoder(x)
                assert F.mse_loss(mu, sigma) == 0
                local_mu.append(mu)
                local_y.append(y)

            local_mu = torch.cat(local_mu, 0).cpu().numpy()
            local_y = torch.cat(local_y, 0).cpu().numpy()

            K_l = len(np.unique(local_y))
            gmm = GaussianMixture(n_components=K_l, covariance_type='diag', random_state=42)
            gmm.fit(local_mu)

            pi_ = torch.tensor(gmm.weights_, dtype=torch.float, device=device)
            c['vade'].rho_.data = torch.log(pi_ + epsilon)
            c['vade'].mu_c.data = torch.tensor(gmm.means_, dtype=torch.float, device=device)
            c['vade'].log_sigma2_c.data = torch.log(torch.tensor(gmm.covariances_, dtype=torch.float, device=device))

            clients_gmm_param.append({
                'rho_': c['vade'].rho_.data.clone(),
                'mu_c': c['vade'].mu_c.data.clone(),
                'log_sigma2_c': c['vade'].log_sigma2_c.data.clone()
            })

    weights = size / sum(size)
    commu_weight = args.xi / (args.xi + client_participate_times / np.sum(client_participate_times))
    participate_weight = weights * commu_weight
    participate_weight = participate_weight / sum(participate_weight)
    participate_weight = torch.from_numpy(participate_weight)

    Z_hat, _, _, _, _, _ = gmm_sampling(clients_gmm_param, participate_weight, args.sample)

    gmm_global = GaussianMixture(n_components=args.nClusters, covariance_type='diag', random_state=42)
    _ = gmm_global.fit_predict(Z_hat)

    pi_ = torch.tensor(gmm_global.weights_, dtype=torch.float, device=device)
    server['vade'].rho_.data = torch.log(pi_ + epsilon)
    server['vade'].mu_c.data = torch.tensor(gmm_global.means_, dtype=torch.float, device=device)
    server['vade'].log_sigma2_c.data = torch.log(torch.tensor(gmm_global.covariances_, dtype=torch.float, device=device))


class Encoder(nn.Module):
    def __init__(self, args, inter_dims=[500,500,2000]):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            *block(args.input_dim, inter_dims[0]),
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )

        self.mu_l = nn.Linear(inter_dims[-1], args.hid_dim)
        self.log_sigma2_l = nn.Linear(inter_dims[-1], args.hid_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.mu_l(e)
        log_sigma2 = self.log_sigma2_l(e)
        return mu, log_sigma2


class Decoder(nn.Module):
    def __init__(self, args, inter_dims=[2000,500,500]):
        super(Decoder, self).__init__()

        self.decoder=nn.Sequential(
            *block(args.hid_dim, inter_dims[-1]),
            *block(inter_dims[-1], inter_dims[-2]),
            *block(inter_dims[-2], inter_dims[-3]),
            nn.Linear(inter_dims[-3], args.input_dim),
        )

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

        L_c_s = 0
        L_c_s += F.mse_loss(x_l, x_s)
        Loss = L_c_s * x_s.size(1)

        L_rec = 0
        z_mu, z_sigma2_log = self.encoder(x_s)
        for l in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
            x_pro = self.decoder(z)
            L_rec += F.mse_loss(x_pro, x_s)

        L_rec /= L
        Loss += L_rec * x_s.size(1)

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