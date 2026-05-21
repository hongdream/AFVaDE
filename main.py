import argparse
import math

from dataloader import *
from experiment import validation
from model import VaDE, vae_avg, pre_train, gmm_sampling
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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

def AFVaDE(args):
    csv_path = f"./csv/{args.dataset}.csv"

    args.nClusters, DL, (X, y), size = make_dataset(csv_path, args.client_num, args.batch_size,
                                                                  args.dataset)
    args.input_dim = X.shape[1]
    args.client_num = len(DL)
    size = np.array(size)

    device = torch.device("cuda" if args.cuda else "cpu")
    server = {
        'vade': VaDE(args),
        'optimizer': None,
        'lr_s': None
    }

    server['vade'] = server['vade'].to(device)

    clients = []
    for i, dl in enumerate(DL):
        clients.append({
            'client_id': i,
            'dataloader': dl,
            'vade': VaDE(args),
            'optimizer': None,
            'lr_s': None
        })

        clients[i]['vade'].load_state_dict(server['vade'].state_dict())
        clients[i]['vade'] = clients[i]['vade'].to(device)

    if os.path.exists(f'./model/{args.dataset}.pk'):
        server['vade'].load_state_dict(torch.load(f'./model/{args.dataset}.pk'))

        iter_bar = tqdm(range(args.iters))
        Result = []
        for iter in iter_bar:
            pre = []
            tru = []

            with torch.no_grad():
                for c in clients:
                    for x, y in c['dataloader']:
                        x = x.to(device)

                        tru.append(y.numpy())
                        pre.append(server['vade'].predict(x))

            tru = np.concatenate(tru, 0)
            pre = np.concatenate(pre, 0)

            purity, ari, nmi, acc= validation(tru, pre)
            tqdm.write(f"Purity: {purity:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}, ACC: {acc:.4f}")
            Result.append([purity, ari, nmi, acc])
        Result = np.stack(Result)
        mean = np.mean(Result, axis=0)
        std = np.std(Result, axis=0)

        print("===============Federated Clustering Performance===============")
        print(f"Data: {args.dataset}, Client: {args.client_num}")
        print(f"Purity: {mean[0]:.4f}±{std[0]:.3f}")
        print(f"ARI: {mean[1]:.4f}±{std[1]:.3f}")
        print(f"NMI: {mean[2]:.4f}±{std[2]:.3f}")
        print(f"ACC: {mean[3]:.4f}±{std[3]:.3f}")
        print("==============================================================")
        return

    weight = size / sum(size)
    pre_train(args, server, clients, size, pre_epoch=args.pre)

    server['optimizer'] = Adam(server['vade'].parameters(), lr=args.lr)
    server['lr_s'] = StepLR(server['optimizer'], step_size=args.stepsize, gamma=args.gamma)

    for c in clients:
        c['optimizer'] = Adam(c['vade'].parameters(), lr=args.lr)
        c['lr_s'] = StepLR(c['optimizer'], step_size=args.stepsize, gamma=args.gamma)

    participation_prob = np.random.rand(args.client_num)
    client_participate_times = np.zeros(args.client_num)
    commu_weight = np.ones(args.client_num)

    epoch_bar = tqdm(range(args.epoch))
    Result = []
    vae_norms_history = []
    gmm_mu_norms_history = []
    for epoch in epoch_bar:

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

        weight = size[participate_clients_idx] / sum(size[participate_clients_idx])
        participate_weight = weight * commu_weight[participate_clients_idx]
        participate_weight = participate_weight / sum(participate_weight)
        participate_weight = torch.from_numpy(participate_weight)

        clients_vae_param = []
        clients_vae_old = []
        L_c = 0
        for c in participate_clients:
            old_state_snapshot = {
                key: value.clone().detach().cpu()
                for key, value in c['vade'].state_dict().items()
            }

            clients_vae_old.append(old_state_snapshot)
            for x, _ in c['dataloader']:
                c['optimizer'].zero_grad()

                x = x.to(device)
                loss = c['vade'].ELBO_Loss(x, args.alpha)
                L_c += loss.detach().cpu().numpy()

                loss.backward()
                c['optimizer'].step()
                c['lr_s'].step()

            clients_vae_param.append(c['vade'].state_dict())

        L_c /= len(participate_clients)

        clients_vae_updates = []

        for old_state, new_state in zip(clients_vae_old, clients_vae_param):
            client_delta = {}

            for key in old_state.keys():
                client_delta[key] = new_state[key].cpu() - old_state[key].cpu()

            clients_vae_updates.append(client_delta)

        server_vade_param = vae_avg(server['vade'].state_dict(), clients_vae_updates, participate_weight)
        server['vade'].load_state_dict(server_vade_param, strict=False)

        L_s = 0
        clients_gmm_param = []
        for c in participate_clients:
            clients_gmm_param.append({
                'rho_': c['vade'].rho_.data.clone().to('cpu'),
                'mu_c': c['vade'].mu_c.data.clone().to('cpu'),
                'log_sigma2_c': c['vade'].log_sigma2_c.data.clone().to('cpu')
            })

        Z_hat, y_hat, lengths, _, _, _ = gmm_sampling(clients_gmm_param, participate_weight, args.sample)
        Z_hat = torch.from_numpy(Z_hat).float()
        Z_hat = Z_hat.to(device)

        X_hat = server['vade'].decoder(Z_hat)

        num_clients = len(lengths)
        N, D = Z_hat.shape

        decoded_X_hat = torch.zeros_like(X_hat)

        start_index = 0

        for client_id in range(num_clients):
            num_clusters_for_client = lengths[client_id]

            client_labels = []

            for cluster_id in range(num_clusters_for_client):
                cluster_label = start_index + cluster_id
                client_labels.append(cluster_label)

            for label in client_labels:
                client_data_points = np.where(y_hat == label)[0]

                client_X_points = Z_hat[client_data_points]

                client_X_points = client_X_points.to(device)
                decoded_client_X = clients[client_id]['vade'].decoder(client_X_points)

                decoded_X_hat[client_data_points] = decoded_client_X

            start_index += num_clusters_for_client

        loss = server['vade'].ELBO_Loss_server(X_hat, decoded_X_hat, args.alpha)
        L_s += loss.detach().cpu().numpy()

        server['optimizer'].zero_grad()
        loss.backward()
        server['optimizer'].step()
        server['lr_s'].step()

        for c in participate_clients:
            c['vade'].load_state_dict(server_vade_param, strict=False)

    iter_bar = tqdm(range(args.iters))
    Result = []
    for iter in iter_bar:
        pre = []
        tru = []

        with torch.no_grad():
            for c in clients:
                for x, y in c['dataloader']:
                    x = x.to(device)

                    tru.append(y.numpy())
                    pre.append(server['vade'].predict(x))

        tru = np.concatenate(tru, 0)
        pre = np.concatenate(pre, 0)

        purity, ari, nmi, acc = validation(tru, pre)
        Result.append([purity, ari, nmi, acc])
    Result = np.stack(Result)
    mean = np.mean(Result, axis=0)

    print(f"Purity: {mean[0]:.4f}, ARI: {mean[1]:.4f}, NMI: {mean[2]:.4f}, ACC: {mean[3]:.4f}")


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size', type=int, default=1024)
    parse.add_argument('--dataset', type=str, default='iris')
    parse.add_argument('--nClusters', type=int, default=0)
    parse.add_argument('--input_dim', type=int, default=0)
    parse.add_argument('--hid_dim', type=int, default=10)
    parse.add_argument('--cuda', type=bool, default=True)
    parse.add_argument('--client_num', type=int, default=10)

    parse.add_argument('--lr', type=float, default=2e-3)
    parse.add_argument('--stepsize', type=int, default=10)
    parse.add_argument('--gamma', type=float, default=0.85)

    parse.add_argument('--pre', type=int, default=50)
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--iters', type=int, default=5)
    parse.add_argument('--alpha', type=float, default=0.01)
    parse.add_argument('--sample', type=int, default=200)
    parse.add_argument('--xi', type=float, default=1)
    args = parse.parse_args()

    try:
        AFVaDE(args)
    except Exception as e:
        print(e)