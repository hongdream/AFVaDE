from munkres import Munkres

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, fowlkes_mallows_score


def labelMapping(y_true, y_pred):
    Label1 = np.unique(y_true)
    Label2 = np.unique(y_pred)
    G = np.zeros((len(Label1), len(Label2)))
    for i in range(len(Label1)):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(len(Label2)):
            ind_cla2 = y_pred == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)

    if len(Label1) != len(Label2):
        max_size = max(len(Label1), len(Label2))
        padded_G = np.zeros((max_size, max_size))
        padded_G[:len(Label1), :len(Label2)] = G

        padded_G[len(Label1):, :] = np.max(G) + 1
        padded_G[:, len(Label2):] = np.max(G) + 1
        G = padded_G

    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    new_label = np.zeros(y_pred.shape)
    for i in range(len(Label2)):
        if c[i] < len(Label1):
            new_label[y_pred == Label2[i]] = Label1[c[i]]
        else:
            new_label[y_pred == Label2[i]] = -1

    return new_label.astype(int)


def PURITY(y_true, y_pred):
    clusters = np.unique(y_pred)
    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(y_pred == c)[0]
        y_temp = y_true[idx, :].reshape(-1)
        count.append(np.bincount(y_temp).max())
    return np.sum(count) / y_true.shape[0]

def ARI(y_true, y_pred, beta=1.):
    score = adjusted_rand_score(y_true, y_pred)
    return score

def NMI(y_true, y_pred):
    score = normalized_mutual_info_score(y_pred, y_true)
    return score

def AC(y_true, y_pred):
    score = accuracy_score(y_true, y_pred)
    return score


def validation(y_true, y_pred):
    y_pred = labelMapping(y_true, y_pred)

    purity = PURITY(y_true, y_pred)
    ari = ARI(y_true, y_pred)
    nmi = NMI(y_true, y_pred)
    acc = AC(y_true, y_pred)

    return purity, ari, nmi, acc