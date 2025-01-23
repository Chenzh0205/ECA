import copy
import gc
import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

from eval import *
from misc import *


def Find_rank(scores):
    _, idx = scores.detach().flatten().sort()
    return idx.detach()


def FRL_Vote(FLmodel, user_updates, initial_scores):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts = torch.sort(user_updates[str(n)])[1]
            sum_args_sorts = torch.sum(args_sorts, 0)
            idxx = torch.sort(sum_args_sorts)[1]
            temp1 = m.scores.detach().clone()
            temp1.flatten()[idxx] = initial_scores[str(n)]
            m.scores = torch.nn.Parameter(temp1)
            del idxx, temp1



def FLTrust(layer_name, user_updates):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    user_update = user_updates.float().cuda()
    server_params = user_update[0]
    server_norm = torch.norm(server_params)
    param_list = user_update[1:]

    ts = torch.zeros(len(param_list)).cuda()
    for i in range(len(param_list)):
        ts[i] = max(cos(server_params, param_list[i]), 0)
        param_list[i] = (server_norm / torch.norm(param_list[i])) * param_list[i] * ts[i]

    args_sorts = torch.sort(param_list)[1]
    sum_args_sorts = torch.sum(args_sorts, 0)
    idxx = torch.sort(sum_args_sorts)[1]
    del args_sorts, sum_args_sorts, ts, param_list, server_params, server_norm, user_updates
    gc.collect()
    return idxx


def FRL_Defense_Vote(FLmodel, user_updates, initial_scores, AGR, path):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            if AGR == "Multi-krum":
                idxx = multi_krum(user_updates[str(n)].tolist(), len(user_updates))
            if AGR == "AFA":
                idxx = AFA(str(n), user_updates[str(n)].tolist(), path)
            if AGR == "FoolsGold":
                idxx = FoolsGold(str(n), user_updates[str(n)])
            if AGR == "FLTrust":
                idxx = FLTrust(str(n), user_updates[str(n)])
            if AGR == "DnC":
                idxx = Dnc(str(n), user_updates[str(n)], len(user_updates[str(n)])//10)
            if AGR == "FRL":
                args_sorts = torch.sort(user_updates[str(n)])[1]
                sum_args_sorts = torch.sum(args_sorts, 0)
                idxx = torch.sort(sum_args_sorts)[1]
                del sum_args_sorts, args_sorts

            temp1 = m.scores.detach().clone()
            temp1.flatten()[idxx] = initial_scores[str(n)]
            m.scores = torch.nn.Parameter(temp1)
            del idxx, temp1
        torch.cuda.empty_cache()


def euclidean_distance(seq1, seq2):
    return np.linalg.norm(seq1 - seq2)


def multi_krum(all_updates, n_user, multi_k=True):
    n_attackers = int(n_user // 10)
    candidates = []
    candidate_indices = []
    remaining_updates = np.array(all_updates)
    all_indices = np.arange(len(all_updates))

    distance_cache = {}

    def get_distance(i, j):
        if (i, j) in distance_cache:
            return distance_cache[(i, j)]
        if (j, i) in distance_cache:
            return distance_cache[(j, i)]
        distance = euclidean_distance(remaining_updates[i], remaining_updates[j])
        distance_cache[(i, j)] = distance
        return distance

    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []


        for i in range(len(remaining_updates)):
            distance = []
            for j in range(len(remaining_updates)):
                if i != j:
                    distance.append(get_distance(i, j))
                else:
                    distance.append(float('inf'))
            distances.append(distance)

        distances = np.array(distances)
        distances = np.sort(distances, axis=1)
        scores = np.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], axis=1)
        indices = np.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0]])
        all_indices = np.delete(all_indices, indices[0])
        candidates.append(remaining_updates[indices[0]])
        remaining_updates = np.delete(remaining_updates, indices[0], axis=0)

        if not multi_k:
            break

    user_updates_np = np.array(candidates)
    user_updates = torch.tensor(user_updates_np)
    args_sorts = torch.sort(user_updates)[1]
    sum_args_sorts = torch.sum(args_sorts, 0)
    idxx = torch.sort(sum_args_sorts)[1]

    aggregated_update = idxx
    return aggregated_update


def AFA1(n, client_updates, xi=2):
    num_clients = len(client_updates)
    distance_cache = {}
    cosine_similarities = np.zeros((num_clients, num_clients))

    cosine_similarities = cosine_similarity(client_updates)

    mean_similarities = np.mean(cosine_similarities, axis=1)
    median_similarity = np.median(mean_similarities)
    std_similarity = np.std(mean_similarities)

    g = set(range(num_clients))
    r = [1]
    r_total = set()
    while len(r):
        r = []
        if median_similarity < np.mean(mean_similarities):
            for k in list(g):
                if mean_similarities[k] < median_similarity - xi * std_similarity:
                    r.append(k)
                    r_total.add(k)
                    g.remove(k)
        else:
            for k in list(g):
                if mean_similarities[k] > median_similarity + xi * std_similarity:
                    r.append(k)
                    r_total.add(k)
                    g.remove(k)
        xi += 0.5

    if r_total:
        print(n, " attackers:", r_total)

    user_updates = torch.tensor([client_updates[k] for k in g])
    args_sorts = torch.sort(user_updates)[1]
    sum_args_sorts = torch.sum(args_sorts, 0)
    idxx = torch.sort(sum_args_sorts)[1]

    return idxx


def AFA(layer_name, client_updates, path, xi=2):
    num_clients = len(client_updates)

    file_path = str(path) + f"/afa_parameters_{layer_name}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            good_hist, bad_hist, alpha, beta = pickle.load(file)
    else:
        good_hist =np.zeros(num_clients, dtype=np.float32)
        bad_hist =np.zeros(num_clients, dtype=np.float32)
        alpha = 3
        beta = 3

    pvalues = np.zeros(num_clients)
    for i in range(num_clients):
        ngood = good_hist[i]
        nbad = bad_hist[i]
        alpha_i = alpha + ngood
        beta_i = beta + nbad
        pvalues[i] = alpha_i / (alpha_i + beta_i)

    cosine_similarities = cosine_similarity(client_updates)
    mean_similarities = np.mean(cosine_similarities, axis=1)
    median_similarity = np.median(mean_similarities)
    std_similarity = np.std(mean_similarities)

    g = set(range(num_clients))
    r = [1]
    r_total = set()
    while len(r):
        r = []
        if median_similarity < np.mean(mean_similarities):
            for k in list(g):
                if mean_similarities[k] < median_similarity - xi * std_similarity:
                    r.append(k)
                    r_total.add(k)
                    g.remove(k)
        else:
            for k in list(g):
                if mean_similarities[k] > median_similarity + xi * std_similarity:
                    r.append(k)
                    r_total.add(k)
                    g.remove(k)
        xi += 0.5

    if r_total:
        print(f"{layer_name} attackers:", r_total)

    for i in range(num_clients):
        if i in r_total:
            bad_hist[i] += 1
        else:
            good_hist[i] += 1

    with open(file_path, "wb") as file:
        pickle.dump((good_hist, bad_hist, alpha, beta), file)

    user_updates = torch.tensor([client_updates[k] for k in g])
    trust_weights = torch.tensor([pvalues[k] for k in g])

    args_sorts = torch.sort(user_updates)[1]
    weighted_updates = args_sorts * trust_weights.unsqueeze(1)
    sum_args_sorts = torch.sum(weighted_updates, 0)
    idxx = torch.sort(sum_args_sorts)[1]

    return idxx


def FoolsGold(layer_name, client_updates):
    num_workers = len(client_updates)
    client_update = client_updates.float().cuda()

    norm_client_updates = client_update / torch.norm(client_update, dim=1, keepdim=True)
    cosine_similarities = torch.mm(norm_client_updates, norm_client_updates.t())

    v = torch.max(cosine_similarities, dim=1)[0]

    for i in range(num_workers):
        mask = v > v[i]
        cosine_similarities[i, mask] *= v[i] / v[mask]

    alpha = 1 - torch.max(cosine_similarities, dim=1)[0]

    alpha = torch.clamp(alpha, 0, 1)
    alpha = alpha / alpha.max()
    alpha[alpha == 1] = 0.99
    alpha = torch.log(alpha / (1 - alpha)) + 0.5
    alpha = torch.clamp(alpha, 0, 1)
    alpha = alpha / alpha.sum()

    args_sorts = torch.sort(client_updates)[1]
    weighted_updates = args_sorts * alpha.unsqueeze(1).cuda()
    sum_args_sorts = torch.sum(weighted_updates, 0)
    idxx = torch.sort(sum_args_sorts)[1]

    del cosine_similarities, alpha, v, mask, norm_client_updates, client_updates, args_sorts, weighted_updates, sum_args_sorts
    torch.cuda.empty_cache()
    return idxx

def FRL_Fang_Vote(val_loader, FLmodel, user_updates, initial_scores, criterion, args, agr):
    n_attackers = int(len(user_updates['convs.0']) // 10)
    g = set(range(len(user_updates['convs.0'])))
    r = set()

    losses = []
    accuracies = []

    start_time = time.time()
    for i in range(len(user_updates['convs.0'])):
        mp = copy.deepcopy(FLmodel).cuda()
        for n, m in mp.named_modules():
            if hasattr(m, "scores"):
                idxx = user_updates[str(n)][i]
                temp1 = m.scores.detach().clone()
                temp1.flatten()[idxx] = initial_scores[str(n)]
                m.scores = torch.nn.Parameter(temp1)
                del idxx, temp1

        loss, acc = test(val_loader, mp, criterion, args)
        losses.append((i, loss))
        accuracies.append((i, acc))
        del mp
        torch.cuda.empty_cache()

    end_time = time.time()
    print("Test time: ", end_time - start_time)

    if agr == "Fang-ERR" or agr == "Fang-Union":
        accuracies.sort(key=lambda x: x[1])
        for i in range(n_attackers):
            client = accuracies[i][0]
            if client in g:
                r.add(client)
                g.remove(client)

    if agr == "Fang-LFR" or agr == "Fang-Union":
        losses.sort(key=lambda x: x[1], reverse=True)
        for i in range(n_attackers):
            client = losses[i][0]
            if client in g:
                r.add(client)
                g.remove(client)

    print("Malicious clients:", r)
    sorted_indices = sorted(g)
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            normal_updates = user_updates[str(n)][sorted_indices]
            args_sorts = torch.sort(normal_updates)[1]
            sum_args_sorts = torch.sum(args_sorts, 0)
            idxx = torch.sort(sum_args_sorts)[1]
            temp1 = m.scores.detach().clone()
            temp1.flatten()[idxx] = initial_scores[str(n)]
            m.scores = torch.nn.Parameter(temp1)
            del idxx, temp1, normal_updates

    del val_loader, FLmodel, user_updates, initial_scores, criterion, args, agr, losses, accuracies, sorted_indices
    torch.cuda.empty_cache()
    gc.collect()


def FABA(FLmodel, user_updates, initial_scores, n_attackers):
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            update_tensor = user_updates[str(n)].float()
            if len(concatenated_updates_tensor) == 0:
                concatenated_updates_tensor = update_tensor
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, update_tensor)])

    mean_update = torch.mean(concatenated_updates_tensor, dim=0)

    distances = torch.cdist(concatenated_updates_tensor.unsqueeze(0), mean_update.unsqueeze(0)).squeeze(0).squeeze()

    k = len(concatenated_updates_tensor) - n_attackers

    selected_indices = torch.topk(distances, k, largest=False).indices

    attacker_indices = torch.topk(distances, n_attackers, largest=True).indices
    print("Attackers:", attacker_indices)


    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            normal_updates = user_updates[str(n)][selected_indices]
            args_sorts = torch.sort(normal_updates)[1]
            sum_args_sorts = torch.sum(args_sorts, 0)
            idxx = torch.sort(sum_args_sorts)[1]
            temp1 = m.scores.detach().clone()
            temp1.flatten()[idxx] = initial_scores[str(n)]
            m.scores = torch.nn.Parameter(temp1)
            del idxx, temp1, normal_updates

    del FLmodel, user_updates, initial_scores, n_attackers, concatenated_updates_tensor, distances, selected_indices, attacker_indices
    torch.cuda.empty_cache()
    gc.collect()


def Dnc(layer, layer_updates, n_attackers, num_iters=2, filter_frac=1.0):
    num_clients, d = layer_updates.shape
    sub_dim = max(int(0.1 * d), 1)

    benign_ids = []
    for _ in range(num_iters):
        indices = torch.randperm(d)[:sub_dim]
        sub_updates = layer_updates[:, indices].float()

        with torch.no_grad():
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            _, _, v = torch.svd(centered_update, some=True)
            v = v[:, 0]

            s = torch.sum((sub_updates - mu) * v, dim=1) ** 2
            k = num_clients - int(filter_frac * n_attackers)
            good = s.topk(k, largest=False).indices.tolist()

        benign_ids.append(set(good))

    intersection_set = set.intersection(*benign_ids)
    benign_clients = list(intersection_set)

    all_client_ids = set(range(num_clients))
    attacker_ids = [client_id for client_id in all_client_ids if client_id not in benign_clients]
    print(layer + " Attacker clients:", attacker_ids)

    benign_updates = layer_updates[benign_clients]

    args_sorts = torch.sort(benign_updates)[1]
    sum_args_sorts = torch.sum(args_sorts, 0)
    idxx = torch.sort(sum_args_sorts)[1]

    return idxx

def train(trainloader, model, criterion, optimizer, device):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item() / 100.0, inputs.size()[0])
        top5.update(prec5.item() / 100.0, inputs.size()[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (losses.avg, top1.avg)

def test(testloader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_ind, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device, torch.float)
            targets = targets.to(device, torch.long)
            outputs = model(inputs)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size()[0])
            top1.update(prec1 / 100.0, inputs.size()[0])
            top5.update(prec5 / 100.0, inputs.size()[0])

    del testloader, model, criterion, device
    gc.collect()
    return losses.avg, top1.avg
