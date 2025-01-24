import collections
import itertools

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import models
from args import args
from attack import analyze_rank_changes_efficient
from utils import *


#####################################FRL#########################################
def FRL_attack(tr_loaders, te_loader, val_loader):
    global mal_rank
    print("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type = "NonAffineNoStatsBN"

    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d" % (args.at_fractions,
                                                                                          n_attackers)
    print(sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n" + str(sss))

    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    initial_scores = {}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)] = m.scores.detach().clone().flatten().sort()[0]
    e = 0
    t1_best_acc = 0

    flag = 0
    count = 0
    target_acc_threshold = 0.30 # 设定目标准确率阈值
    smallest_difference = 0.01

    # FRL_defense: FRL,AFA,Multi-krum,Fang-ERR,Fang-LFR,Fang-Union,FLTrust,FABA,DnC
    agr = "FRL"

    file_path = os.path.join(args.run_base_dir, "ranking1.pkl")
    his_path = os.path.join(args.run_base_dir, "history.pkl")
    model_path = os.path.join(args.run_base_dir, "best_model.pth")
    mal_path = os.path.join(args.run_base_dir, "mal.pth")


    if not os.path.exists(file_path):
        print(f"File does not exist, creating...'{file_path}'...")
        with open(file_path, 'wb') as f:
            ranking1 = collections.defaultdict(list)
            pickle.dump(ranking1, f)

    if not os.path.exists(his_path):
        print(f"File does not exist, creating...'{his_path}'...")
        with open(his_path, 'wb') as f:
            history = collections.defaultdict(list)
            pickle.dump(history, f)

    if not os.path.exists(model_path):
        print(f"File does not exist, creating...'{model_path}'...")
        with open(model_path, 'wb') as f:
            best_model_info = collections.defaultdict(list)
            pickle.dump(best_model_info, f)

    if not os.path.exists(mal_path):
        print(f"文件不存在，正在创建 '{mal_path}'...")
        with open(mal_path, 'wb') as f:
            last_mal = collections.defaultdict(list)
            pickle.dump(last_mal, f)

    ranking1.clear()
    history.clear()
    best_model_info.clear()
    last_mal.clear()
    previous_ranking = {}

    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache()

        with open(file_path, 'rb') as f:
            ranking1 = pickle.load(f)

        with open(his_path, 'rb') as f:
            history = pickle.load(f)

        with open(model_path, 'rb') as f:
            best_model_info = pickle.load(f)

        with open(mal_path, 'rb') as f:
            last_mal = pickle.load(f)

        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious) >= args.round_nclients / 2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]

        user_updates = collections.defaultdict(list)

        mal_estimate = collections.defaultdict(list)

        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp1 = copy.deepcopy(FLmodel).cuda()
            optimizer1 = optim.SGD([p for p in mp1.parameters() if p.requires_grad], lr=args.lr * (args.lrdc ** e),
                                   momentum=args.momentum, weight_decay=args.wd)
            scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp1, criterion, optimizer1, args.device)

                optimizer1.step()
                scheduler1.step()
                del train_loss, train_acc

            for n, m in mp1.named_modules():
                if hasattr(m, "scores"):
                    scores_clone = m.scores.detach().clone()
                    rank = Find_rank(scores_clone)
                    user_updates[str(n)] = rank[None, :] if len(user_updates[str(n)]) == 0 else torch.cat(
                        (user_updates[str(n)], rank[None, :]), 0)
                    del rank
            del optimizer1, mp1, scheduler1
        torch.cuda.empty_cache()
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal = {}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                mp1 = copy.deepcopy(FLmodel).cuda()
                optimizer1 = optim.SGD([p for p in mp1.parameters() if p.requires_grad], lr=args.lr * (args.lrdc ** e),
                                       momentum=args.momentum, weight_decay=args.wd)
                scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.local_epochs)

                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp1, criterion, optimizer1, args.device)
                    optimizer1.step()
                    scheduler1.step()
                    del train_loss, train_acc
                for n, m in mp1.named_modules():
                    if hasattr(m, "scores"):
                        rank = Find_rank(m.scores.detach().clone())
                        mal_estimate[str(n)] = rank[None, :] if len(mal_estimate[str(n)]) == 0 else torch.cat(
                            (mal_estimate[str(n)], rank[None, :]), 0)
                        rank_arg = torch.sort(rank)[1]
                        if str(n) in sum_args_sorts_mal:
                            sum_args_sorts_mal[str(n)] += rank_arg
                        else:
                            sum_args_sorts_mal[str(n)] = rank_arg
                        del rank_arg, rank
                del optimizer1, mp1, scheduler1
                torch.cuda.empty_cache()


            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    edge = list(set(itertools.chain.from_iterable(history[str(n)])))
                    args_sorts = torch.sort(mal_estimate[str(n)])[1]
                    sum_args_sorts = torch.sum(args_sorts, 0)
                    local_mal_vote = torch.sort(sum_args_sorts)[1]

                    if flag:
                        if len(best_model_info[str(n)]) == 0:
                            best_model_info[str(n)] = local_mal_vote.tolist()
                        x = torch.tensor(ranking1[str(n)][-1]).cuda()

                        # my attack - this_mal_round
                        # mal_vote = attack(local_mal_vote.tolist(),edge,best_model_info[str(n)])

                        # my attack - last_round
                        # mal_vote = attack(ranking1[str(n)][-1], edge, best_model_info[str(n)])

                        # random attack
                        mal_vote = x.cpu().numpy()
                        np.random.shuffle(mal_vote)

                        if len(mal_vote.shape) >= 2:
                            user_updates[str(n)] = torch.cat((user_updates[str(n)], mal_vote), 0)
                            a = torch.sort(mal_vote)[1]
                            b = torch.sum(a, 0)
                            last_mal[str(n)] = torch.sort(b)[1].tolist()
                        else:
                            mal_rank = torch.from_numpy(mal_vote).cuda()
                            user_updates[str(n)] = torch.cat(([user_updates[str(n)]] + [mal_rank[None, :]] * len(round_malicious)), 0)
                            last_mal[str(n)] = mal_vote.tolist()

                        del x
                    else:
                        mal_rank = torch.sort(sum_args_sorts_mal[str(n)])[1]
                        last_mal[str(n)] = mal_rank
                        user_updates[str(n)] = torch.cat(([user_updates[str(n)]] + [mal_rank[None, :]] * len(round_malicious)), 0)
                        del mal_rank

            del sum_args_sorts_mal
            torch.cuda.empty_cache()

        ########################################Server agr#########################################
        if agr.startswith("Fang"):
            FRL_Fang_Vote(val_loader, FLmodel, user_updates, initial_scores, criterion, args.device, agr)
        elif agr.startswith("FABA"):
            FABA(FLmodel, user_updates, initial_scores, len(round_users) // 5)
        else:
            FRL_Defense_Vote(FLmodel, user_updates, initial_scores, agr, args.run_base_dir)

        # 保存当前轮次的排名
        if e >= 0:
            t1_loss, t1_acc = test(te_loader, FLmodel, criterion, args.device)

            if t1_acc > t1_best_acc:
                t1_best_acc = t1_acc

            if t1_acc >= target_acc_threshold:
                flag = 1
                count += 1
            else:
                flag = 0
                if count:
                    count += 1

            current_difference = abs(t1_acc - target_acc_threshold)

            if current_difference < smallest_difference:
                smallest_difference = current_difference

                for n, m in FLmodel.named_modules():
                    if hasattr(m, "scores"):
                        current_scores = Find_rank(m.scores.detach().clone()).tolist()
                        best_model_info[str(n)] = current_scores


            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    current_scores = Find_rank(m.scores.detach().clone()).tolist()

                    if len(ranking1[str(n)]) <= 10:
                        ranking1[str(n)].append(current_scores.tolist())
                    if len(ranking1[str(n)]) >= 11:
                        ranking1[str(n)].pop(0)

                    if str(n) in previous_ranking:
                        previous_scores = previous_ranking[str(n)]
                        total_edges = len(current_scores)
                        moved_to_used, moved_to_unused = analyze_rank_changes_efficient(current_scores, previous_scores)
                        edge_ids = []
                        for edge in moved_to_used:
                            edge_ids.append(edge[0])
                        for edge in moved_to_unused:
                            edge_ids.append(edge[0])

                        if len(history[str(n)]) <= 10:
                            history[str(n)].append(edge_ids)
                        if len(history[str(n)]) >= 11:
                            history[str(n)].pop(0)
                    previous_ranking[str(n)] = current_scores

            with open(file_path, 'wb') as f:
                pickle.dump(ranking1, f)
            ranking1.clear()
            with open(his_path, 'wb') as f:
                pickle.dump(history, f)
            history.clear()
            with open(model_path, 'wb') as f:
                pickle.dump(best_model_info, f)
            best_model_info.clear()
            with open(mal_path, 'wb') as f:
                pickle.dump(last_mal, f)
            last_mal.clear()

            sss = 'e %d | malicious users: %d | test1 acc %.4f | test1 loss %.4f | best test1_acc %.4f' % (
                e, len(round_malicious), t1_acc, t1_loss, t1_best_acc)
            print(sss)

            with (args.run_base_dir / "output.txt").open("a") as f1:
                f1.write("\n" + str(sss))

            del user_updates, mal_estimate
            torch.cuda.empty_cache()
            gc.collect()

        e += 1
        if count == 300:
            print(f"Early stopping triggered on {e} round.")
            break


#####################################FRL#########################################
def FRL(tr_loaders, te_loader):
    print("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type = "NonAffineNoStatsBN"

    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d" % (args.at_fractions,
                                                                                          n_attackers)
    print(sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n" + str(sss))

    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)

    initial_scores = {}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)] = m.scores.detach().clone().flatten().sort()[0]

    e = 0
    t_best_acc = 0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache()
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious) >= args.round_nclients / 2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]

        user_updates = collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr * (args.lrdc ** e),
                                  momentum=args.momentum, weight_decay=args.wd)

            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                if hasattr(m, "scores"):
                    rank = Find_rank(m.scores.detach().clone())
                    user_updates[str(n)] = rank[None, :] if len(user_updates[str(n)]) == 0 else torch.cat(
                        (user_updates[str(n)], rank[None, :]), 0)
                    del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal = {}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr * (args.lrdc ** e),
                                      momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()

                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank = Find_rank(m.scores.detach().clone())
                        rank_arg = torch.sort(rank)[1]
                        if str(n) in sum_args_sorts_mal:
                            sum_args_sorts_mal[str(n)] += rank_arg
                        else:
                            sum_args_sorts_mal[str(n)] = rank_arg
                        del rank, rank_arg
                del optimizer, mp, scheduler

            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    rank_mal_agr = torch.sort(sum_args_sorts_mal[str(n)], descending=True)[1]
                    for kk in round_malicious:
                        user_updates[str(n)] = rank_mal_agr[None, :] if len(user_updates[str(n)]) == 0 else torch.cat(
                            (user_updates[str(n)], rank_mal_agr[None, :]), 0)
            del sum_args_sorts_mal
        ########################################Server AGR#########################################
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e + 1) % 1 == 0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device)
            if t_acc > t_best_acc:
                t_best_acc = t_acc

            sss = 'e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (
            e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print(sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n" + str(sss))
        e += 1

