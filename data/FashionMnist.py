from args import args
from torchvision import datasets, transforms
import torch
from data.Dirichlet_noniid import *

class FashionMNIST:
    def __init__(self):
        # FashionMNIST有10个类别
        args.output_size = 10

        Mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # FashionMNIST的归一化参数
        ])

        train_dataset = datasets.FashionMNIST(root=args.data_loc, train=True, download=True, transform=Mytransform)
        test_dataset = datasets.FashionMNIST(root=args.data_loc, train=False, download=True, transform=Mytransform)

        tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients,
                                                                                  alpha=args.non_iid_degree,
                                                                                  force=False)

        self.tr_loaders = []
        tr_count = 0
        for pos, indices in tr_per_participant_list.items():
            if len(indices) == 1 or len(indices) == 0:
                print(pos)
            tr_count += len(indices)
            batch_size = args.batch_size
            self.tr_loaders.append(get_train(train_dataset, indices, batch_size))
        # print("number of total training points:", tr_count)
        self.te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    def get_tr_loaders(self):
        return self.tr_loaders

    def get_te_loader(self):
        return self.te_loader