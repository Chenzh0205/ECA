import os
import tracemalloc

from args import args
import random
import numpy as np
import pathlib
import torch

import data
from FL_train import *


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/FRL~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"~try={i}"
            break
        i += 1

    (run_base_dir / "output.txt").write_text(str(args))
    args.run_base_dir = run_base_dir

    print(f"=> Saving data in {run_base_dir}")

    #distribute the dataset
    print("dataset to use is: ", args.set)
    print("number of FL clients: ", args.nClients)
    print("non-iid degree data distribution: ", args.non_iid_degree)
    print("batch size is : ", args.batch_size)
    print("test batch size is: ", args.test_batch_size)

    data_distributer = getattr(data, args.set)()
    tr_loaders = data_distributer.get_tr_loaders()
    te_loader = data_distributer.get_te_loader()

    subset_size = 100  # Number of samples to use for validation
    indices = np.random.choice(len(te_loader.dataset), subset_size, replace=False)
    test_subset = Subset(te_loader.dataset, indices)
    val_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print("use_cuda: ", use_cuda)

    # 检查目录是否存在
    if os.path.exists(run_base_dir) and os.path.isdir(run_base_dir):
        # 遍历目录下的所有文件
        for filename in os.listdir(run_base_dir):
            file_path = os.path.join(run_base_dir, filename)

            # 如果是文件，则删除
            if file_path.endswith(".pkl"):
                os.remove(file_path)
                print(f"已删除文件：{file_path}")

    #Federated Learning
    print("type of FL: ", args.FL_type)
    if args.FL_type == "FRL_attack":
        FRL_train(tr_loaders, te_loader, val_loader)
    elif args.FL_type == "FRL":
        FRL(tr_loaders, te_loader)


if __name__ == "__main__":
    # snapshot = start_memory_trace()
    main()
    # end_memory_trace(snapshot)
