
import argparse
import sys

args = None


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description="FRL")
    
    parser.add_argument(
        "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="Logs",
        help="Location to logs/checkpoints",)
    
    parser.add_argument("--set", type=str, default="CIFAR10", help="FashionMNIST EMNIST CIFAR10")
    parser.add_argument("--model", type=str, default="Conv8",
                        help="EMNISTNet Conv8 LeNet(default: Conv8().")
    parser.add_argument(
        "--nClients", type=int, default=1000, help="number of clients participating in FL (default: 1000)")
    parser.add_argument(
        "--at_fractions", type=float, default=0.2, help="fraction of malicious clients (default: 0%)")
    
    parser.add_argument(
        "--non_iid_degree",
        type=float,
        default=1.0,
        help="non-iid degree data distribution given to Dirichlet Distribution (default: 1.0)",
    )
    parser.add_argument(
        "--iid_degree",
        type=float,
        default=0.0,
        help="iid data distribution, use 1.0 for iid ",
    )
   
    parser.add_argument(
        "--conv_init",
        type=str,
        default="default",
        help="How to initialize the conv weights.",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)/femnist：10",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        help="input batch size for testing (default: 128)",
    )
    
    parser.add_argument(
        "--data_loc", type=str, default="/scratch/hamid/CIFAR10/", help="Location to store data",
    )
    
    parser.add_argument(
        "--conv_type", type=str, default="MaskConv", help="Type of conv layer (defualt: MaskConv)"
    )
    
    parser.add_argument(
        "--FL_type", type=str, default="FRL_attack", help="Type of FL (defualt: FRL_attack,FRL)"
    )
    
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=5,
        help="number of local epochs to train in each FL client (default: 5)",
    )
    parser.add_argument(
        "--FL_global_epochs",
        type=int,
        default=2000,
        help="number of FL global epochs to train the global model (default: 2000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.4,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--lrdc",
        type=float,
        default=0.999,
        help="learning rate decay (default: 0.999)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        metavar="M",
        help="Weight decay (default: 0.0001)",
    )
    
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="how sparse is each layer, when using MaskConv"
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    
    parser.add_argument(
        "--round_nclients", type=int, default=25, help="Number of selected clients in each round"
    )
    parser.add_argument(
        "--rand_mal_clients", type=int, default=25, help="Number of selected malicious clients in each round to generate the malicious update"
    )
    parser.add_argument("--name", type=str, default="FRL_no_mal", help="Experiment id.")
    
    parser.add_argument(
        "--config", type=str, default=None, help="Config file to use"
    )

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if args.config is not None:
        get_config(args)

    return args


def get_config(args):
    """Parses the config file and returns the values of the arguments."""
    load_args={}
    with open(args.config, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = value
            load_args[key] = value
    args.__dict__.update(load_args)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
