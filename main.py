import os
import sys
import argparse

from Logger import set_logger
from init_train import load_or_init_train
from Modules import Net, Centers, Memory
from init_train import Model
from stream_train import Stream
from Evaluation import Evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CILF')
    parser.add_argument('--dataset', type=str, default="CIFAR10",
                        choices=["CIFAR10", "CINIC", "SVHN", "MNIST", "FASHIONMNIST"])
    parser.add_argument('--device', type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    args.rate_inter_loss = 0.2  # λ in Eq.1
    args.memory_ipc = 300

    # args.pacing_init = 0.1  # α in Eq.9
    # args.pacing_inc = 3.0  # δ in Eq.9
    # args.pacing_step = 10  # φ in Eq.9

    args.novel_percent = 0.2
    args.epochs_per_cluster = 10
    args.epochs_fine_tune = 10
    args.epochs_per_pool = 30

    # initialize
    logger, log_folder = set_logger(name=f"init_{args.dataset}")

    if sys.platform == "win32":
        pool_size = 128
        batch_size = 32
    else:
        pool_size = 6000
        batch_size = 128
    K = 100
    #
    for dataset_name in ["MNIST", "FASHIONMNIST", "CIFAR10", "CINIC", "SVHN"]:
        net = Net().cuda()
        centers = Centers(rate_old=0.8)
        memory = Memory(K=300)
        model = Model(logger=logger, log_folder=log_folder, dataset_name=dataset_name,
                      net=net, centers=centers, memory=memory,
                      batch_size=128, margin=1.0, memory_ipc=300, rate_old_centers=0.8, rate_inter_loss=0.8)

        load_or_init_train(logger=logger, model=model, train=False, epochs=0, lr=0.0001)

        evaluation = Evaluation(logger=logger, log_folder=log_folder)
        # evaluation.evaluate_on_test(args.dataset, model, Centers.centers_index, Centers.centers,
        #                             [0, 1, 2, 3, 4], [], tsne_name=f"{args.dataset}_init_e60")

        logger.debug(f"pool_size = {pool_size}    batch_size = {batch_size}    K = {K}")
        stream = Stream(logger=logger, log_folder=log_folder, evaluation=evaluation, model=model,
                        dataset_name=dataset_name, batch_size=128, pool_size=pool_size,
                        epochs_per_window=5,
                        K=K, pacing_init=0.1, pacing_inc=3.0, pacing_step=10)
        stream.stream_train()

# CIFAR10
# K = 20     % of novel = 90.0%
# K = 50     % of novel = 86.0%  1
# K = 100    % of novel = 84.0%
# K = 200    % of novel = 84.0%
# K = 300                        1
# K = 400    % of novel = 83.0%
# K = 1000   % of novel = 76.5%  1
# K = 1600   % of novel = 68.8%
# K = 2000   % of novel = 64.9%
