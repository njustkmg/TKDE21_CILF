import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.dataset as dataset

import numpy as np
from numpy import ndarray

import os
from logging import Logger
from typing import Dict, NoReturn, List

from Dataset import Dataset
from Modules import Net, Centers, Memory


class Model:
    """
    This class provides methods for model updating.
    You should call the last method train(dataloader,epochs,prefix) to update the model.
    And other methods are supporting methods for train(dataloader,epochs,prefix).
    """

    def __init__(self, logger: Logger, log_folder: str, dataset_name: str,
                 net: Net, centers: Centers, memory: Memory,
                 batch_size: int = 128, margin: float = 1.0, memory_ipc: int = 300,
                 rate_old_centers: float = 0.8, rate_inter_loss: float = 0.8):
        self.logger = logger
        self.log_folder = log_folder
        self.dataset_name = dataset_name

        self.batch_size = batch_size
        self.margin = margin
        self.rate_old_centers = rate_old_centers
        self.rate_inter_loss = rate_inter_loss

        self.net = net
        self.centers = centers
        self.memory = memory
        self.memory_ipc = memory_ipc

    def _show_data_info(self, batch_labels: ndarray) -> NoReturn:
        """
        Count the number of instances that each class has in current batch.
        This function logs a string like below:
        "C0:24    C1:20    C2:22    C3:18    C4:22    C5:22"
        This shows that there are 24 instances of label 0, 20 instances of label 1, etc.
        """
        labels, counts = np.unique(batch_labels, return_counts=True)
        data_info = [f"C{k}:{c}" for k, c in zip(labels, counts)]
        data_info = "    ".join(data_info)
        self.logger.debug(data_info)

    @staticmethod
    def _calc_masks(batch_labels: ndarray) -> Dict[int, List[int]]:
        """
        Assign each instance's index to its belonging class.
        For example, batch_label = [0, 1, 2, 3, 4, 0, 1, 2, 3],
        then masks = {0: [0, 5], 1: [1, 6], 2: [2, 7], 3: [3, 8], 4: [4]}.
        This shows that the 0th and 5th instances belong to class 0, 1st and 6th instances belong to class 1, etc.
        """
        masks = {k: [] for k in set(batch_labels)}
        for i, k in enumerate(batch_labels):
            masks[k].append(i)
        return masks

    def _update_centers(self, batch_features: Tensor, masks: Dict[int, List[int]]) -> NoReturn:
        """
        Update class centers. Centers is a global variable that records class centers. It shapes like:
        Centers = {0: 128d-tensor, 1: 128d-tensor, 2: 128d-tensor, ... }
        The updating formula shows as below:
        Centers[k] = RATE_OLD_CENTER * old_centers[k] + (1 - RATE_OLD_CENTER) * Batch_Centers[k]
        """
        # update centers
        with torch.no_grad():
            for k, k_index_list in masks.items():
                k_center = torch.mean(batch_features[k_index_list], dim=0)
                if k in self.centers.centers_index:
                    self.centers.centers[k] = self.rate_old_centers * self.centers.centers[k] + \
                                              (1 - self.rate_old_centers) * k_center
                else:
                    self.centers.centers_index.add(k)
                    self.centers.centers[k] = k_center

    def _calc_probability_matrix(self, batch_features: Tensor) -> Tensor:
        """
        Calculate the probability matrix.
        "probability_matrix[i][k]" is the probability of the i-th instance belonging to class k.
        """
        n_instances = batch_features.size(0)
        probability_matrix = torch.zeros((n_instances, 10), dtype=torch.float32)
        for i, feature in enumerate(batch_features):
            probability = -torch.norm(feature - self.centers.centers, dim=1)
            probability_matrix[i] = func.softmax(probability, dim=0)
        return probability_matrix

    @staticmethod
    def _calc_cross_entropy_loss(batch_labels: ndarray, probability_matrix: Tensor) -> Tensor:
        """
        Calculate cross-entropy loss using probability.
        """
        index = np.arange(start=0, stop=batch_labels.shape[0])
        cross_entropy_loss = torch.mean(-torch.log(probability_matrix[index, batch_labels]))
        return cross_entropy_loss

    @staticmethod
    def _calc_distance_matrix(batch_features: Tensor) -> Tensor:
        """
        Calculate the distance matrix.
        "distance_matrix[i][j]" is the distance between the i-th instance and the j-th instance.
        """
        n_instances = batch_features.size(0)
        distance_matrix = torch.zeros((n_instances, n_instances), dtype=torch.float32)
        for i, feature_i in enumerate(batch_features):
            for j, feature_j in enumerate(batch_features):
                distance_matrix[i][j] = torch.norm(feature_i - feature_j)
        return distance_matrix

    def _calc_triplet_loss(self, batch_labels: ndarray, distance_matrix: Tensor, masks: Dict[int, List[int]]) \
            -> Tensor:
        """
        Generate triplets using batch-hard method and calculate triplet loss.
        """
        n_instances = batch_labels.shape[0]
        n_triplets = 0
        triplet_loss = torch.tensor(0., dtype=torch.float32, requires_grad=True)

        for anchor, k in enumerate(batch_labels):
            # find the max positive
            positive = None
            for j in masks[k]:
                if positive is None or distance_matrix[anchor][j] > distance_matrix[anchor][positive]:
                    positive = j
            # find the min negative
            negative = None
            for j in range(n_instances):
                if j not in masks[k]:
                    if negative is None or distance_matrix[anchor][j] < distance_matrix[anchor][negative]:
                        negative = j
            # accumulate triplet loss if valid
            if positive and negative:
                t_loss = distance_matrix[anchor][positive] - distance_matrix[anchor][negative] + self.margin
                if t_loss > 0:
                    triplet_loss = triplet_loss + t_loss
                    n_triplets += 1

        if n_triplets:
            triplet_loss /= n_triplets

        return triplet_loss

    # def _update_memory(self, batch_raw_data: Tensor, batch_data: Tensor, batch_labels: Tensor,
    #                    update_existing: bool) -> NoReturn:
    #     """
    #     Merge instances of current batch and instances in memory, then choose the best several instances.
    #     The best several is determined by "NUM_INSTANCES_PER_CLASS_IN_MEMORY".
    #     """
    #     # update instances in Memory
    #     self.net.eval()
    #
    #     transform = transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.ToTensor(),
    #     ])
    #
    #     if update_existing:
    #         with torch.no_grad():
    #             for k, data_array in self.memory.items():
    #                 for i, instance in enumerate(data_array):
    #                     raw_image = instance[:-1]
    #                     width = int(np.sqrt(raw_image.size / 3))
    #                     raw_image = raw_image.reshape(3, width, width).transpose(1, 2, 0).astype("uint8")
    #                     image = Image.fromarray(raw_image, mode="RGB")
    #                     # image.show()
    #                     data = transform(image)
    #                     data = data.unsqueeze(0)
    #                     feature = self.net(data.cuda(self.device)).cpu()
    #                     probability = -torch.norm(feature - self.centers, dim=1)
    #                     probability = func.softmax(probability, dim=0)
    #                     self.memory[k][i, -1] = probability[k].item()
    #
    #     # add current batch instances to Memory
    #     for raw_data, data, k in zip(batch_raw_data, batch_data, batch_labels):
    #         raw_data = np.array(raw_data, dtype=np.float32).reshape(1, -1)
    #         data = data.unsqueeze(0)
    #         k = k.item()
    #         feature = self.net(data.cuda(self.device)).cpu()
    #         probability = -torch.norm(feature - self.centers, dim=1)
    #         probability = func.softmax(probability, dim=0)
    #         instance = np.c_[raw_data, probability[k].item()]
    #         if k in self.memory.keys():
    #             self.memory[k] = np.r_[self.memory[k], instance]
    #         else:
    #             self.memory[k] = instance.copy()
    #
    #     # choose the best several instances
    #     for k, data_array in self.memory.items():
    #         data_array = data_array[np.argsort(data_array[:, -1])]
    #         self.memory[k] = data_array[-self.memory_ipc:]

    def train(self, dataset: Dataset, lr: float, epochs: int) -> NoReturn:
        """
        Main train method. Using specific "dataset" to train model for specific "epochs".
        "prefix" is used for file naming. All saved files start with "prefix".
        """
        # initialize
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)

        for epoch in range(1, epochs + 1):
            for iteration, (batch_raw_data, batch_data, batch_labels) in enumerate(dataloader, start=1):
                # self.logger.debug(f"---- Epoch {epoch} / {epochs} - Iteration {iteration} ----")

                batch_labels_np = np.array(batch_labels, dtype=np.int32)
                self._show_data_info(batch_labels_np)

                self.net.train()
                optimizer.zero_grad()
                batch_features = self.net(batch_data.cuda()).cpu()
                masks = self._calc_masks(batch_labels_np)

                self._update_centers(batch_features, masks)

                probability_matrix = self._calc_probability_matrix(batch_features)
                distance_matrix = self._calc_distance_matrix(batch_features)
                intra_loss = self._calc_cross_entropy_loss(batch_labels_np, probability_matrix)
                inter_loss = self._calc_triplet_loss(batch_labels_np, distance_matrix, masks)
                loss = (1 - self.rate_inter_loss) * intra_loss + self.rate_inter_loss * inter_loss
                loss.backward()
                optimizer.step()

                self.logger.debug(f"    epoch {epoch} - iter {iteration} - "
                                  f"loss = {loss:.5f} ="
                                  f" {1 - self.rate_inter_loss:.3f} * {intra_loss:.5f}(intra) + " +
                                  f"{self.rate_inter_loss:.3f} * {inter_loss:.5f}(inter)")


def load_or_init_train(logger: Logger, model: Model, train: bool, epochs: int, lr: float):
    if not os.path.exists("parameters"):
        os.makedirs("parameters")

    dataset_name = model.dataset_name
    net_file = f"parameters/{dataset_name}_init_e60.pkl"
    memory_file = f"parameters/{dataset_name}_init_e60.memory.npz"
    centers_file = f"parameters/{dataset_name}_init_e60.centers.npz"

    if train:
        dataset = Dataset(dataset_name=dataset_name, dataset_type="initial")
        model.train(dataset=dataset, lr=lr, epochs=epochs)
        model.net.save(net_file)
        logger.info(f'Saved Net to "{net_file}".')

        logger.debug("Filling Memory ...")
        model.memory.init_fill(dataset_name=dataset_name)
        logger.debug(f"Finished Filling Memory: {model.memory.__str__()}.")
        model.memory.save(memory_file)
        logger.info(f'Saved Memory to "{memory_file}".')

        logger.debug("Calculating Centers ...")
        model.centers.init_calc(dataset_name=dataset_name, model=model.net)
        logger.debug(f"Finished Calculating Centers: {model.centers.__str__()}.")
        model.centers.save(centers_file)
        logger.info(f'Saved Centers to "{centers_file}".')
    else:
        model.net.load(net_file)
        logger.info(f'Loaded Net from "{net_file}".')
        model.memory.load(memory_file)
        logger.info(f'Loaded Memory from "{memory_file}".')
        model.centers.load(centers_file)
        logger.info(f'Loaded Centers from "{centers_file}".')
