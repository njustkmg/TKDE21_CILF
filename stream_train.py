import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.dataset as dataset

import numpy as np
from numpy import ndarray

import random
import logging
from sklearn import metrics
from sklearn.cluster import KMeans
from typing import NoReturn, Tuple

from init_train import Model
from Dataset import Dataset, PoolDataset
from Evaluation import Evaluation


class Stream:
    def __init__(self, logger: logging, log_folder: str, evaluation: Evaluation, model: Model,
                 dataset_name: str, batch_size: int, pool_size: int, epochs_per_window: int, K: int,
                 pacing_init: float, pacing_inc: float, pacing_step: float):
        self.logger = logger
        self.log_folder = log_folder
        self.evaluation = evaluation
        self.model = model

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.epochs_per_window = epochs_per_window

        self.K = K

        self.pacing_init = pacing_init
        self.pacing_inc = pacing_inc
        self.pacing_step = pacing_step

    def _show_data_info(self, batch_labels: ndarray) -> NoReturn:
        labels, counts = np.unique(batch_labels, return_counts=True)
        data_info = [f"C{k}:{c}" for k, c in zip(labels, counts)]
        data_info = "    ".join(data_info)
        self.logger.debug(data_info)

    def _calc_features(self, pool_data: Tensor) -> Tensor:
        """
        Calculate features of current pool. This function is equivalent to the code below:
        pool_features = net(pool_data)
        But the pool size is big. So we use this function to calculate features in batches.
        """
        self.model.net.eval()
        n_instances = pool_data.size(0)
        pool_features = ops.Zeros(n_instances, 128)
        with torch.no_grad():
            for i, data in enumerate(pool_data):
                data = data.unsqueeze(0).cuda()
                pool_features[i] = self.model.net(data).cpu()
        return pool_features

    def calc_scores(self, pool_features: Tensor, _pool_true_labels: Tensor) -> ndarray:
        """
        Calculate scores of current pool and sort from best to worst.
        The return value shapes like:
        [{"id": 127, "score": 0.125}, {"id": 0, "score": 0.875}]
        """
        pool_scores = np.zeros(_pool_true_labels.size(), dtype=[
            ("id", np.int32),
            ("predict_label", np.int32),
            ("true_label", np.int32),
            ("probability", np.float32),
        ])

        with torch.no_grad():
            for i, (feature, true_label) in enumerate(zip(pool_features, _pool_true_labels)):
                probability = -torch.norm(feature - self.model.centers.centers, dim=1)
                probability = func.softmax(probability, dim=0)
                for k in self.model.centers.centers_index:
                    if probability[k].item() > pool_scores[i]["probability"]:
                        pool_scores[i]["predict_label"] = k
                        pool_scores[i]["probability"] = probability[k].item()
                pool_scores[i]["id"] = i
                pool_scores[i]["true_label"] = true_label.item()

        mid_probability = 0.2 * np.min(pool_scores["probability"]) + 0.8 * np.max(pool_scores["probability"])
        pool_scores["probability"] = (pool_scores["probability"] - mid_probability) ** 2
        pool_scores = pool_scores[np.argsort(pool_scores["probability"])]

        return pool_scores

    def pacing_function(self, epoch: int) -> int:
        """
        Fixed exponential pacing function.
        "index_limit" means that we should choose data in pool_data[0:index_limit].
        """
        epoch -= 1
        rate = self.pacing_init * (self.pacing_inc ** (epoch / self.pacing_step))
        index_limit = int(min(1.0, rate) * self.pool_size)
        self.logger.debug(f"limit = {index_limit}")
        return index_limit

    def generate_batch_data(self, pool_raw_data: Tensor, pool_data: Tensor, pool_features: Tensor,
                            _pool_true_labels: Tensor, pool_scores: ndarray, index_limit: int) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Randomly choose data in pool_data[0:index_limit].
        """
        k = min(pool_data.shape[0], index_limit)
        chosen_ids = random.sample(list(range(index_limit)), k=k)
        chosen_ids = pool_scores[chosen_ids]["id"]
        pool_raw_data = pool_raw_data[chosen_ids]
        batch_data = pool_data[chosen_ids]
        batch_features = pool_features[chosen_ids]
        batch_predict_labels = pool_scores["predict_label"][chosen_ids]
        _batch_true_labels = _pool_true_labels[chosen_ids]
        return pool_raw_data, batch_data, batch_features, batch_predict_labels, _batch_true_labels

    def _cluster_centers(self, batch_features: Tensor) -> NoReturn:
        """
        Cluster "batch_features" using "centers".
        """
        centers_index = list(self.model.centers.centers_index)
        batch_features = np.array(batch_features)
        k_means = KMeans(n_clusters=len(centers_index), n_init=1, init=self.model.centers.centers[centers_index])
        k_means.fit(batch_features)
        self.model.centers.centers[centers_index] = torch.tensor(k_means.cluster_centers_, dtype=torch.float32)

    # def guess_novel_data(self, batch_data: Tensor, batch_features: Tensor, batch_labels_true: Tensor,
    #                      centers_index, centers) -> Tuple[Tensor, Tensor]:
    #     """
    #     Guess novel data in current batch and pick them up.
    #     "batch_labels_true" is the real labels of current batch and is only used for logger.
    #     """
    #     centers = torch.tensor(centers, dtype=torch.float32)
    #     novel_label = 5
    #     batch_scores = []
    #     for i, (feature, true_label) in enumerate(zip(batch_features, batch_labels_true)):
    #         probability = torch.softmax(-torch.norm(feature - centers, dim=1), dim=0)
    #         score, k = torch.max(probability, dim=0)
    #         if k == novel_label:
    #             data_item = {"id": i, "score": score.item(), "label": k.item(), "true_label": true_label.item()}
    #             batch_scores.append(data_item)
    #     batch_scores.sort(key=lambda s: s["score"], reverse=True)
    #     num_instances = int(self.novel_percent * batch_data.size(0))
    #     batch_scores = batch_scores[:num_instances]
    #
    #     novel_data = torch.tensor([])
    #     novel_labels = novel_label * torch.ones(len(batch_scores), dtype=torch.int)
    #     for i, data_item in enumerate(batch_scores):
    #         novel_data = torch.cat((novel_data, batch_data[data_item["id"]].unsqueeze(0)))
    #         self.logger.debug(f"id = {data_item['id']:3d}    predicted_label = {data_item['label']}    " +
    #                           f"true_label = {data_item['true_label']}    score = {data_item['score']:.8f}")
    #
    #     return novel_data, novel_labels

    def mix_with_memory(self, batch_raw_data: Tensor, batch_labels: Tensor, num_instances: int) \
            -> Tuple[Tensor, Tensor]:
        """
        Mix novel data with known data in "Memory".
        "num_instances" means that use "num_instances" instances per class in "Memory".
        If num == 0, no data in "Memory" is used.
        If num == -1, all data in "Memory" are used.
        If num == 10, 10 instances per class in "Memory" are used.
        """
        mixed_data = np.array(batch_raw_data, dtype=np.float32)
        mixed_labels = np.array(batch_labels, dtype=np.int32)

        for k, data_array in self.model.memory.data.items():
            k_data = data_array[-num_instances:]["data"]
            k_labels = np.full(k_data.shape[0], k, dtype=np.int32)
            mixed_data = np.r_[mixed_data, k_data]
            mixed_labels = np.r_[mixed_labels, k_labels]

        return mixed_data, mixed_labels

    def _predict(self, features: Tensor, true_labels: Tensor):
        """
        Predict and log labels of current pool.
        """
        y_pred = []
        y_true = []
        for i, (feature, true_label) in enumerate(zip(features, true_labels)):
            best_k, min_dis = 0, 1e5
            for k in self.model.centers.centers_index:
                dis = torch.norm(feature - self.model.centers.centers[k])
                if dis < min_dis:
                    best_k = k
                    min_dis = dis
            y_pred.append(best_k)
            y_true.append(true_label.item())

        self.logger.info(f"$ y_pred = {y_pred}")
        self.logger.info(f"$ y_true = {y_true}")

        acc = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
        self.logger.debug(f"    Test Accuracy = {acc * 100:5.2f}% ")
        m = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)
        self.logger.debug(f"\n{m}")

    def stream_train(self):
        self.logger.debug("----    Start Stream Processing    ----")
        pool_dataset = Dataset(dataset_name=self.dataset_name, dataset_type="stream")
        pool_dataloader = DataLoader(pool_dataset, batch_size=self.pool_size, shuffle=False, num_workers=4)

        for pool_epoch, (pool_raw_data, pool_data, _pool_true_labels) in enumerate(pool_dataloader, start=1):
            self.logger.debug(f"----    Window #{pool_epoch}    ----")
            self._show_data_info(_pool_true_labels)
            novel_label = 5

            if _pool_true_labels.max() < 7:
                continue

            for epoch in range(1, self.epochs_per_window + 1):
                pool_features = self._calc_features(pool_data)
                pool_scores = self.calc_scores(pool_features, _pool_true_labels)

                low_K_array = pool_scores[:self.K]
                novel_raw_data = pool_raw_data[low_K_array["id"]]
                novel_data = pool_data[low_K_array["id"]]
                novel_labels = torch.full((novel_data.shape[0],), novel_label, dtype=torch.int32)
                _novel_true_labels = _pool_true_labels[low_K_array["id"]]
                percent_novel = 100.0 * torch.sum(_novel_true_labels == novel_label) / _novel_true_labels.shape[0]
                self.logger.debug(f"    % of novel = {percent_novel:5.2f}%")

                limit = self.pacing_function(epoch)
                batch_raw_data, batch_data, batch_features, batch_predict_labels, _batch_true_labels = \
                    self.generate_batch_data(pool_raw_data, pool_data, pool_features, _pool_true_labels,
                                             pool_scores, limit)

                if epoch == 1:
                    self.model.centers.centers_index.add(novel_label)
                    self._cluster_centers(batch_features)
                    self.evaluation.draw_tsne(batch_features, _batch_true_labels,
                                              self.model.centers.centers_index, self.model.centers.centers, "cluster")

                mixed_raw_data, mixed_predict_labels = \
                    self.mix_with_memory(novel_raw_data, novel_labels, 100)
                mixed_dataset = PoolDataset(torch.tensor(mixed_raw_data), None, torch.tensor(mixed_predict_labels))
                self.model.train(mixed_dataset, epochs=10, lr=0.0001)

            pool_features = self._calc_features(pool_data)
            self._predict(pool_features, _pool_true_labels)

            return
