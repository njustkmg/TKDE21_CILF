from typing import Dict, List, NoReturn, Tuple, Optional, Union, Iterable
import mindspore
import mindspore.dataset as dataset
from sklearn import metrics
from mindspore import Tensor

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from Dataset import Dataset, PoolDataset


class Evaluation:
    def __init__(self, logger, log_folder):
        self.logger = logger
        self.log_folder = log_folder
        self.e = 1e-8

    def __show_data_info(self, batch_labels: Iterable) -> NoReturn:
        batch_labels = np.array(batch_labels, dtype=np.int32)
        labels, counts = np.unique(batch_labels, return_counts=True)
        data_info = [f"C{k}:{c}" for k, c in zip(labels, counts)]
        data_info = "    ".join(data_info)
        self.logger.debug(data_info)

    def draw_tsne(self, features: Iterable, labels: Iterable, centers_index: List[int], centers: Iterable,
                  filename: str) -> NoReturn:
        """
        Draw T-SNE figure of features and centers
        """
        # prepare data
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)
        centers = np.array(centers, dtype=np.float32)
        n_samples = len(labels_array)

        for k in centers_index:
            features_array = np.r_[features_array, np.array(centers[k]).reshape(1, -1)]
            labels_array = np.r_[labels_array, k]

        # train tsne
        tsne = TSNE()
        tsne.fit_transform(features_array)
        df = pd.concat((pd.DataFrame(tsne.embedding_), pd.DataFrame(labels_array)), axis=1)
        df.columns = ["x", "y", "label"]
        features_df = df[:n_samples]
        centers_df = df[n_samples:]

        # plot tsne
        legends = []
        colors = ["xkcd:red", "xkcd:orange", "xkcd:yellow", "xkcd:green", "xkcd:blue",
                  "xkcd:indigo", "xkcd:pink", "xkcd:purple", "xkcd:brown", "xkcd:grey"]
        plt.clf()
        for k in set(labels_array):
            legends.append(str(k))
            r = features_df.loc[features_df["label"] == k]
            plt.scatter(r["x"], r["y"], c=colors[k], marker=".", label=str(k))
        for k in centers_index:
            r = centers_df.loc[centers_df["label"] == k]
            plt.scatter(r["x"], r["y"], c="white", marker="8")
        for k in centers_index:
            r = centers_df.loc[centers_df["label"] == k]
            plt.scatter(r["x"], r["y"], c=colors[k], marker="x")
        plt.legend(legends)

        # save tsne
        tsne_path = os.path.join(self.log_folder, f"{filename}.jpg")
        plt.savefig(tsne_path)
        self.logger.debug(f'Successfully saved "{filename}.jpg".')

    def evaluate_metrics(self, y_pred: Iterable, y_true: Iterable, known_classes: list) -> NoReturn:
        """
        Evaluate and log four metrics: Average Accuracy, Average F1, M_new and F_new.
        """
        y_pred = np.array(y_pred, dtype=np.int32)
        y_true = np.array(y_true, dtype=np.int32)

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        confusion_matrix = np.zeros((10, 10), dtype=np.int32)

        for pred_label, true_label in zip(y_pred, y_true):
            confusion_matrix[true_label][pred_label] += 1
            if pred_label in known_classes:
                # if true_label in known_classes:
                if true_label == pred_label:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                # if true_label not in known_classes:
                if true_label == pred_label:
                    true_negative += 1
                else:
                    false_negative += 1

        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
        f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        m_new = false_positive / (true_negative + false_positive + self.e)
        f_new = false_negative / (true_positive + false_negative + self.e)

        print(f"Accuracy:       {accuracy:.2f}")
        print(f"Precision:      {precision:.2f}")
        print(f"Recall:         {recall:.2f}")
        print(f"F1:             {f1:.2f}")
        print(f"M_new:          {m_new:.2f}")
        print(f"F_new:          {f_new:.2f}")
        print(f"Confusion Matrix:\n{confusion_matrix}")

    def calc_centers_distance(self, centers_index: List[int], centers: Tensor):
        self.logger.info("Distances between centers")
        with torch.no_grad():
            for i in centers_index:
                for j in centers_index:
                    if i < j:
                        distance = torch.norm(centers[i] - centers[j]).item()
                        self.logger.info(f"    d(C{i},C{j}) = {distance:5f}")

    def evaluate_on_test(self, dataset_name: str, net, centers_index: List[int], centers: torch,
                         tsne_name: Optional[str] = None) -> NoReturn:
        dataset = Dataset(dataset_name=dataset_name, dataset_type="test")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
        features_list = []
        y_pred = []
        y_true = []

        net.eval()
        with torch.no_grad():
            # evaluate results
            for i, (_, batch_data, batch_labels) in enumerate(data_loader):
                batch_data = batch_data[batch_labels <= 5]
                batch_labels = batch_labels[batch_labels <= 5]
                if batch_data.shape[0] == 0:
                    continue
                self.__show_data_info(batch_labels)
                batch_features = net(batch_data.cuda()).cpu()
                batch_labels = batch_labels.tolist()
                features_list.extend(batch_features.tolist())
                y_true.extend(batch_labels)
                for feature, label in zip(batch_features, batch_labels):
                    best_k, min_dis = 0, 1e5
                    for k in centers_index:
                        dis = torch.norm(feature - centers[k])
                        if dis < min_dis:
                            best_k = k
                            min_dis = dis
                    y_pred.append(best_k)

        self.evaluate_metrics(y_pred=y_pred, y_true=y_true, known_classes=[0, 1, 2, 3, 4])
        self.logger.info(f"$ y_pred = {y_pred}")
        self.logger.info(f"$ y_true = {y_true}")
        self.calc_centers_distance(centers_index, centers)

        # draw tsne
        if tsne_name:
            self.draw_tsne(features_list, y_true, centers_index, centers, tsne_name)
