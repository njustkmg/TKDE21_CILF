import random
import numpy as np
from numpy import ndarray

from typing import NoReturn, Tuple
from Dataset import Dataset, PoolDataset

from mindvision.classification.models import resnet18
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet18(pretrained=True).backbone
        self.resnet =  nn.CellList(list(self.resnet.cells()))
        self.fc = nn.Dense(in_channels=512, out_channels=128)
        self.norm = nn.BatchNorm1d(num_features=512)
        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        f512 = self.resnet(x)
        f512 = f512.view(-1, 512)
        f512 = self.norm(f512)
        f512 = self.relu(f512)
        f128 = self.fc(f512)
        return f128

    def load(self, filename: str) -> NoReturn:
        ms.load_param_into_net(self, ms.load_checkpoint(filename))

    def save(self, filename: str) -> NoReturn:
        ms.save_checkpoint(self.parameters_dict(), filename)


class Centers:
    def __init__(self, rate_old: float):
        self.rate_old = rate_old
        self.centers_index = set()
        self.centers = ops.Zeros((10, 512), dtype=ms.float32)

    def __str__(self):
        return f"centers_index = {self.centers_index}    shape = {self.centers.shape}"

    def init_calc(self, dataset_name: str, model: Net) -> NoReturn:
        dataset_name = Dataset(dataset_name=dataset_name, dataset_type="init")
        dataloader = DataLoader(dataset_name, batch_size=256, shuffle=True, num_workers=4)

        self.centers_index = set()
        self.centers = ops.Zeros((10, 512), dtype=ms.float32)

        with torch.no_grad():
            for _, batch_data, batch_labels in dataloader:
                batch_data = batch_data.cuda()
                batch_features, _ = model(batch_data)
                for k in set(batch_labels.tolist()):
                    k = int(k)
                    k_features = batch_features[batch_labels == k]
                    center = torch.mean(k_features, dim=0)
                    if k in self.centers_index:
                        self.centers[k] = self.rate_old * self.centers[k] + (1 - self.rate_old) * center
                    else:
                        self.centers_index.add(k)
                        self.centers[k] = center

    def load(self, filename: str) -> NoReturn:
        centers_files = np.load(filename)
        self.centers_index = set(centers_files["centers_index"].tolist())
        centers_array = centers_files["centers_array"]
        self.centers = torch.tensor(centers_array, dtype=torch.float32, requires_grad=False)

    def save(self, filename: str) -> NoReturn:
        centers_index = np.array(list(self.centers_index), dtype=np.int32)
        centers_array = self.centers.cpu().numpy()
        np.savez(filename, centers_index=centers_index, centers_array=centers_array)


class Memory:
    def __init__(self, K: int):
        self.data = dict()
        self.K = K

    def __str__(self):
        count_list = []
        for k, k_data in self.data.items():
            count_list.append(f"C{k}:{k_data.shape[0]}")
        count_str = " ".join(count_list)
        return count_str

    def init_fill(self, dataset_name: str) -> NoReturn:
        dataset = Dataset(dataset_name=dataset_name, dataset_type="init")
        data = np.array(dataset.data, dtype=np.float32)
        labels = np.array(dataset.labels, dtype=np.int32)
        for k in set(labels.tolist()):
            k = int(k)
            k_data = data[labels == k]
            random_index = random.sample(list(range(k_data.shape[0])), k=k_data.shape[0])
            self.data[k] = k_data[random_index][:self.K]

    def get_data(self) -> Tuple[ndarray, ndarray]:
        data = None
        labels = None
        for k, k_data in self.data.items():
            k_labels = np.full(k_data.shape[0], k, dtype=np.int32)
            data = k_data if data is None else np.r_[data, k_data]
            labels = k_labels if labels is None else np.r_[labels, k_labels]
        return data, labels

    def load(self, filename: str) -> NoReturn:
        memory_files = np.load(filename)

        self.data = dict()

        for k in memory_files.files:
            raw_data = memory_files[k]

            if raw_data.shape[1] == 3 * 28 * 28 + 1:
                data = np.zeros(raw_data.shape[0], dtype=[
                    ("data", np.float32, (3, 28, 28)),
                    ("label", np.int32)
                ])
                data[:]["data"] = raw_data[:, :-1].reshape(-1, 3, 28, 28)
                data[:]["label"] = raw_data[:, -1]
            else:
                data = np.zeros(raw_data.shape[0], dtype=[
                    ("data", np.float32, (3, 32, 32)),
                    ("label", np.int32)
                ])
                data[:]["data"] = raw_data[:, :-1].reshape(-1, 3, 32, 32)
                data[:]["label"] = raw_data[:, -1]

            self.data[int(k)] = data

    def save(self, filename: str) -> NoReturn:
        memory_files = {}
        for k, k_data in self.data.items():
            memory_files[str(k)] = k_data
        np.savez(filename, **memory_files)
