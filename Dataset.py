import mindspore
import mindspore.dataset as dataset
from mindspore.dataset import vision, transforms
from mindspore import Tensor

import numpy as np

import os
from typing import Tuple, Optional

# "DATASETS_PATH" is the parent path of all datasets' folders.
# All data files are stored in numpy format using "np.load()" and "np.save()".
# The tree is shown as below:
#
# ├────CILF
# │    ├────main.py
# │    ├────Dataset.py (This File)
# │    ├────......
# │    └────......
# └────DATASETS_PATH
#      ├────CIFAR10
#      │    ├────cifar10_initial.npy
#      │    ├────cifar10_stream.npy
#      │    └────cifar10_test.npy
#      ├────CINIC
#      │    ├────cinic_initial.npy
#      │    ├────cinic_stream.npy
#      │    └────cinic_test.npy
#      ├────SVHN
#      │    ├────svhn_initial.npy
#      │    ├────svhn_stream.npy
#      │    └────svhn_test.npy
#      ├────MNIST
#      │    ├────mnist_initial.npy
#      │    ├────mnist_stream.npy
#      │    └────mnist_test.npy
#      └────FASHIONMNIST
#           ├────fashionmnist_initial.npy
#           ├────fashionmnist_stream.npy
#           └────fashionmnist_test.npy


DATASETS_PATH = "../../DataSets"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, dataset_type: str):
        # check validation
        if dataset_name in ["m", "MNIST"]:
            dataset_name = "MNIST"
        elif dataset_name in ["fm", "FASHIONMNIST"]:
            dataset_name = "FASHIONMNIST"
        elif dataset_name in ["c10", "CIFAR10"]:
            dataset_name = "CIFAR10"
        elif dataset_name in ["cinic", "CINIC"]:
            dataset_name = "CINIC"
        elif dataset_name in ["svhn", "SVHN"]:
            dataset_name = "SVHN"
        else:
            raise KeyError('"dataset" must be in ["CIFAR10", "CINIC", "SVHN", "MNIST", "FASHIONMNIST"].')
        if dataset_type not in ["initial", "stream", "test"]:
            raise KeyError('"split" must be in ["initial", "stream", "test"].')

        # set transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

        # load data
        filename = os.path.join(DATASETS_PATH, dataset_name.upper(), f"{dataset_name.lower()}_{dataset_type}.npy")
        dataset = np.load(filename)
        # self.data = torch.tensor(dataset["data"], dtype=torch.float32)
        # if self.data.shape[1] == 1:
        #     self.data = self.data.repeat(1, 3, 1, 1)
        # self.labels = torch.tensor(dataset["label"], dtype=torch.long)

        self.data = torch.tensor(dataset[:, :-1], dtype=torch.float32)
        if dataset_name in ["MNIST", "FASHIONMNIST"]:
            self.data = self.data.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
        else:
            self.data = self.data.view(-1, 3, 32, 32)
        self.labels = torch.tensor(dataset[:, -1], dtype=torch.int32)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        data = self.data[idx]
        data_224 = self.transform(data)
        label = self.labels[idx]
        return data, data_224, label


class PoolDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data: Optional[Tensor], data: Optional[Tensor], labels: Optional[Tensor]):
        self.raw_data = raw_data
        self.data = data
        self.labels = labels.long() if labels is not None else labels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        if self.raw_data is not None:
            return self.raw_data.size(0)
        elif self.data is not None:
            return self.data.size(0)
        else:
            return self.labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        raw_data = self.raw_data[idx] if self.raw_data is not None else torch.tensor(0.)
        data = self.data[idx] if self.data is not None else self.transform(raw_data)
        landmark = self.labels[idx] if self.labels is not None else torch.tensor(0.)
        return raw_data, data, landmark
