#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class TorchImageDataset(Dataset):
    """
    A Pytorch-type encapsulation for SINGA-AUTO ImageFilesDataset to support training/evaluation
    """

    def __init__(self,
                 sa_dataset,
                 image_scale_size,
                 norm_mean,
                 norm_std,
                 is_train=False):
        self.sa_dataset = sa_dataset
        if is_train:
            self._transform = transforms.Compose([
                transforms.Resize((image_scale_size, image_scale_size)),
                #transforms.RandomCrop(crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
        else:
            self._transform = transforms.Compose([
                transforms.Resize((image_scale_size, image_scale_size)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])

        # initialize parameters for Self-paced Learning (SPL) module
        self.zero_sample_score()
        self._loss_threshold = -0.00001
        # No threshold means all data samples are effective
        self._effective_dataset_size = self.sa_dataset.size
        # equivalent mapping in default i.e.
        # 0 - 0
        # 1 - 1
        # ...
        # N - N
        self._indice_mapping = np.linspace(start=0,
                                           stop=self.sa_dataset.size - 1,
                                           num=self.sa_dataset.size).astype(
                                               np.int32)

    def __len__(self):
        return self._effective_dataset_size

    def __getitem__(self, idx):
        """
        return datasample by given idx

        parameters:
            idx: integer number in range [0 .. self._effective_data_size - 1]

        returns:
            NOTE: being different from the standard procedure, the function returns
            tuple that contains RAW datasample index [0 .. self.sa_dataset.size - 1] as
            the first element
        """
        # translate the index to raw index in singa-auto dataset
        idx = self._indice_mapping[idx]

        image, image_class = self.sa_dataset.get_item(idx)
        image_class = torch.tensor(image_class)
        if self._transform:
            image = self._transform(image)
        else:
            image = torch.tensor(image)

        return (idx, image, image_class)

    def update_sample_score(self, indices, scores):
        """
        update the scores for datasamples

        parameters:
            indices: RAW indices for self.sa_dataset
            scores: scores for corresponding data samples
        """
        self._scores[indices] = scores

    def zero_sample_score(self):
        self._scores = np.zeros(self.sa_dataset.size)

    def update_score_threshold(self, threshold):
        self._loss_threshold = threshold
        effective_data_mask = self._scores > self._loss_threshold
        self._indice_mapping = np.linspace(
            start=0, stop=self.sa_dataset.size - 1,
            num=self.sa_dataset.size)[effective_data_mask].astype(np.int32)
        self._effective_dataset_size = len(self._indice_mapping)
        print("dataset threshold = {}, the effective sized = {}".format(
            threshold, self._effective_dataset_size))
