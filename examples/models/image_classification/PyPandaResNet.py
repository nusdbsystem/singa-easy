from __future__ import division
from __future__ import print_function
import os
import argparse
import sys
import base64
import abc
import tempfile
import json
import time
import shutil
import importlib
from collections import OrderedDict
from typing import Union, Dict, Optional, Any, List


# PyTorch Dependency
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet101
from torch.utils.data import Dataset, DataLoader

# Misc Third-party Machine-Learning Dependency
import numpy as np

# Misc Third-party Machine-Learning Dependency
import sklearn.metrics

# singa easy Modules Dependency
from singa_easy.models.TorchModel import TorchModel

# Singa-auto Dependency
from singa_auto.model import CategoricalKnob, FixedKnob, utils
from singa_auto.model.knob import BaseKnob
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import test_model_class


KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]




class PyPandaResNet(TorchModel):
    """
    Implementation of PyTorch ResNet
    """
    def __init__(self, **knobs):
        super().__init__(**knobs)

    def _create_model(self, scratch: bool, num_classes: int):
        model = resnet101(pretrained=not scratch, num_classes=num_classes)
        print("create model {}".format(model))
        # model = nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def get_knob_config():
        return {
        	'model_class':CategoricalKnob(['resnent101_mnist']),
            # Learning parameters
            'lr':FixedKnob(0.0001), ### learning_rate
            'weight_decay':FixedKnob(0.0),
            'drop_rate':FixedKnob(0.0),
            'max_epochs': FixedKnob(30), 
            'batch_size': CategoricalKnob([200]),
            'max_iter': FixedKnob(20),
            'optimizer':CategoricalKnob(['adam']),
            'scratch':FixedKnob(True),

            # Data augmentation
            'max_image_size': FixedKnob(32),
            'share_params': CategoricalKnob(['SHARE_PARAMS']),
            'tag':CategoricalKnob(['relabeled']),
            'workers':FixedKnob(8),
            'seed':FixedKnob(123456),
            'scale':FixedKnob(512),
            'horizontal_flip':FixedKnob(True),
     
            # Hyperparameters for PANDA modules
            # Self-paced Learning and Loss Revision
            'enable_spl':FixedKnob(False),
            'spl_threshold_init':FixedKnob(16.0),
            'spl_mu':FixedKnob(1.3),
            'enable_lossrevise':FixedKnob(False),
            'lossrevise_slop':FixedKnob(2.0),

            # Label Adaptation
            'enable_label_adaptation':FixedKnob(False), # error occurs 

            # GM Prior Regularization
            'enable_gm_prior_regularization':FixedKnob(False),
            'gm_prior_regularization_a':FixedKnob(0.001),
            'gm_prior_regularization_b':FixedKnob(0.0001),
            'gm_prior_regularization_alpha':FixedKnob(0.5),
            'gm_prior_regularization_num':FixedKnob(4),
            'gm_prior_regularization_lambda':FixedKnob(0.0001),
            'gm_prior_regularization_upt_freq':FixedKnob(100),
            'gm_prior_regularization_param_upt_freq':FixedKnob(50),
            
            # Explanation
            'enable_explanation': FixedKnob(False),
            'explanation_gradcam': FixedKnob(True),
            'explanation_lime': FixedKnob(False),

            # Model Slicing
            'enable_model_slicing':FixedKnob(False),
            'model_slicing_groups':FixedKnob(0),
            'model_slicing_rate':FixedKnob(1.0),
            'model_slicing_scheduler_type':FixedKnob('randomminmax'),
            'model_slicing_randnum':FixedKnob(1),

            # MC Dropout
            'enable_mc_dropout':FixedKnob(False),
            'mc_trials_n':FixedKnob(10)
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/fashion_mnist_train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/fashion_mnist_test.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/fashion_mnist_val.zip', help='Path to test dataset')
    print (os.getcwd())
    parser.add_argument(
        '--query_path', 
        type=str, 
        default=
        # 'examples/data/image_classification/1463729893_339.jpg,examples/data/image_classification/1463729893_326.jpg,examples/data/image_classification/eed35e9d04814071.jpg',
        'examples/data/image_classification/fashion_mnist_test_1.png',
        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()

    queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    
    test_model_class(
        model_file_path=__file__,
        model_class='PyPandaResNet',
        task='IMAGE_CLASSIFICATION',
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            ModelDependency.TORCHVISION: '0.2.2',
        },
        train_dataset_path=args.train_path,
        val_dataset_path=args.val_path,
        test_dataset_path=args.test_path,
        queries=queries
    )
