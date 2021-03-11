import os 
import tempfile
import zipfile
import numpy as np
from PIL import Image

import torchvision
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from singa_easy.datasets.TorchImageDataset import TorchImageDataset
from singa_auto.datasets.image_classification_dataset import ImageDataset4Clf

def _transform_onehotlabel_gpu(data, labels, _num_classes, train=False):
    """
    Send data to GPU
    """
    inputs = data
    labels = torch.tensor(labels)
    one_hot_labels = torch.zeros(labels.shape[0], _num_classes)
    one_hot_labels[range(one_hot_labels.shape[0]), labels.squeeze()] = 1
    one_hot_labels = one_hot_labels.type(torch.FloatTensor)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs, one_hot_labels = inputs.to(device), one_hot_labels.to(device)
    except:
        pass

    return inputs, one_hot_labels

def _transform(if_train=False, image_scale_size=128, norm_mean=None, norm_std=None):
    if if_train:
            _transform = transforms.Compose([
                # transforms.Resize((image_scale_size, image_scale_size)),
                # transforms.RandomCrop(crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
    else:
            _transform = transforms.Compose([
                # transforms.Resize((image_scale_size, image_scale_size)),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
            ])
    return _transform

def processing_and_reading_existing_datasets(dataset_name, 
                                             data_dir='../data', 
                                             min_image_size=None, 
                                             max_image_size=None, 
                                             mode=None, 
                                             if_shuffle=False, 
                                             batch_size=32, 
                                             image_scale_size=None, 
                                             dataset_type = 'train'):
    '''
    :return: x_train, y_train, x_test, y_test. Four lists of torch tensors, and each list is of the dataset size. Lists will be moved to GPU while GPU is available.
    For example, x_train can be seen as [<img tensor>, <img tensor>, ...], and y_train [<label tensor>, <label tensor>, ...]
    labels are one-hot labels.
    '''
    assert min_image_size<=max_image_size
    dataset_path = f'{data_dir}{dataset_name}_{dataset_type}.zip' 
    assert os.path.exists(dataset_path)
    
    print('Loading & splitting dataset...')
    dataset= ImageDataset4Clf(dataset_path,
                              min_image_size=min_image_size,
                              max_image_size=max_image_size,
                              mode=mode,
                              if_shuffle=if_shuffle)

    _normalize_mean, _normalize_std = dataset.get_stat()
    _num_classes = dataset.classes

    # add RandomHorizontalFlip() to transforms() if is_train,
    is_train = dataset_type=='train'
    dataset = TorchImageDataset(sa_dataset=dataset,
                                image_scale_size=image_scale_size,
                                norm_mean=_normalize_mean,
                                norm_std=_normalize_std,
                                is_train=is_train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=if_shuffle)
    inputs_list, labels_list = [], []
    for batch_idx, (raw_indices, traindata,
                        batch_classes) in enumerate(dataloader):
        inputs, labels = _transform_onehotlabel_gpu(traindata,batch_classes,_num_classes,train=True)
        inputs_list.extend(inputs) 
        labels_list.extend(labels)

    return inputs_list, labels_list

def processing_and_reading_downloaded_xray(dataset_name, data_dir='data', min_image_size=None, max_image_size=None, mode=None, if_shuffle=False, image_scale_size=None):
    '''
    This method is specific for freshly downloaded paultimothymooney/chest-xray-pneumonia kaggle dataset
    '''
    source_dataset_path=f'{data_dir}{dataset_name}.zip'
    print('Loading & splitting dataset...')
    # This block is for obtaining tensor images and classes 
    with tempfile.TemporaryDirectory() as d:
        dataset_zipfile = zipfile.ZipFile(source_dataset_path, 'r')

        image_paths = [
            x for x in dataset_zipfile.namelist() 
            if (not x.endswith('/'))  and ('._' not in x) and ('/.' not in x) and ('/_' not in x)
        ]

        for fileName in image_paths:
            dataset_zipfile.extract(fileName, path=d)
        dataset_zipfile.close()

        train_image_classes, train_tensor_images, test_image_classes, test_tensor_images= [], [], [], []
        for root, directories, files in os.walk(d):
            # This will coordinate the files with its directories
            x=[[np.nan, np.nan, np.nan]]
            for name in files[:2]: ####
                image_path= os.path.join(root, name)
                if 'NORMAL' in image_path and '._' not in image_path and 'train' in image_path and not name.startswith('.'):
                    train_image_classes.append(0)
                    image = Image.open(image_path).convert("RGB")
                    mu_i = np.mean(image, axis=(0, 1))
                    mu_i = np.expand_dims(mu_i, axis=0)
                    x = np.concatenate((x, mu_i), axis=0)
                    train_tensor_images.append(image)

                elif 'NORMAL' in image_path and '._' not in image_path and 'test' in image_path and not name.startswith('.'):
                    test_image_classes.append(0)
                    image = Image.open(image_path).convert("RGB")
                    mu_i = np.mean(image, axis=(0, 1))
                    mu_i = np.expand_dims(mu_i, axis=0)
                    x = np.concatenate((x, mu_i), axis=0)
                    test_tensor_images.append(image)

                elif 'PNEUMONIA' in image_path and '._' not in image_path and 'train' in image_path and not name.startswith('.'):
                    train_image_classes.append(1)
                    image = Image.open(image_path).convert("RGB")
                    mu_i = np.mean(image, axis=(0, 1))
                    mu_i = np.expand_dims(mu_i, axis=0)
                    x = np.concatenate((x, mu_i), axis=0)
                    train_tensor_images.append(image)

                elif 'PNEUMONIA' in image_path and '._' not in image_path and 'test' in image_path and not name.startswith('.'):
                    test_image_classes.append(1)
                    image = Image.open(image_path).convert("RGB")
                    mu_i = np.mean(image, axis=(0, 1))
                    mu_i = np.expand_dims(mu_i, axis=0)
                    x = np.concatenate((x, mu_i), axis=0)
                    test_tensor_images.append(image)

                else:
                    continue

        x = x[1:] / 255
        mu = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        train_transform = _transform(if_train=True, norm_mean=mu, norm_std=std, image_scale_size=image_scale_size)
        test_transform = _transform(if_train=False, norm_mean=mu, norm_std=std, image_scale_size=image_scale_size)

        train_tensor_images = [train_transform(images) for images in train_tensor_images]
        test_tensor_images = [test_transform(images) for images in test_tensor_images]

        train_tensor_images, train_image_classes = _transform_onehotlabel_gpu(train_tensor_images, train_image_classes, _num_classes=2, train=True)
        test_tensor_images, test_image_classes= _transform_onehotlabel_gpu(test_tensor_images, test_image_classes, _num_classes=2, train=False)

        return {'x_train':train_tensor_images, 'y_train':train_image_classes, 'x_test':test_tensor_images, 'y_test':test_image_classes}

def image_dataset_download_load_and_split(dataset_name, data_dir='data', min_image_size=None, max_image_size=None, mode=None, image_scale_size=None, dataset_types=['train','test']):
    ''' 
    :return: x_train, y_train, x_test, y_test. 

    x_train,  x_test are lists of images in torch tensor format.
    y_train, y_test are lists of labels, corresponding to the x_ sets.
    
    :param dataset_name: list of str.
    :param if_download: True/False. True means to download datasets to the nominated data_dir. False stands for loading the existing datasets.
    :param data_dir: directory that datasets 'download to' or 'load from'.
    '''
    # if the dataset is not prepared in the directory yet
    assert min_image_size<=max_image_size
    if dataset_name == 'cifar10':
        if os.path.exists(data_dir+'cifar10_train.zip') is False or os.path.exists(data_dir+'cifar10_test.zip') is False :
            print ('Dataset Downloading ... ')
            from examples.datasets.image_files.load_cifar10 import load_cifar10
            # This will generate train, test, val datasets to 'data/' directory. When validation_split set to 0, means the val dataset is empty.
            load_cifar10(limit=None, validation_split=0,
                out_train_dataset_path=data_dir+'cifar10_train.zip',
                out_val_dataset_path=data_dir+'cifar10_val.zip',
                out_test_dataset_path=data_dir+'cifar10_test.zip',
                out_meta_csv_path=data_dir+'cifar10_meta.csv')
    elif dataset_name == 'fashion_mnist':
        if os.path.exists(data_dir+'fashion_mnist_train.zip') is False or os.path.exists(data_dir+'fashion_mnist_test.zip') is False :
            print ('Dataset Downloading ... ')
            from examples.datasets.image_files.load_fashion_mnist import load_fashion_mnist
            # This will generate train, test, val datasets to 'data/' directory. When validation_split set to 0, means the val dataset is empty.
            load_fashion_mnist(limit=None, validation_split=0,
                out_train_dataset_path=data_dir+'fashion_mnist_train.zip',
                out_val_dataset_path=data_dir+'fashion_mnist_val.zip',
                out_meta_csv_path=data_dir+'fashion_mnist_meta.csv',
                out_test_dataset_path=data_dir+'fashion_mnist_test.zip')
    elif dataset_name == 'mnist':
        if os.path.exists(data_dir+'mnist_train.zip') is False or os.path.exists(data_dir+'mnist_test.zip') is False :
            print ('Dataset Downloading ... ')
            from examples.datasets.image_files.load_mnist import load_mnist
            # This will generate train, test, val datasets to 'data/' directory. When validation_split set to 0, means the val dataset is empty.
            load_mnist(limit=None, validation_split=0,
                out_train_dataset_path=data_dir+'mnist_train.zip',
                out_val_dataset_path=data_dir+'mnist_val.zip',
                out_meta_csv_path=data_dir+'mnist_meta.csv',
                out_test_dataset_path=data_dir+'mnist_test.zip')

    elif dataset_name == 'xray' or dataset_name == 'chest-xray-pneumonia':
        if os.path.exists(data_dir+'chest-xray-pneumonia.zip') is False:
            print ('Dataset Downloading ... ')
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path=data_dir, unzip=False)

    # reading downloaded dataset zipfiles
    if dataset_name == 'xray' or dataset_name == 'chest-xray-pneumonia':
        datasets_loaded = processing_and_reading_downloaded_xray('chest-xray-pneumonia', 
                                                   data_dir, 
                                                   min_image_size=min_image_size, 
                                                   max_image_size=max_image_size, 
                                                   mode=mode, 
                                                   image_scale_size=image_scale_size)
    else:
        datasets_loaded = dict()
        for dataset_type in dataset_types:
            (x, y) = processing_and_reading_existing_datasets(dataset_name, 
                                                              data_dir=data_dir,  
                                                              min_image_size=min_image_size, 
                                                              max_image_size=max_image_size, 
                                                              mode=mode, 
                                                              image_scale_size=image_scale_size,
                                                              dataset_type=dataset_type)
            datasets_loaded.update({f'x_{dataset_type}': x, f'y_{dataset_type}': y})

    return datasets_loaded

if __name__ == '__main__':
    dataset_names = ['xray','cifar10','fashion_mnist','mnist']
    min_image_size=32
    max_image_size=32
    mode='RGB'
    image_scale_size=64
    data_dir = f'{os.getcwd()}/data/'

    # by default, we assume there are only two dataset types,
    dataset_types = ['train','test']

    datasets_loaded = dict()
    for dataset_name in dataset_names:
        datasets_loaded[dataset_name]={}
        print (f'{data_dir}{dataset_name}_{dataset_types[0]}.zip', os.path.exists(f'{data_dir}{dataset_name}_{dataset_types[0]}.zip'))
        if os.path.exists(f'{data_dir}{dataset_name}_{dataset_types[0]}.zip'):
            for dataset_type in dataset_types:
        	    # if datasets exist locally.
                (x, y) = processing_and_reading_existing_datasets(dataset_name, 
                                                              data_dir=data_dir, 
                                                              min_image_size=min_image_size, 
                                                              max_image_size=max_image_size, 
                                                              mode=mode, 
                                                              image_scale_size=image_scale_size,
                                                              dataset_type=dataset_type)
                datasets_loaded[dataset_name].update({f'x_{dataset_type}': x, f'y_{dataset_type}': y})
        else:
            # if datasets need to be downloaded or just downloaded.
            print ('Train, test datsets do not exist beforehand...')
            datasets_loaded[dataset_name]= image_dataset_download_load_and_split(dataset_name, 
                                                                               data_dir=data_dir, 
                                                                               min_image_size=min_image_size, 
                                                                               max_image_size=max_image_size, 
                                                                               mode=mode, 
                                                                               image_scale_size=image_scale_size,
                                                                               dataset_types=dataset_types)
