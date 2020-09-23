import os 
import kaggle
import tempfile
import zipfile
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import datasets, transforms
from singa_auto.datasets.image_classification_dataset import ImageDataset4Clf
from examples.datasets.image_files.load_fashion_mnist import load_fashion_mnist
from examples.datasets.image_files.load_cifar10 import load_cifar10

def LOAD(dataset_name, data_dir='data'):
    image_paths, image_classes = [], []
    source_dataset_path=f'/{data_dir}/{dataset_name}.zip'
    print('Loading & splitting dataset...')
    test= ImageDataset4Clf(dataset_path=os.getcwd()+source_dataset_path)
    # This block is for obtaining tensor images and classes 
    tensor_images, image_classes=[], []
    for index in range(test.size):
        pil_image = test._extract_item(item_path=test._image_names[index])
        tensor_images.append(transforms.ToTensor()(pil_image))
        image_class = test._image_classes[index]
        image_classes.append(image_class)
    return tensor_images, image_classes


def LOAD2(dataset_name, data_dir='data'):
    source_dataset_path=f'/{data_dir}/{dataset_name}.zip'
    print('Loading & splitting dataset...')
    # This block is for obtaining tensor images and classes 
    with tempfile.TemporaryDirectory() as d:
        dataset_zipfile = zipfile.ZipFile(os.getcwd()+source_dataset_path, 'r')
        dataset_zipfile.extractall(path=d)
        dataset_zipfile.close()

        train_image_classes, train_tensor_images, test_image_classes, test_tensor_images= [], [], [], []
        for root, directories, files in os.walk(d):
            # This will coordinate the files with its directories
            for name in files:
                image_path= os.path.join(root, name)
                if 'NORMAL' in image_path and '._' not in image_path and 'train' in image_path and not name.startswith('.'):
                    train_image_classes.append(0)
                    train_pil_image = Image.open(image_path).convert("RGB")
                    train_tensor_images.append(transforms.ToTensor()(np.array(train_pil_image)))
                elif 'NORMAL' in image_path and '._' not in image_path and 'test' in image_path and not name.startswith('.'):
                    test_image_classes.append(0)
                    test_pil_image = Image.open(image_path).convert("RGB")
                    test_tensor_images.append(transforms.ToTensor()(np.array(test_pil_image)))
                elif 'PNEUMONIA' in image_path and '._' not in image_path and 'train' in image_path and not name.startswith('.'):
                    train_image_classes.append(1)
                    train_pil_image = Image.open(image_path).convert("RGB")
                    train_tensor_images.append(transforms.ToTensor()(np.array(train_pil_image)))

                elif 'PNEUMONIA' in image_path and '._' not in image_path and 'test' in image_path and not name.startswith('.'):
                    test_image_classes.append(1)
                    test_pil_image = Image.open(image_path).convert("RGB")
                    test_tensor_images.append(transforms.ToTensor()(np.array(test_pil_image)))
                else:
                    continue
        return train_tensor_images, train_image_classes, test_tensor_images, test_image_classes


def image_dataset_download_load_and_split(dataset_name, if_download=False, data_dir='data'):
    ''' 
    :return: x_train, y_train, x_test, y_test. 

    x_train,  x_test are lists of images in torch tensor format.
    y_train, y_test are lists of labels, corresponding to the x_ sets.
    
    :param dataset_name: list of str.
    :param if_download: True/False. True means to download datasets to the nominated data_dir. False stands for loading the existing datasets.
    :param data_dir: directory that datasets 'download to' or 'load from'.
    '''
    # if the dataset is not prepared in the directory yet
    if if_download:
        print ('Dataset Downloading ... ')
        if dataset_name == 'cifar10':
            # This will generate train, test, val datasets to 'data/' directory. When validation_split set to 0, means the val dataset is empty.
            load_cifar10(limit=None, validation_split=0,
                out_train_dataset_path=data_dir+'/cifar10_train.zip',
                out_val_dataset_path=data_dir+'/cifar10_val.zip',
                out_test_dataset_path=data_dir+'/cifar10_test.zip',
                out_meta_csv_path=data_dir+'/cifar10_meta.csv')
        elif dataset_name == 'fashion_mnist':
            # This will generate train, test, val datasets to 'data/' directory. When validation_split set to 0, means the val dataset is empty.
            load_fashion_mnist(limit=None, validation_split=0,
                out_train_dataset_path=data_dir+'/fashion_mnist_train.zip',
                out_val_dataset_path=data_dir+'/fashion_mnist_val.zip',
                out_meta_csv_path=data_dir+'/fashion_mnist_meta.csv',
                out_test_dataset_path=data_dir+'/fashion_mnist_test.zip')

        elif dataset_name == 'xray':
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path=data_dir, unzip=False)

        else:
            raise
    if dataset_name == 'xray':
        dataset_name='chest-xray-pneumonia'
        (x_train, y_train, x_test, y_test) = LOAD2(dataset_name, data_dir)
    else:
        x_train, y_train = LOAD(dataset_name+'_train', data_dir)
        x_test, y_test = LOAD(dataset_name+'_test', data_dir)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset_names = ['xray','cifar10','fashion_mnist']
    if_download=False
     # When if_download = False, we assume the dataset already exists at data/inage_classification/ with name <dataset_name>.zip, e.g. mnist.zip
    datasets_loaded = dict()
    for dataset_name in dataset_names:
        (x_train, y_train, x_test, y_test) = image_dataset_download_load_and_split(dataset_name, if_download=if_download, data_dir='data')

        datasets_loaded[dataset_name] = {'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test}

