import os 
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import datasets, transforms
from singa_auto.datasets.image_classification_dataset import ImageDataset4Clf
from examples.datasets.image_files.load_fashion_mnist import load_fashion_mnist
from examples.datasets.image_files.load_cifar10 import load_cifar10

def LOAD(dataset_name):
    image_paths, image_classes = [], []
    source_dataset_path=f'data/{dataset_name}.zip'
    print('Loading & splitting dataset...')
    test= ImageDataset4Clf(dataset_path=source_dataset_path)
    # This block is for obtaining tensor images and classes 
    tensor_images, image_classes=[], []
    for index in range(test.size):
        pil_image = test._extract_item(item_path=test._image_names[index])
        tensor_images.append(transforms.ToTensor()(pil_image))

        image_class = test._image_classes[index]
        image_classes.append(image_class)

    return tensor_images, image_classes


def image_dataset_download_load_and_split(dataset_name, if_download=False, data_dir='data'):
    ''' 
    This function will return x_train, y_train, x_test, y_test. 
    x_train,  x_test are lists of images in torch tensor format.
    y_train, y_test are lists of labels, corresponding to the x_ sets.
    
    Usage:
    (x_train, y_train, x_test, y_test) = image_dataset_download_load_and_split(
    '''
    
    # if the dataset is not prepared in the directory yet
    if if_download:
        if dataset_name == 'cifar10':
            # This will generate train, test, val datasets to 'data/' directory. When validation_split set to 0, means the val dataset is empty.
            load_cifar10(limit=None, validation_split=0)
        elif dataset_name == 'fashion_mnist':
            # This will generate train, test, val datasets to 'data/' directory. When validation_split set to 0, means the val dataset is empty.
            load_fashion_mnist(limit=None, validation_split=0)
        else:
            raise
    x_train, y_train = LOAD(dataset_name+'_train')
    x_test, y_test = LOAD(dataset_name+'_test')

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset_names = ['cifar10','fashion_mnist']
    if_download=True
    # When if_download = False, we assume the dataset already exists at data/inage_classification/ with name <dataset_name>.zip, e.g. mnist.zip

    datasets_loaded = dict()
    for dataset_name in dataset_names:
        (x_train, y_train, x_test, y_test) = image_dataset_download_load_and_split(dataset_name, if_download=if_download, data_dir='data')
        datasets_loaded[dataset_name] = {'x_train':x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test}
