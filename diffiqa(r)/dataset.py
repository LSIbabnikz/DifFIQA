
import os
import pickle

import torch
from torchvision.transforms import Compose

import numpy as np
from PIL import Image

from utils import *


class WrapperDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 item_list,
                 trans) -> None:
        """ Helper class for training and validation dataset.

        Args:
            item_list (list): List of all items.
            trans (Compose): The transformation to use on loaded images.
        """

        self.item_list = item_list
        self.trans = trans

    def __len__(self):
        return len(self.item_list)
    
    def __getitem__(self, x):
        location, score = self.item_list[x]
        return (self.trans(Image.open(location).convert("RGB")), torch.tensor(score))


def construct_datasets(dataset_args: Arguments, trans: Compose) -> Tuple[WrapperDataset, WrapperDataset]:
    """ Constructs the train and validation datasets for training the DifFIQA(R) approach.

    Args:
        dataset_args (Arguments): Arguments of the dataset.
        trans (Compose): Transformation of images used for training.

    Returns:
        Tuple[Dataset, Dataset]: Returns the tuple of trainig and validation datasets.
    """

    # Read all quality attribute files
    with open("./quality_scores/vggface2-qs.pkl", "rb") as pkl_in:
        qs_data = pickle.load(pkl_in)
        
    # 
    qs_data = list(map(lambda x: (os.path.join(dataset_args.image_loc, x[0]), x[1]), qs_data.items()))
    qs_data = list(filter(lambda x: os.path.isfile(x[0]), qs_data))
    qs_data.sort(key=lambda x: x[1])
    min_score, max_score = min(qs_data, key=lambda x: x[1])[1], max(qs_data, key=lambda x: x[1])[1]
    qs_data = list(map(lambda x: (x[0], (x[1] - min_score) / (max_score - min_score)), qs_data))

    #
    test_indices = np.arange(0, len(qs_data), 1./dataset_args.val_split)
    test_items = [qs_data[int(k)] for k in test_indices]
    train_items = list(set(qs_data).difference(set(test_items)))

    return  WrapperDataset(train_items, trans), \
            WrapperDataset(test_items, trans)


class ImageDataset():

    def __init__(self, 
                 image_loc, 
                 trans) -> None:
        """ Helper class that loads all images from a given directory.

        Args:
            image_loc (str): The location of the directory containing the desired images.
            trans (Compose): Transformations used on loaded images.
        """
        self.image_loc = image_loc
        self.trans = trans

        self.items = []
        for (dir, subdirs, files) in os.walk(self.image_loc):
            self.items.extend([os.path.join(dir, file) for file in files if isimagefile(file)])

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, x):
        path = self.items[x]
        return (path, self.trans(Image.open(path).convert("RGB")))