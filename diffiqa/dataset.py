
import os

from torch.utils.data import Dataset

from PIL import Image

from utils import *


class ImageDataset(Dataset):

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
        img_base = Image.open(path).convert("RGB")
        return (path, self.trans(img_base), self.trans(img_base.transpose(Image.FLIP_LEFT_RIGHT)))