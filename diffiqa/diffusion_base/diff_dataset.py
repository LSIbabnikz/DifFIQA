

from pathlib import Path
from random import random, shuffle

from torch.utils.data import Dataset

from torchvision import transforms as T

from PIL import Image

from bsrgan.degrade import auto_degrade


# dataset classes
class DiffDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        shuffle(self.paths)

        self.transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor()
        ])

        self.lq_transform = T.Compose([
            T.RandomApply([T.Lambda(auto_degrade)], p=1),
            T.RandomApply([T.Lambda(auto_degrade)], p=.2),
            T.RandomApply([T.Lambda(auto_degrade)], p=.01),
            T.Resize((112, 112)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        img = img if random() < .5 else img.transpose(Image.FLIP_LEFT_RIGHT)

        return (self.lq_transform(img), self.transform(img))