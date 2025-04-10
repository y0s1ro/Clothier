from torchvision.datasets.folder import pil_loader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable



class FashionStyle14(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable] = None) -> None:
        """
        Custom Dataset class for loading FashionStyle14 dataset.
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "dataset", self.annotations.iloc[idx, 0])
        image = pil_loader(img_path)
        label = torch.tensor(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return (image, label)
    
    
