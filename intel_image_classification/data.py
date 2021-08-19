import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import re
from glob import glob

class ImageDataset(Dataset):

    def __init__(self, x, y, device = "cpu", transform=None, target_transform=None, **kwargs):
        """
        This is to generate the dataset 
        x: Feature vector np.array, pd.DataFrame, list can be accepted
        y: Target vector np.array, pd.DataFrame, list can be accepted
        transform: this will be image transformer torchvision.transform
        target_transform: this will be target variable transform techanic.
        """

        self.x = x
        self.y = y
        self.device = device
        self.transform = transform
        if target_transform is not None:
            self.y = target_transform.transform(y)
            self.target_transform = target_transform
        self.channel_first = kwargs.get("channel_first", False)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        img = torch.tensor(np.array(Image.open(self.x[index])), dtype=torch.float32, device=self.device)
        img = torch.transpose(img, 0, 2) if self.channel_first else img
        img = self.transform(img) if self.transform else img
        return {
            "x": img,
            "y": torch.tensor(self.y[index], device=self.device, dtype=torch.long)
        } 


def get_dataloader(x, y, num_workers = 0, shuffle=True, batch_size=32, **kwargs):
    """
    This function uses for generating dataloaders and dataset
    """
    dataset = ImageDataset(x, y, **kwargs)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers), dataset


def read_files(datapath, format = ""):
    """
    This function will iterate over the folder path and scrap all the paths.
    Also, extract the target name from the file path
    """
    if not os.path.exists(datapath):
        raise FileExistsError("Folder path doesn't exists")
    location = os.path.join(datapath, "**", f"*.{format}")
    file_paths = glob(location, recursive=True)
    target = [re.split(r"[//\\]", x)[-2]  for x in file_paths]
    return np.array(file_paths), np.array(target).reshape(-1, 1)