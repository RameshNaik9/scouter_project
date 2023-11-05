from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch
import numpy as np
import gdown
from torchvision.datasets.utils import extract_archive


class BT(Dataset):
    def __init__(self, root, args, train=True, transform=None, download=False):
        super(BT, self).__init__()
        self.root = root
        self.size = args.img_size
        self.num = args.num_classes
        self.train = train
        self.transform_ = transform
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.training_file = 'train_dataset.csv'
        self.test_file = 'test_dataset.csv'
        self.resource_id = "1bIam2OVs-OUSVBZm-yEC_Sgn5aiLfrsB"

        if download:
            self.download()

        if train:
            csv_file = os.path.join(self.processed_folder, self.training_file)
            img_dir = os.path.join(self.processed_folder, 'Training')
        else:
            csv_file = os.path.join(self.processed_folder, self.test_file)
            img_dir = os.path.join(self.processed_folder, 'Testing')

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __getitem__(self, index):
        label = self.annotations.iloc[index, 1]
        image_path = os.path.join(self.img_dir, str(
            label), self.annotations.iloc[index, 0])

        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label)
        label = torch.from_numpy(np.array(label))
        if self.transform_ is not None:
            img = self.transform_(img)
        return {"image": img, "label": label}

    def __len__(self):
        return len(self.annotations)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the BT data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        output = os.path.join(self.root, 'archive.zip')
        
        gdown.download(id=self.resource_id, output=output, quiet=False)
        extract_archive(from_path=output, remove_finished=True)
