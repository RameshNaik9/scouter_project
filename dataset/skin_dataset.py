# import os
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import pandas as pd
# from sklearn.model_selection import train_test_split


# class SkinCancerDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.img_dir, self.annotations.iloc[index]["path"])
#         image = Image.open(img_path).convert("RGB")
#         label = int(self.annotations.iloc[index]["cell_type_idx"])

#         if self.transform:
#             image = self.transform(image)

#         return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


# def get_train_val_split(df, test_size=0.2, random_state=101):
#     # Perform train-validation split and stratify by the cell type index
#     y = df["cell_type_idx"]
#     df_train, df_val = train_test_split(
#         df, test_size=test_size, random_state=random_state, stratify=y
#     )
#     return df_train, df_val

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


class SkinCancerDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.annotations = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index]["path"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[index]["cell_type_idx"])

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


def get_train_val_split(df, test_size=0.2, random_state=101):
    # Perform train-validation split and stratify by the cell type index
    y = df["cell_type_idx"]
    df_train, df_val = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )
    return df_train, df_val
