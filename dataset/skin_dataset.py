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
from dataset.transform_func import make_transform
from glob import glob  # Add this line


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


def load_skin_cancer_dataset(args):
    """
    Load the HAM10000 dataset without preprocessing.
    """
    args.dataset_dir = "/content/scouter_project/data/skin-cancer-mnist-ham10000"
    csv_file = os.path.join(args.dataset_dir, "HAM10000_metadata.csv")
    img_dir = os.path.join(args.dataset_dir, "HAM10000_images_part_*")
    df_original = pd.read_csv(csv_file)

    # Map lesion types
    lesion_type_dict = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }
    df_original["cell_type"] = df_original["dx"].map(lesion_type_dict.get)
    df_original["cell_type_idx"] = pd.Categorical(df_original["cell_type"]).codes

    # Add the path column
    img_dir_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(args.dataset_dir, "HAM10000_images_part_*", "*.jpg"))
    }
    df_original["path"] = df_original["image_id"].map(img_dir_dict.get)

    # Verify 'path' column
    if df_original["path"].isnull().any():
        raise ValueError(
            "Some images in the dataset are missing paths. Check your dataset files."
        )

    print(f"\nDataset Statistics:")
    print(f"Total Samples: {len(df_original)}")
    print("Class Distribution:")
    print(df_original["cell_type"].value_counts())
    print()

    # Train-validation split
    df_train, df_val = get_train_val_split(df_original)

    # Create datasets
    dataset_train = SkinCancerDataset(
        df_train, args.dataset_dir, transform=make_transform(args, "train")
    )
    dataset_val = SkinCancerDataset(
        df_val, args.dataset_dir, transform=make_transform(args, "val")
    )
    return dataset_train, dataset_val
