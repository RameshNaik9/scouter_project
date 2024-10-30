# from dataset.mnist import MNIST
# from dataset.CUB200 import CUB_200
# from dataset.ConText import ConText, MakeList, MakeListImage
# from dataset.transform_func import make_transform


# def select_dataset(args):
#     if args.dataset == "MNIST":
#         dataset_train = MNIST('./data/mnist', train=True, download=True, transform=make_transform(args, "train"))
#         dataset_val = MNIST('./data/mnist', train=False, transform=make_transform(args, "val"))
#         return dataset_train, dataset_val
#     if args.dataset == "CUB200":
#         dataset_train = CUB_200(args, train=True, transform=make_transform(args, "train"))
#         dataset_val = CUB_200(args, train=False, transform=make_transform(args, "val"))
#         return dataset_train, dataset_val
#     if args.dataset == "ConText":
#         train, val = MakeList(args).get_data()
#         dataset_train = ConText(train, transform=make_transform(args, "train"))
#         dataset_val = ConText(val, transform=make_transform(args, "val"))
#         return dataset_train, dataset_val
#     if args.dataset == "ImageNet":
#         train, val = MakeListImage(args).get_data()
#         dataset_train = ConText(train, transform=make_transform(args, "train"))
#         dataset_val = ConText(val, transform=make_transform(args, "val"))
#         return dataset_train, dataset_val

#     raise ValueError(f'unknown {args.dataset}')


from dataset.mnist import MNIST
from dataset.CUB200 import CUB_200
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.skin_dataset import (
    SkinCancerDataset,
    get_train_val_split,
)
from dataset.transform_func import make_transform
import os
import pandas as pd
from glob import glob  # <-- Add this line


def select_dataset(args):
    if args.dataset == "MNIST":
        dataset_train = MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=make_transform(args, "train"),
        )
        dataset_val = MNIST(
            "./data/mnist", train=False, transform=make_transform(args, "val")
        )
        return dataset_train, dataset_val

    if args.dataset == "CUB200":
        dataset_train = CUB_200(
            args, train=True, transform=make_transform(args, "train")
        )
        dataset_val = CUB_200(args, train=False, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    if args.dataset == "ConText":
        train, val = MakeList(args).get_data()
        dataset_train = ConText(train, transform=make_transform(args, "train"))
        dataset_val = ConText(val, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    if args.dataset == "SkinCancer":
        # Define paths to image folder and CSV file
        args.dataset_dir = "/content/scouter_project/data/skin-cancer-mnist-ham10000"
        csv_file = os.path.join(args.dataset_dir, "HAM10000_metadata.csv")
        img_dir = os.path.join(args.dataset_dir, "HAM10000_images_part_")

        # Read the dataset
        df_original = pd.read_csv(csv_file)

        # Map the 'dx' column to human-readable lesion types
        lesion_type_dict = {
            "nv": "Melanocytic nevi",
            "mel": "Melanoma",
            "bkl": "Benign keratosis-like lesions",
            "bcc": "Basal cell carcinoma",
            "akiec": "Actinic keratoses",
            "vasc": "Vascular lesions",
            "df": "Dermatofibroma",
        }

        # Create a new 'cell_type' column based on the 'dx' column
        df_original["cell_type"] = df_original["dx"].map(lesion_type_dict.get)

        # Create the 'cell_type_idx' column as numeric indices for each lesion type
        df_original["cell_type_idx"] = pd.Categorical(df_original["cell_type"]).codes

        # Add the path column
        # Assuming images are stored in two folders: HAM10000_images_part_1 and HAM10000_images_part_2
        imageid_path_dict = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(
                os.path.join(args.dataset_dir, "HAM10000_images_part_*", "*.jpg")
            )
        }
        df_original["path"] = df_original["image_id"].map(imageid_path_dict.get)

        # Perform the train-validation split
        df_train, df_val = get_train_val_split(df_original)

        # Define the datasets
        dataset_train = SkinCancerDataset(
            df_train, img_dir, transform=make_transform(args, "train")
        )
        dataset_val = SkinCancerDataset(
            df_val, img_dir, transform=make_transform(args, "val")
        )

        return dataset_train, dataset_val

    if args.dataset == "ImageNet":
        train, val = MakeListImage(args).get_data()
        dataset_train = ConText(train, transform=make_transform(args, "train"))
        dataset_val = ConText(val, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    raise ValueError(f"unknown {args.dataset}")
