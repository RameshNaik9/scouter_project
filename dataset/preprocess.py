import pandas as pd
import os
from dataset.skin_dataset import SkinCancerDataset, get_train_val_split
from dataset.transform_func import make_transform
from glob import glob  # Add this line


def print_dataset_statistics(df, title="Dataset Statistics"):
    print(f"\n{title}")
    print(f"Total Samples: {len(df)}")
    print("Class Distribution:")
    print(df["cell_type"].value_counts())
    print()


def preprocess_skin_cancer(args):
    """
    Preprocess the HAM10000 dataset to handle class imbalance and return train/val loaders.
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

    print_dataset_statistics(df_original, title="Before Preprocessing")

    # Train-validation split
    df_train, df_val = get_train_val_split(df_original)

    # Handle class imbalance
    print("Balancing dataset with augmentation...")
    max_class_size = df_train["cell_type_idx"].value_counts().max()
    augmented_data = [df_train]

    for class_idx in df_train["cell_type_idx"].unique():
        class_data = df_train[df_train["cell_type_idx"] == class_idx]
        required_size = max_class_size - len(class_data)
        if required_size > 0:
            augmented_data.append(
                class_data.sample(n=required_size, replace=True, random_state=101)
            )

    df_train = pd.concat(augmented_data, ignore_index=True)

    print_dataset_statistics(df_train, title="After Preprocessing")

    # Create datasets
    dataset_train = SkinCancerDataset(
        df_train, args.dataset_dir, transform=make_transform(args, "train")
    )
    dataset_val = SkinCancerDataset(
        df_val, args.dataset_dir, transform=make_transform(args, "val")
    )
    return dataset_train, dataset_val
