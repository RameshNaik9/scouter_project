from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import warnings
from sloter.utils.vis import apply_colormap_on_image
from sloter.slot_model import SlotModel
from train import get_args_parser
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.CUB200 import CUB_200
from dataset.choose_dataset import select_dataset

# Ignoring deprecated warnings
warnings.filterwarnings("ignore")


def test(args, model, device, img, image, label, vis_id):
    model.to(device)
    model.eval()
    image = image.to(device, dtype=torch.float32)
    output = model(torch.unsqueeze(image, dim=0))
    pred = output.argmax(
        dim=1, keepdim=True
    )  # Get the index of the max log-probability
    print("Predicted label:", pred.item())

    # Visualization
    image_raw = img.convert("L")  # Convert to grayscale
    image_raw.save("sloter/vis/image_grayscale.png")
    print("Model output for vis_id:", torch.argmax(output[vis_id]).item())
    model.train()

    # Adjust the number of slots
    num_slots = (
        args.num_classes
        if args.slots_per_class == 1
        else args.num_classes * args.slots_per_class
    )

    # Apply colormap and overlay heatmap on grayscale image
    for idx in range(num_slots):
        grayscale_image = Image.open("sloter/vis/image_grayscale.png").convert("RGB")
        slot_image_path = f"sloter/vis/slot_{idx}.png"
        if not os.path.exists(slot_image_path):
            continue  # Skip if the slot image does not exist
        slot_image = np.array(
            Image.open(slot_image_path).resize(
                grayscale_image.size, resample=Image.BILINEAR
            ),
            dtype=np.uint8,
        )
        heatmap_only, heatmap_on_image = apply_colormap_on_image(
            grayscale_image, slot_image, "jet"
        )
        heatmap_on_image.save(f"sloter/vis/slot_mask_{idx}_on_grayscale.png")

    # Calculate attention area size if required
    if args.cal_area_size:
        slot_idx = str(label) if args.loss_status > 0 else str(label + 1)
        slot_image = np.array(
            Image.open(f"sloter/vis/slot_{slot_idx}.png"), dtype=np.uint8
        )
        slot_image_size = slot_image.shape
        attention_ratio = float(slot_image.sum()) / (
            slot_image_size[0] * slot_image_size[1] * 255
        )
        print(f"Attention ratio: {attention_ratio}")


def main():
    parser = argparse.ArgumentParser(
        "model training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    args_dict = vars(args)
    args_for_evaluation = ["num_classes", "lambda_value", "power", "slots_per_class"]
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    os.makedirs("sloter/vis", exist_ok=True)

    model_name = (
        f"{args.dataset}_"
        f"{'use_slot_' if args.use_slot else 'no_slot_'}"
        f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"
        f"{'for_area_size_' + str(args.lambda_value) + '_' + str(args.slots_per_class) + '_' if args.cal_area_size else ''}"
        "checkpoint.pth"
    )
    args.use_pre = False

    device = torch.device(args.device)

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )

    # Dataset selection
    if args.dataset == "ConText":
        train, val = MakeList(args).get_data()
        dataset_val = ConText(val, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True
        )
        data = next(iter(data_loader_val))
        image = data["image"][0]
        label = data["label"][0].item()
        image_orl = Image.fromarray(
            (image.cpu().detach().numpy() * 255).astype(np.uint8).transpose((1, 2, 0)),
            mode="RGB",
        )
        image = transform(image_orl)
        transform = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    elif args.dataset == "ImageNet":
        train, val = MakeListImage(args).get_data()
        dataset_val = ConText(val, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True
        )
        data = next(iter(data_loader_val))
        image = data["image"][0]
        label = data["label"][0].item()
        image_orl = Image.fromarray(
            (image.cpu().detach().numpy() * 255).astype(np.uint8).transpose((1, 2, 0)),
            mode="RGB",
        )
        image = transform(image_orl)
        transform = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    elif args.dataset == "SkinCancer":
        _, dataset_val = select_dataset(args)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        data = next(iter(data_loader_val))
        image = data["image"][0]
        label = data["label"][0].item()
        image_orl = Image.fromarray(
            (image.cpu().detach().numpy() * 255).astype(np.uint8).transpose((1, 2, 0)),
            mode="RGB",
        )
        image = transform(image_orl)
        transform = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    elif args.dataset == "MNIST":
        dataset_val = datasets.MNIST("./data/mnist", train=False, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True
        )
        data = next(iter(data_loader_val))
        image = data[0][0]
        label = data[1][0].item()
        image_orl = Image.fromarray(
            (image.cpu().detach().numpy() * 255).astype(np.uint8)[0], mode="L"
        )
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    elif args.dataset == "CUB200":
        dataset_val = CUB_200(args, train=False, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True
        )
        data = next(iter(data_loader_val))
        image = data["image"][0]
        label = data["label"][0].item()
        image_orl = Image.fromarray(
            (image.cpu().detach().numpy() * 255).astype(np.uint8).transpose((1, 2, 0)),
            mode="RGB",
        )
        image = transform(image_orl)
        transform = transforms.Compose(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    image = transform(image)

    print("Label:", label)
    model = SlotModel(args)
    # Load model checkpoint
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=device)
    model.load_state_dict(checkpoint["model"])

    test(args, model, device, image_orl, image, label, vis_id=args.vis_id)


if __name__ == "__main__":
    main()
