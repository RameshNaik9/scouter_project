from __future__ import print_function
import argparse
import torch
from PIL import Image
import numpy as np
import os, os.path
from sloter.slot_model import SlotModel
from train import get_args_parser
# from tqdm import tqdm

from torchvision import transforms
from dataset.BT import BT

# ignoring deprecated warnings
import warnings
warnings.filterwarnings("ignore")

def test(model, device, image):
    model.to(device)
    model.eval()
    image = image.to(device, dtype=torch.float32)
    output = model(torch.unsqueeze(image, dim=0))
    # # get the index of the max log-probability
    pred = output.argmax(dim=1, keepdim=True)
    r = pred[0][0].int().item()
    return r

def main():
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args_dict = vars(args)
    args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    model_name = f"{args.dataset}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}"\
                + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"\
                + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}" + 'checkpoint.pth'
    args.use_pre = False

    device = torch.device(args.device)
    model = SlotModel(args)

    # Map model to be loaded to specified single gpu.
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            ])

    root = os.path.join(os.curdir, 'data', 'BT')
    dataset_test = BT(root, args=args, train=False, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset_test, 1, shuffle=False, num_workers=1, pin_memory=True)
    dataiter = iter(data_loader)

    total = 0
    correct = 0

    print("Starting test :")
    for sample in dataiter:
        image = sample["image"][0]
        label = sample["label"][0]

        # 
        if(label.int().item() == 1):
            continue

        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            ])
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        image = transform(image)

        result = test(model, device, image)

        total += 1
        if result == label:
            correct += 1

    print("Total = ", total)
    print("Correct = ", correct)

    accuracy = correct / total * 100
    print("Accuracy = ", accuracy)


if __name__ == '__main__':
    main()
