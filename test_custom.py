from PIL import Image
import argparse
import torch
from torchvision import transforms
from PIL import Image
from sloter.slot_model import SlotModel
from train import get_args_parser
from torchvision import transforms

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


def main(args):
    args_dict = vars(args)
    args_for_evaluation = ['num_classes',
                           'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    model_name = f"{args.dataset}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}"\
        + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"\
        + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}" + 'checkpoint.pth'
    args.use_pre = False

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    test_image_raw = Image.open(
        f'test-images/BT/{args.image}')

    if test_image_raw.mode == 'L':
        test_image_raw = test_image_raw.convert('RGB')

    image_orl = test_image_raw

    image = transform(image_orl)

    transform = transforms.Compose(
        [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image = transform(image)

    model = SlotModel(args)
    # # Map model to be loaded to specified single gpu.
    checkpoint = torch.load(f"{args.output_dir}/" +
                            model_name, map_location=args.device)

    model.load_state_dict(checkpoint["model"])

    result = test(model, device, image)
    return result


def run_custom_test(image='eight.png', model='resnet18', dataset='MNIST', device='cuda', batch_size=32, epochs=10, viz=False):
    test_args = argparse.Namespace(image=image, model=model, dataset=dataset, device=device, channel=512, lr=0.0001, lr_drop=70, batch_size=batch_size, weight_decay=0.0001, epochs=epochs, num_classes='4', img_size=260, pre_trained=True, use_slot=True, use_pre=True, aug=False, grad=False, grad_min_level=0.0, iterated_evaluation_num=1, cal_area_size=False, thop=False, loss_status=1,
                                   freeze_layers=2, hidden_dim=64, slots_per_class='1', power='1', to_k_layer=1, lambda_value='1.', vis=viz, vis_id=0, dataset_dir='../PAN/bird_200/CUB_200_2011/CUB_200_2011/', output_dir='saved_model/', pre_dir='pre_model/', num_workers=4, start_epoch=0, resume=False, world_size=1, local_rank=None, dist_url='env://')
    # print(test_args)
    return main(test_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'model training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--image', default='')
    args = parser.parse_args()
    result = main(args)
    print(result)
