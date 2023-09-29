# Scouter - Slot Attention-based Classifier for Explainable Image Recognition

## Abstract
Explainable artificial intelligence has been gaining attention in the past few years. However, most existing methods are based on gradients or intermediate features, which are not directly involved in the decision-making process of the classifier. In this paper, we propose a slot attention-based classifier called SCOUTER for transparent yet accurate classification. Two major differences from other attention-based methods include: (a) SCOUTER's explanation is involved in the final confidence for each category, offering more intuitive interpretation, and (b) all the categories have their corresponding positive or negative explanation, which tells "why the image is of a certain category" or "why the image is not of a certain category." We design a new loss tailored for SCOUTER that controls the model's behavior to switch between positive and negative explanations, as well as the size of explanatory regions. Experimental results show that SCOUTER can give better visual explanations while keeping good accuracy on small and medium-sized datasets.

### Google-colab implementation

https://colab.research.google.com/drive/1KvJXBS5TuJEWyKYSE2TsBqTD29xuPg7U?usp=sharing


## Usage

### MNIST Dataset

##### Pre-training for MNIST dataset

```bash
python train.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 --num_classes 10 --use_slot false --vis false --aug false
```

##### Positive Scouter for MNIST dataset

```bash
python train.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 --num_classes 10 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 1 --to_k_layer 1 --lambda_value 1. --vis false --channel 512 --aug false
```

##### Negative Scouter for MNIST dataset

```bash
python train.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 --num_classes 10 --use_slot true --use_pre false --loss_status -1 --slots_per_class 2 --power 2 --to_k_layer 1 --lambda_value 1.5 --vis false --channel 512 --aug false --freeze_layers 3
```

##### Visualization of Positive Scouter for MNIST dataset

```bash
python test.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 --num_classes 10 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 1 --to_k_layer 1 --lambda_value 1. --vis true --channel 512 --aug false
```

##### Visualization of Negative Scouter for MNIST dataset

```bash
python test.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 --num_classes 10 --use_slot true --use_pre false --loss_status -1 --slots_per_class 2 --power 2 --to_k_layer 1 --lambda_value 1.5 --vis true --channel 512 --aug false --freeze_layers 3
```

##### Visualization using torchcam for MNIST dataset

```bash
python torchcam_vis.py --dataset MNIST --model resnet18 --batch_size 64 --num_classes 10 --grad true --use_pre true
```
