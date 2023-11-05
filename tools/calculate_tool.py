import torch
import csv
import os
import datetime


def evaluateTop1(logits, labels):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return torch.eq(pred, labels).sum().float().item()/labels.size(0)


def evaluateTop5(logits, labels):
    with torch.no_grad():
        maxk = max((1, 5))
        labels_resize = labels.view(-1, 1)
        _, pred = logits.topk(maxk, 1, True, True)
        return torch.eq(pred, labels_resize).sum().float().item()/labels.size(0)


class MetricLog():
    def __init__(self, dataset=''):
        self.record = {"train": {"loss": [], "acc": [], "log_loss": [], "att_loss": []},
                       "val": {"loss": [], "acc": [], "log_loss": [], "att_loss": []}, 'epoch': 0}
        self.dataset = dataset

        self.log_dir = os.path.join(os.curdir, "logs", dataset)
        os.makedirs(self.log_dir, exist_ok=True)

    def print_metric(self):
        print("train loss:", self.record["train"]["loss"])
        print("val loss:", self.record["val"]["loss"])
        print("train acc:", self.record["train"]["acc"])
        print("val acc:", self.record["val"]["acc"])
        print("train CE loss", self.record["train"]["log_loss"])
        print("val CE loss", self.record["val"]["log_loss"])
        print("train attention loss", self.record["train"]["att_loss"])
        print("val attention loss", self.record["val"]["att_loss"])

        self.log_metric()

    def log_metric(self):
        train_log_file = os.path.join(self.log_dir, "train.csv")
        val_log_file = os.path.join(self.log_dir, "val.csv")

        train_log_data = [
            'epoch', self.record['epoch'],
            'loss', self.record["train"]["loss"][-1],
            'acc', self.record["train"]["acc"][-1],
            'ce_loss', self.record["train"]["log_loss"][-1],
            'attention_loss', self.record["train"]["att_loss"][-1],
            'time', datetime.datetime.utcnow()
        ]

        val_log_data = [
            'epoch', self.record['epoch'],
            'loss', self.record["val"]["loss"][-1],
            'acc', self.record["val"]["acc"][-1],
            'ce_loss', self.record["val"]["log_loss"][-1],
            'attention_loss', self.record["val"]["att_loss"][-1],
            'time', datetime.datetime.utcnow()
        ]

        self.save_log(train_log_file, train_log_data)
        self.save_log(val_log_file, val_log_data)

    def save_log(self, log_file, data):
        file = open(log_file, "a", newline='')
        writer = csv.writer(file)
        writer.writerow(data)
        file.close()
