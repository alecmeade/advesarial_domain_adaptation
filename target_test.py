import argparse
import cuda_utils
import torch
import torch.nn as nn
import logging_utils
import os
import numpy as np
import re

from dataloaders import DatasetType, get_dataloader
from lenet import LeNet
from ignite import metrics



def compute_metrics(metrics):
    values = []
    for name, metric in metrics.items():
        v = metric.compute()
        if "confusion" not in name:
            print(name, v)
            values.append(str(v))
        else:
            cm_str = str(np.array(v))
            cm_str = re.sub('\s+',', ', cm_str)
            values.append(cm_str)
    return values


def update_metrics(metrics, pred, y_batch):
    for name, metric in metrics.items():
        metric.update((pred, y_batch))


def eval_step(device, model, criterion, x_batch, y_batch):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    pred = model(x_batch)
    loss = criterion(pred, y_batch)
    return pred


def get_best_model(model_dir):
    max_score = 0
    max_model = None
    for f in os.listdir(model_dir):
        if f.endswith('.pt'):
            model_score = float(f.split("_")[2])
            if model_score > max_score:
                max_score = model_score
                max_model = f

    return max_model

def target_test(args):
    device = cuda_utils.maybe_get_cuda_device()

    dataloader_params = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'drop_last': False,
    }


    train_loader = get_dataloader(DatasetType[args.target_dataset], 
                                  True, 
                                  dataloader_params, 
                                  (args.img_dim, args.img_dim),
                                  args.n_train_samples)


    dataloader_params['drop_last'] = False
    eval_loader = get_dataloader(DatasetType[args.target_dataset], 
                                  False, 
                                  dataloader_params, 
                                  (args.img_dim, args.img_dim),
                                  args.n_test_samples)

    n_classes = 10
    model = LeNet()
    model_name = get_best_model(args.model_dir)
    source_dataset = model_name.split("_")[0]
    model = torch.load(os.path.join(args.model_dir, model_name))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    

    train_metrics = {
        'train_loss': metrics.Loss(criterion),
        'train_accuracy': metrics.Accuracy(),
        'train_confusion': metrics.ConfusionMatrix(n_classes),
    }
    
    val_metrics = {
        'val_loss': metrics.Loss(criterion),
        'val_accuracy': metrics.Accuracy(),
        'val_confusion': metrics.ConfusionMatrix(n_classes),
    }


    logger = logging_utils.Logger("_".join(["target_test", 
                                            source_dataset, 
                                            args.target_dataset,
                                  str(args.n_test_samples), args.log_suffix]),
                                  args,
                                  (list(train_metrics.keys())  + list(val_metrics.keys())))


    # Evaluate
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            pred = eval_step(device, model, criterion, x_batch, y_batch)
            update_metrics(train_metrics, pred, y_batch)

        print("Train:")
        train_values = compute_metrics(train_metrics)


        for x_batch, y_batch in eval_loader:
            pred = eval_step(device, model, criterion, x_batch, y_batch)
            update_metrics(val_metrics, pred, y_batch)

        print("Validate:")
        val_values = compute_metrics(val_metrics)

        logger.log_metrics_line(list(train_values) + list(val_values))
    

    logger.close_log()

    return model


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_dir', type=str, default = 'logs/source_train_SVHN_None_None__0')
    arg_parser.add_argument('--target_dataset', type=str, default = "MNIST")
    arg_parser.add_argument('--n_train_samples', type=int, default = None)
    arg_parser.add_argument('--n_test_samples', type=int, default = None)
    arg_parser.add_argument('--img_dim', type=int, default = 32)
    arg_parser.add_argument('--batch_size', type=int, default = 32) 
    arg_parser.add_argument('--log_suffix', type=str, default = "_0")
    args = arg_parser.parse_args()

    model = target_test(args)
    