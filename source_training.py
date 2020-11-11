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


def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset()


def update_metrics(metrics, pred, y_batch):
    for name, metric in metrics.items():
        metric.update((pred, y_batch))


def train_step(device, model, optimizer, criterion, x_batch, y_batch):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()
    pred = model(x_batch)
    loss = criterion(pred, y_batch)
    loss.backward()
    optimizer.step()
    return loss


def eval_step(device, model, criterion, x_batch, y_batch):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    pred = model(x_batch)
    loss = criterion(pred, y_batch)
    return pred


def source_train(args):
    device = cuda_utils.maybe_get_cuda_device()

    dataloader_params = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'drop_last': True,
    }



    train_loader = get_dataloader(DatasetType[args.source_dataset], 
                                  True, 
                                  dataloader_params, 
                                  (args.img_dim, args.img_dim),
                                  args.n_train_samples)


    dataloader_params['drop_last'] = False
    eval_loader = get_dataloader(DatasetType[args.source_dataset], 
                                  False, 
                                  dataloader_params, 
                                  (args.img_dim, args.img_dim),
                                  args.n_test_samples)

    n_classes = 10
    model = LeNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
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


    logger = logging_utils.Logger("_".join(["source_train", args.source_dataset, str(args.n_train_samples), 
                                  str(args.n_test_samples), args.log_suffix]),
                                  args,
                                  (["epoch"] + list(train_metrics.keys())  + list(val_metrics.keys())))


    for epoch in range(args.epochs):
        print("Epoch: %d" % epoch)
        # Train
        model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            loss = train_step(device, model, optimizer, criterion, x_batch, y_batch)
            
            if batch_idx % args.log_steps == 0:
                print ("Epoch: %d | Batch: %d | Loss: %0.2f" % (epoch, batch_idx, loss))

        # Evaluate
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                pred = eval_step(device, model, criterion, x_batch, y_batch)
                update_metrics(train_metrics, pred, y_batch)

            print("Train:")
            train_values = compute_metrics(train_metrics)
            reset_metrics(train_metrics)


            for x_batch, y_batch in eval_loader:
                pred = eval_step(device, model, criterion, x_batch, y_batch)
                update_metrics(val_metrics, pred, y_batch)

            print("Validate:")
            val_values = compute_metrics(val_metrics)

            model_name = "%s_%d_%0.4f_.pt" % (args.source_dataset, epoch, val_metrics['val_accuracy'].compute())
            torch.save(model, os.path.join(logger.log_dir, model_name))
            reset_metrics(val_metrics)

            logger.log_metrics_line([str(epoch)] + list(train_values) + list(val_values))
        

    logger.close_log()

    return model


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_dataset', type=str, default = "SVHN")
    arg_parser.add_argument('--n_train_samples', type=int, default = 7000)
    arg_parser.add_argument('--n_test_samples', type=int, default = 2000)
    arg_parser.add_argument('--img_dim', type=int, default = 32)
    arg_parser.add_argument('--learning_rate', type=float, default = 0.001)
    arg_parser.add_argument('--batch_size', type=int, default = 32) 
    arg_parser.add_argument('--epochs', type=int, default = 30)
    arg_parser.add_argument('--log_steps', type=int, default = 50)
    arg_parser.add_argument('--model_checkpoint_epochs', type=int, default = 1)
    arg_parser.add_argument('--log_suffix', type=str, default = "_0")
    args = arg_parser.parse_args()

    model = source_train(args)