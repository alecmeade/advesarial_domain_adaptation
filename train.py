import logging_utils
import numpy as np
import os
import torch
import torch.nn as nn
import utils

from dataloaders import get_dataloader
from evaluate import eval_step
from ignite import metrics
from models import LeNet


def train_step(device, model, optimizer, criterion, x_batch, y_batch):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    pred = model(x_batch)

    optimizer.zero_grad()
    loss = criterion(pred, y_batch)
    loss.backward()
    optimizer.step()
    return loss


def train_model(logger, train_dataset, args):
    device = utils.maybe_get_cuda_device()
    train_loader = get_dataloader(train_dataset, True, args.batch_size, args.n_train_samples)
    eval_loader = get_dataloader(train_dataset, False, args.batch_size, args.n_test_samples)

    model = LeNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    train_metrics = {
        'train_loss': metrics.Loss(criterion),
        'train_acc': metrics.Accuracy(),
    }
    
    val_metrics = {
        'val_loss': metrics.Loss(criterion),
        'val_acc': metrics.Accuracy(),
    }


    log_prefix = logging_utils.TRAIN_PREFIX + "_" + str(train_dataset.name)
    logger.create_log(log_prefix)
    header_cols = ["epoch"] + list(train_metrics.keys())  + list(val_metrics.keys())
    logger.write_log_line(header_cols)

    for epoch in range(args.train_epochs):
        print("%s - Epoch: %d" % (log_prefix, epoch))
        model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            loss = train_step(device, model, optimizer, criterion, x_batch, y_batch)
            
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                pred = eval_step(device, model, criterion, x_batch, y_batch)
                utils.update_metrics(train_metrics, pred, y_batch.to(device))


            for x_batch, y_batch in eval_loader:
                pred = eval_step(device, model, criterion, x_batch, y_batch)
                utils.update_metrics(val_metrics, pred, y_batch.to(device))

            val_values = utils.compute_metrics(val_metrics)
            train_values = utils.compute_metrics(train_metrics)
            
            score = val_metrics['val_acc'].compute()
            logger.save_model(log_prefix, model, epoch, score)

            utils.reset_metrics(train_metrics)
            utils.reset_metrics(val_metrics)

            logger.write_log_line([str(epoch)] + 
                                   list(train_values) + 
                                   list(val_values))
        
    logger.close_log()

    return model
