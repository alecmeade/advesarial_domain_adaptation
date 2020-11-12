import logging_utils
import numpy as np
import os
import torch
import torch.nn as nn
import utils

from dataloaders import get_dataloader
from ignite import metrics
from models import LeNet


def eval_step(device, model, criterion, x_batch, y_batch):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    pred = model(x_batch)
    loss = criterion(pred, y_batch)
    return pred


def evaluate_model(logger, eval_dataset, eval_model_prefix, args):
    device = utils.maybe_get_cuda_device()

    train_loader = get_dataloader(eval_dataset, True, args.batch_size, args.n_train_samples)
    eval_loader = get_dataloader(eval_dataset, False, args.batch_size, args.n_test_samples)

    model = logger.load_best_model(eval_model_prefix)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_metrics = {
        'train_loss': metrics.Loss(criterion),
        'train_acc': metrics.Accuracy(),
        'train_conf': metrics.ConfusionMatrix(10),
    }
    
    val_metrics = {
        'val_loss': metrics.Loss(criterion),
        'val_acc': metrics.Accuracy(),
        'val_conf': metrics.ConfusionMatrix(10),
    }

    logger.create_log(logging_utils.EVAL_PREFIX + "_" + 
                      eval_model_prefix + "_" +
                      str(eval_dataset.name))
    header_cols = list(train_metrics.keys())  + list(val_metrics.keys())
    logger.write_log_line(header_cols)

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            pred = eval_step(device, model, criterion, x_batch, y_batch)
            utils.update_metrics(train_metrics, pred, y_batch)

        for x_batch, y_batch in eval_loader:
            pred = eval_step(device, model, criterion, x_batch, y_batch)
            utils.update_metrics(val_metrics, pred, y_batch)

        train_values = utils.compute_metrics(train_metrics)
        val_values = utils.compute_metrics(val_metrics)

        logger.write_log_line(list(train_values) + list(val_values))
    
    logger.close_log()

    return model
