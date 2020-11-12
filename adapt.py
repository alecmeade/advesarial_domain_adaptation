import logging_utils
import numpy as np
import os
import torch
import torch.nn as nn
import utils

from dataloaders import get_dataloader
from ignite import metrics
from models import Discriminator, LeNet
from evaluate import eval_step

def adapt_model(logger, source_dataset, target_dataset, args):
    device = utils.maybe_get_cuda_device()
    source_train_loader = get_dataloader(source_dataset, True, args.batch_size, args.n_train_samples)
    target_train_loader = get_dataloader(target_dataset, True, args.batch_size, args.n_train_samples)
    target_eval_loader = get_dataloader(target_dataset, False, args.batch_size, args.n_train_samples)

    source_model = logger.load_best_model(logging_utils.TRAIN_PREFIX)
    target_model = logger.load_best_model(logging_utils.TRAIN_PREFIX)
    discriminator = Discriminator()

    source_model.to(device)
    target_model.to(device)
    discriminator.to(device)

    d_optimizer = torch.optim.Adam(discriminator.parameters())
    t_optimizer = torch.optim.Adam(target_model.encoder.parameters())
    
    criterion = nn.CrossEntropyLoss()

    target_train_metrics = {
        'target_train_loss': metrics.Loss(criterion),
    }

    target_val_metrics = {
        'target_val_loss': metrics.Loss(criterion),
        'target_val_acc': metrics.Accuracy(),
    }

    discrim_metrics = {
        'discrim_loss': metrics.Loss(criterion),
        'discrim_acc': metrics.Accuracy(),
    }

    
    log_prefix = (logging_utils.ADAPT_PREFIX + "_" + 
                  str(source_dataset.name) + "_" + 
                  str(target_dataset.name))

    logger.create_log(log_prefix)

    header_cols = (["epoch"] + 
                   list(target_train_metrics.keys())  + 
                   list(target_val_metrics.keys())  + 
                   list(discrim_metrics.keys()))
    logger.write_log_line(header_cols)


    for epoch in range(args.adapt_epochs):
        print("%s - Epoch: %d" % (log_prefix, epoch))
        target_model.train()
        dual_loader = zip(source_train_loader, target_train_loader)
        for i, ((source_x_batch, source_y_batch), (target_x_batch, target_y_batch)) in enumerate(dual_loader):

            s_x_batch = source_x_batch.to(device)
            t_x_batch = target_x_batch.to(device)

            s_features = source_model.encoder(s_x_batch).view(s_x_batch.shape[0], -1)
            s_labels = torch.ones(s_x_batch.shape[0]).long()

            t_features = target_model.encoder(t_x_batch).view(t_x_batch.shape[0], -1)
            t_labels = torch.zeros(t_x_batch.shape[0]).long()

            s_t_features = torch.cat((s_features, t_features), 0)
            s_t_labels = torch.cat((s_labels, t_labels), 0)

            s_t_features = s_t_features.to(device)
            s_t_labels = s_t_labels.to(device)
            s_labels = s_labels.to(device)

            d_optimizer.zero_grad()
            t_optimizer.zero_grad()

            d_pred = discriminator(s_t_features).squeeze()
            d_loss = criterion(d_pred, s_t_labels)
            d_loss.backward()
            d_optimizer.step()

            utils.update_metrics(discrim_metrics, d_pred, s_t_labels)

            t_features = target_model.encoder(t_x_batch).view(t_x_batch.shape[0], -1)

            d_optimizer.zero_grad()
            t_optimizer.zero_grad()

            d_pred = discriminator(t_features).squeeze()
            t_loss = criterion(d_pred, s_labels)
            t_loss.backward()
            t_optimizer.step()

            utils.update_metrics(target_train_metrics, d_pred, s_labels)
    

        target_model.eval()
        with torch.no_grad():
            for x_batch, y_batch in target_eval_loader:
                pred = eval_step(device, target_model, criterion, x_batch, y_batch)
                utils.update_metrics(target_val_metrics, pred,
                        y_batch.to(device))


        target_train_values = utils.compute_metrics(target_train_metrics)
        target_val_values = utils.compute_metrics(target_val_metrics)
        discrim_values = utils.compute_metrics(discrim_metrics)
        
        score = target_val_metrics['target_val_acc'].compute()
        logger.save_model(log_prefix, target_model, epoch, score)
        logger.write_log_line([str(epoch)] + 
                               list(target_train_values) + 
                               list(target_val_values) + 
                               list(discrim_values))
        
        utils.reset_metrics(target_train_metrics)
        utils.reset_metrics(target_val_metrics)
        utils.reset_metrics(discrim_metrics)


    logger.close_log()


