import argparse
import cuda_utils
import torch
import torch.nn as nn
import logging_utils
import os
import numpy as np
import re
import copy

from adda_discriminator import AddaDiscriminator
from dataloaders import DatasetType, get_dataloader
from lenet import LeNet
from ignite import metrics



def compute_metrics(metrics):
    values = []
    for name, metric in metrics.items():
        v = metric.compute()
        if "confusion" not in name:
            values.append(str(v))
        else:
            cm_str = str(np.array(v))
            cm_str = re.sub('\s+',', ', cm_str)
            values.append(cm_str)
    return values


def update_metrics(metrics, pred, y_batch):
    for name, metric in metrics.items():
        try:
            metric.update(pred, y_batch)

        except:
            bin_pred = (pred > 0).float()
            metric.update((bin_pred, y_batch))



def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset()


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

def domain_adapt(args):
    device = cuda_utils.maybe_get_cuda_device()

    dataloader_params = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'drop_last': True,
    }


    source_train_loader = get_dataloader(DatasetType[args.source_dataset], 
                                  True, 
                                  dataloader_params, 
                                  (args.img_dim, args.img_dim),
                                  args.n_train_samples)

    target_train_loader = get_dataloader(DatasetType[args.target_dataset], 
                                  True, 
                                  dataloader_params, 
                                  (args.img_dim, args.img_dim),
                                  args.n_train_samples)

    dataloader_params['drop_last'] = False
    target_eval_loader = get_dataloader(DatasetType[args.target_dataset], 
                                          False, 
                                          dataloader_params, 
                                          (args.img_dim, args.img_dim),
                                          args.n_test_samples)


    n_classes = 10
    model_name = get_best_model(args.model_dir)
    source_dataset = model_name.split("_")[0]
    source_model = torch.load(os.path.join(args.model_dir, model_name))
    target_model = copy.deepcopy(source_model)

    source_encoder = source_model.encoder
    discriminator = AddaDiscriminator()

    source_encoder.to(device)
    target_model.to(device)
    discriminator.to(device)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = args.descrim_learning_rate)
    t_optimizer = torch.optim.Adam(target_model.encoder.parameters(), lr = args.target_learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    eval_criterion = nn.CrossEntropyLoss()

    target_train_metrics = {
        'target_loss': metrics.Loss(criterion),
    }

    target_eval_metrics = {
        'target_eval_loss': metrics.Loss(eval_criterion),
        'target_eval_accuracy': metrics.Accuracy(eval_criterion),
        # 'target_eval_confusion': : metrics.ConfusionMatrix(n_classes),
    }
    
    discrim_metrics = {
        'discrim_loss': metrics.Loss(criterion),
        'discrim_accuracy': metrics.Accuracy(),
    }


    logger = logging_utils.Logger("_".join(["domain_adapt", 
                                            source_dataset, 
                                            args.target_dataset,
                                            str(args.n_train_samples), 
                                            args.log_suffix]),
                                            args,
                                            (list(target_train_metrics.keys()) + list(target_eval_metrics.keys())  + list(discrim_metrics.keys())))


    for epoch in range(args.epochs):
        print("Epoch: %d" % epoch)
        for i, ((source_x_batch, source_y_batch), (target_x_batch, target_y_batch)) in enumerate(zip(source_train_loader, target_train_loader)):
            if i % 2 == 0:
                s_x_batch = source_x_batch.to(device)
                s_y_batch = source_y_batch.to(device)
                t_x_batch = target_x_batch.to(device)
                t_y_batch = target_y_batch.to(device)

                s_features = source_encoder(s_x_batch)
                t_features = target_model.encoder(t_x_batch)
                s_t_features = torch.cat((s_features, t_features), 0).squeeze()

                s_t_labels = torch.cat([torch.ones(s_x_batch.shape[0]),
                                        torch.zeros(t_x_batch.shape[0])])
                s_t_labels.to(device)


                s_t_features.to(device)
                s_t_labels.to(device)

                # Update discriminator to predict target or source domain better.
                d_pred = discriminator(s_t_features).squeeze()
                
                d_optimizer.zero_grad()
                t_optimizer.zero_grad()
                d_loss = criterion(d_pred, s_t_labels)
                d_loss.backward()
                d_optimizer.step()
                update_metrics(discrim_metrics, d_pred, s_t_labels)

            else:
                # Update target encoder to make it difficult for discriminator to distiniguish
                t_x_batch = target_x_batch.to(device)
                t_y_batch = target_y_batch.to(device)
                t_flipped_descrim_labels = torch.ones(t_x_batch.shape[0])

                t_features = target_model.encoder(t_x_batch).squeeze()
                d_pred = discriminator(t_features).squeeze()

                t_optimizer.zero_grad()
                d_optimizer.zero_grad()
                t_loss = criterion(d_pred, t_flipped_descrim_labels)
                t_loss.backward()
                t_optimizer.step()
                update_metrics(target_train_metrics, d_pred, t_flipped_descrim_labels)
    

        for x_batch, y_batch in target_eval_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = target_model(x_batch)
            loss = eval_criterion(pred, y_batch)
            update_metrics(target_eval_metrics, pred, y_batch)

        target_train_values = compute_metrics(target_train_metrics)
        target_eval_values = compute_metrics(target_eval_metrics)
        discrim_values = compute_metrics(discrim_metrics)
        
        acc = target_eval_metrics['target_eval_accuracy'].compute()
        model_name = "%s_%d_%0.4f_.pt" % ((args.source_dataset + "-" + args.target_dataset), epoch, acc)
        
        torch.save(target_model, os.path.join(logger.log_dir, model_name))
        logger.log_metrics_line([str(epoch)] + list(target_train_values) + list(target_eval_values) + list(discrim_values))
        
        reset_metrics(target_train_metrics)
        reset_metrics(target_eval_metrics)
        reset_metrics(discrim_metrics)


    logger.close_log()



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_dir', type=str, default = 'logs/source_train_MNIST_7000_2000__0')
    arg_parser.add_argument('--source_dataset', type=str, default = "MNIST")
    arg_parser.add_argument('--target_dataset', type=str, default = "USPS")
    arg_parser.add_argument('--n_train_samples', type=int, default = 7000)
    arg_parser.add_argument('--n_test_samples', type=int, default = 2000)
    arg_parser.add_argument('--img_dim', type=int, default = 32)
    arg_parser.add_argument('--batch_size', type=int, default = 32) 
    arg_parser.add_argument('--epochs', type=int, default = 30)
    arg_parser.add_argument('--log_steps', type=int, default = 50)
    arg_parser.add_argument('--descrim_learning_rate', type=float, default = 0.001)
    arg_parser.add_argument('--target_learning_rate', type=float, default = 0.001)
    arg_parser.add_argument('--log_suffix', type=str, default = "_0")
    args = arg_parser.parse_args()
    domain_adapt(args)

