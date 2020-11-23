import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import utils

from dataloaders import get_dataloader, DatasetType
from ignite import metrics
from logging_utils import Logger, TRAIN_PREFIX, EVAL_PREFIX
from models import LeNet, DomainCritic
from torch import autograd


def gradient_penalty(critic, s_features, t_features, device):
    line_points = s_features + (torch.rand(s_features.size(0), 1).to(device) * (t_features - s_features))
    all_features = torch.stack([line_points, s_features, t_features]).requires_grad_()
    preds = critic(all_features)
    gradients = autograd.grad(preds, all_features, grad_outputs=torch.ones_like(preds), 
                               retain_graph=True, create_graph=True)[0]
    return torch.pow((gradients.norm(2, dim=1) - 1), 2).mean()


def train_wdgrl_model(logger, source_dataset, target_dataset, args):
    device = utils.maybe_get_cuda_device()
    source_train_loader = get_dataloader(source_dataset, True, args.batch_size, args.n_train_samples)
    source_eval_loader = get_dataloader(source_dataset, False, args.batch_size, args.n_test_samples)
    target_train_loader = get_dataloader(target_dataset, True, args.batch_size, args.n_train_samples)
    target_eval_loader = get_dataloader(target_dataset, False, args.batch_size, args.n_test_samples)

    model = LeNet()
    encoder = model.encoder
    classifier = model.classifier
    critic = DomainCritic()

    model.to(device)
    encoder.to(device)
    classifier.to(device)
    critic.to(device)

    classifier_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    classifier_criterion = nn.CrossEntropyLoss()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
    

    log_prefix = (TRAIN_PREFIX + "_" + 
                  str(source_dataset.name) + "_" + 
                  str(target_dataset.name))

    logger.create_log(log_prefix)


    classifier_metrics = {
        'classifier_loss': metrics.Loss(classifier_criterion),
        'classifier_acc': metrics.Accuracy(),
    }

    source_val_metrics = {
        'source_val_loss': metrics.Loss(classifier_criterion),
        'source_val_acc': metrics.Accuracy(),
    }

    target_val_metrics = {
        'target_val_loss': metrics.Loss(classifier_criterion),
        'target_val_acc': metrics.Accuracy(),
    }

    critic_metrics = {
        'critic_loss': 0,
    }
    
    header_cols = (["epoch"] + 
                   list(classifier_metrics.keys())  + 
                   list(source_val_metrics.keys())  + 
                   list(target_val_metrics.keys())  +
                   list(critic_metrics.keys()))
    logger.write_log_line(header_cols)


    for epoch in range(args.train_epochs):
        print("%s - Epoch: %d" % (log_prefix, epoch))
        model.train()
        critic.train()
        dual_loader = zip(source_train_loader, target_train_loader)
        for i, ((source_x_batch, source_y_batch), (target_x_batch, target_y_batch)) in enumerate(dual_loader):
            s_x_batch = source_x_batch.to(device)
            s_y_batch = source_y_batch.to(device)
            t_x_batch = target_x_batch.to(device)
            t_y_batch = target_y_batch.to(device)

            # Train Critic
            classifier_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            with torch.no_grad():
                s_features = encoder(s_x_batch).data.view(s_x_batch.shape[0], -1)
                t_features = encoder(t_x_batch).data.view(t_x_batch.shape[0], -1)


            for _ in range(args.critic_steps):
                l_grad = gradient_penalty(critic, s_features, t_features, device)
                critic_s_loss = critic(s_features)
                critic_t_loss = critic(t_features)

                wasserstein_distance = critic_s_loss.mean() - critic_t_loss.mean()
                critic_loss = (args.gamma * l_grad) - wasserstein_distance

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                critic_metrics['critic_loss'] += critic_loss

            # Train Classifier
            classifier_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            s_features = encoder(s_x_batch).view(s_x_batch.shape[0], -1)
            t_features = encoder(t_x_batch).view(t_x_batch.shape[0], -1)

            s_pred = classifier(s_features)
            l_classifier = classifier_criterion(s_pred, s_y_batch)

            wasserstein_distance = None
            wasserstein_distance = critic(s_features).mean() - critic(t_features).mean()
            
            loss = l_classifier + (0.1 * wasserstein_distance)
            classifier_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            utils.update_metrics(classifier_metrics, s_pred, s_y_batch)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in source_eval_loader:
                pred = eval_step(device, model, classifier_criterion, x_batch, y_batch)
                utils.update_metrics(source_val_metrics, pred, y_batch.to(device))

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in target_eval_loader:
                pred = eval_step(device, model, classifier_criterion, x_batch, y_batch)
                utils.update_metrics(target_val_metrics, pred, y_batch.to(device))

        



        classifier_values = utils.compute_metrics(classifier_metrics)
        source_val_values = utils.compute_metrics(source_val_metrics)
        target_val_values = utils.compute_metrics(target_val_metrics)

        s_mean = source_val_metrics['source_val_acc'].compute()
        t_mean = target_val_metrics['target_val_acc'].compute()
        score = np.mean([s_mean, t_mean])

        logger.save_model(log_prefix, model, epoch, score)
        logger.write_log_line([str(epoch)] + 
                               list(classifier_values) + 
                               list(source_val_values) + 
                               list(target_val_values) +
                               [str(float(critic_metrics['critic_loss'].item()))])
        
        utils.reset_metrics(classifier_metrics)
        utils.reset_metrics(source_val_metrics)
        utils.reset_metrics(target_val_metrics)
        for k in critic_metrics.keys():
            critic_metrics[k] = 0

    logger.close_log()


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
    
    log_prefix = (EVAL_PREFIX + "_" + 
                  eval_model_prefix + "_" +
                  str(eval_dataset.name))
    logger.create_log(log_prefix)
    header_cols = list(train_metrics.keys())  + list(val_metrics.keys())
    logger.write_log_line(header_cols)

    model.eval()
    with torch.no_grad():
        print("%s" % (log_prefix))
        for x_batch, y_batch in train_loader:
            pred = eval_step(device, model, criterion, x_batch, y_batch)
            utils.update_metrics(train_metrics, pred, y_batch.to(device))

        for x_batch, y_batch in eval_loader:
            pred = eval_step(device, model, criterion, x_batch, y_batch)
            utils.update_metrics(val_metrics, pred, y_batch.to(device))

        train_values = utils.compute_metrics(train_metrics)
        val_values = utils.compute_metrics(val_metrics)

        logger.write_log_line(list(train_values) + list(val_values))
    
    logger.close_log()

    return model

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_suffix', type=str, default = "0")
    arg_parser.add_argument('--log_dir', type=str, default = "wdgrl_logs")
    arg_parser.add_argument('--source_dataset', type=str, default = "SVHN")
    arg_parser.add_argument('--target_datasets', type=str, default = "MNIST,USPS")
    arg_parser.add_argument('--n_train_samples', type=int, default = 7000)
    arg_parser.add_argument('--n_test_samples', type=int, default = 2000)
    arg_parser.add_argument('--batch_size', type=int, default = 64) 
    arg_parser.add_argument('--train_epochs', type=int, default = 300)
    arg_parser.add_argument('--lr', type=int, default = 0.0004)
    arg_parser.add_argument('--gamma', type=int, default = 10)
    arg_parser.add_argument('--critic_steps', type=int, default = 5)
    args = arg_parser.parse_args()

    logger = Logger(args.log_dir,
                "_".join([args.source_dataset, 
                              str(args.n_train_samples), 
                              str(args.n_test_samples),
                              args.log_suffix]))

    logger.create_log("params")
    for name, value in vars(args).items():
        logger.write_log_line([name, str(value)])

    source = DatasetType[args.source_dataset]

    for target in args.target_datasets.split(","):
        target = DatasetType[target]
        train_wdgrl_model(logger, source, target, args)
        evaluate_model(logger, source, TRAIN_PREFIX, args)
        evaluate_model(logger, target, TRAIN_PREFIX, args)

