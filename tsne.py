import argparse
import numpy as np
import matplotlib.pyplot as plt
import os 
import torch

from dataloaders import DatasetType, get_dataloader
from logging_utils import Logger, TRAIN_PREFIX, ADAPT_PREFIX
from sklearn import manifold, datasets


def get_model_score(model_name):
    return float(model_name.split("_")[-2])


def get_best_model(prefix, log_dir):
    max_score = 0
    max_model_name = None
    for f in os.listdir(log_dir):
        if f.startswith(prefix) and f.endswith('.pt'):
            model_score = get_model_score(f)
            if model_score > max_score:
                max_score = model_score
                max_model_name = f

    return max_model_name

def load_best_model(prefix, log_dir):
    max_model_name = get_best_model(prefix, log_dir)
    return torch.load(os.path.join(log_dir, max_model_name))


def plot_tsne(source, source_dataset, source_model, target, target_dataset, target_model):
    s_encoder = source_model.encoder
    t_encoder = target_model.encoder

    s_s_x = None
    t_s_x = None
    t_t_x = None
    s_s_y = None
    t_s_y = None
    t_t_y = None

    for i, ((s_x_batch, s_y_batch), (t_x_batch, t_y_batch)) in enumerate((zip(source_dataset, target_dataset))):
        batch_size_s = s_x_batch.shape[0]
        batch_size_t = t_x_batch.shape[0]
        s_s_encoding = s_encoder(s_x_batch).view(batch_size_s, -1).detach().numpy()
        t_s_encoding = s_encoder(t_x_batch).view(batch_size_t, -1).detach().numpy()
        t_t_encoding = t_encoder(t_x_batch).view(batch_size_t, -1).detach().numpy()

        if s_s_x is None:
            s_s_x  = s_s_encoding
            t_s_x  = t_s_encoding
            t_t_x  = t_t_encoding

            s_s_y  = s_y_batch
            t_s_y  = t_y_batch
            t_t_y  = t_y_batch

        else:
            s_s_x  = np.concatenate((s_s_x, s_s_encoding))
            t_s_x  = np.concatenate((t_s_x, t_s_encoding))
            t_t_x  = np.concatenate((t_t_x, t_t_encoding))

            s_s_y  = np.concatenate((s_s_y, s_y_batch))
            t_s_y  = np.concatenate((t_s_y, t_y_batch))
            t_t_y  = np.concatenate((t_t_y, t_y_batch))
        
        if i > args.include_batches:
            break


    features = np.concatenate((s_s_x, t_s_x, t_t_x))
    tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=300)
    tsne_features = tsne.fit_transform(features)

    s_s = s_s_x.shape[0]
    t_s = t_s_x.shape[0]
    t_t = t_s_x.shape[0]

    s1 = s_s
    s2 = s_s + t_s
    s3 = s_s + t_s + t_t
    
    plt.figure()
    plt.title("Source: %s | Target: %s" % (target, source))
    plt.scatter(tsne_features[:s1, 0], tsne_features[:s1, 1], color='r', label="Source X - Source Encoder", alpha=0.3)
    plt.scatter(tsne_features[s1:s2, 0], tsne_features[s1:s2, 1], color='g', label="Target X - Source Encoder", alpha=0.3)
    plt.scatter(tsne_features[s2:s3, 0], tsne_features[s2:s3, 1], color='b', label="Target X - Target Encoder", alpha=0.3)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_dir', type=str, default = "logs/USPS_7000_2000_2")
    arg_parser.add_argument('--source_dataset', type=str, default = "USPS")
    arg_parser.add_argument('--target_datasets', type=str, default = "MNIST")
    arg_parser.add_argument('--tsne_samples', type=int, default = 1000)
    arg_parser.add_argument('--batch_size', type=int, default = 256) 
    arg_parser.add_argument('--include_batches', type=int, default = 3) 
    args = arg_parser.parse_args()


    source_dataset = DatasetType[args.source_dataset]
    s_eval_loader = get_dataloader(source_dataset, False, args.batch_size, None)
    s_model = load_best_model(TRAIN_PREFIX, args.log_dir)

    for target in args.target_datasets.split(","):
        target_dataset = DatasetType[target]
        t_eval_loader = get_dataloader(target_dataset, False, args.batch_size, None)
        t_model = load_best_model(ADAPT_PREFIX + "_" + args.source_dataset + "_" + target, args.log_dir)

        plot_tsne(args.source_dataset, s_eval_loader, s_model, target, t_eval_loader, t_model)