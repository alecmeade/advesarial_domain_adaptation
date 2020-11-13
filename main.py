import argparse


from adapt import adapt_model
from dataloaders import DatasetType
from evaluate import evaluate_model
from logging_utils import Logger, TRAIN_PREFIX, ADAPT_PREFIX
from train import train_model


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_suffix', type=str, default = "2")
    arg_parser.add_argument('--source_dataset', type=str, default = "MNIST")
    arg_parser.add_argument('--target_datasets', type=str, default = "USPS")
    arg_parser.add_argument('--n_train_samples', type=int, default = None)
    arg_parser.add_argument('--n_test_samples', type=int, default = None)
    arg_parser.add_argument('--batch_size', type=int, default = 64) 
    arg_parser.add_argument('--train_epochs', type=int, default = 40)
    arg_parser.add_argument('--adapt_epochs', type=int, default = 1000)
    arg_parser.add_argument('--adapt_lr', type=int, default = 0.0001)
    arg_parser.add_argument('--adapt_discrim_batch_interval', type=int, default = 3)
    arg_parser.add_argument('--adapt_target_batch_interval', type=int, default = 1)
    args = arg_parser.parse_args()

    logger = Logger("_".join([args.source_dataset, 
                              str(args.n_train_samples), 
                              str(args.n_test_samples),
                              args.log_suffix]))

    logger.create_log("params")
    for name, value in vars(args).items():
        logger.write_log_line([name, str(value)])

    source = DatasetType[args.source_dataset]
    train_model(logger, source, args)
    evaluate_model(logger, source, TRAIN_PREFIX, args)

    for target in args.target_datasets.split(","):
        target = DatasetType[target]
        evaluate_model(logger, target, TRAIN_PREFIX, args)
        adapt_model(logger, source, target, args)
        evaluate_model(logger, target, ADAPT_PREFIX, args)

