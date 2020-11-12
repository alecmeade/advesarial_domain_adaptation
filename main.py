import argparse


from adapt import adapt_model
from dataloaders import DatasetType
from evaluate import evaluate_model
from logging_utils import Logger, TRAIN_PREFIX, ADAPT_PREFIX
from train import train_model


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_suffix', type=str, default = "0")
    arg_parser.add_argument('--source_dataset', type=str, default = "USPS")
    arg_parser.add_argument('--target_datasets', type=str, default = "SVHN,MNIST")
    arg_parser.add_argument('--n_train_samples', type=int, default = None)
    arg_parser.add_argument('--n_test_samples', type=int, default = None)
    arg_parser.add_argument('--batch_size', type=int, default = 128) 
    arg_parser.add_argument('--train_epochs', type=int, default = 30)
    arg_parser.add_argument('--adapt_epochs', type=int, default = 300)
    args = arg_parser.parse_args()

    print(args.target_datasets.split(","))
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

