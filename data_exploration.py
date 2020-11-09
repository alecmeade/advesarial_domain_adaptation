import dataloaders

data_dir = "data/"
dataloader_params = {
    'batch_size': 16,
    'pin_memory': True
}

mnist_train = dataloaders.get_mnist_dataloader(data_dir, True, dataloader_params)
mnist_test = dataloaders.get_mnist_dataloader(data_dir, False, dataloader_params)

usps_train = dataloaders.get_usps_dataloader(data_dir, True, dataloader_params)
usps_test = dataloaders.get_usps_dataloader(data_dir, False, dataloader_params)

svhn_train = dataloaders.get_svhn_dataloader(data_dir, True, dataloader_params)
svhn_test = dataloaders.get_svhn_dataloader(data_dir, False, dataloader_params)