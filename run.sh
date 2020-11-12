python3 main.py --source_dataset=MNIST --target_datasets=SVHN,USPS  --n_train_samples=7000 --n_test_samples=2000
python3 main.py --source_dataset=USPS  --target_datasets=MNIST,SVHN  --n_train_samples=7000 --n_test_samples=2000
python3 main.py --source_dataset=SVHN  --target_datasets=MNIST,USPS  --n_train_samples=7000 --n_test_samples=2000
python3 main.py --source_dataset=MNIST --target_datasets=SVHN       
python3 main.py --source_dataset=SVHN  --target_datasets=MNIST        