import dataloaders

from collections import defaultdict

def get_label_dist(data):
	label_dist = defaultdict(int)
	total = 0
	for _, y in data:
		for i in range(y.shape[0]):
			label_dist[y[i].item()] += 1
			total += 1

	sorted_keys = sorted(label_dist.keys())
	dist = []
	for i in range(len(sorted_keys)):
		k = sorted_keys[i]
		v = 100 * label_dist[k] / float(total)
		dist.append([k, round(v, 1)])

	return dist

def count_samples(data):
	cnt = 0
	for x, y in data:
		cnt += y.shape[0]
	return cnt

def summarize_dataset(name, train, test):
	train_n = count_samples(train)
	test_n = count_samples(test)

	print("\n")
	print(name)
	print("Train Samples: %d" % train_n)
	print("Test Samples: %d" % test_n)
	print("Train / Test Ratio: %f" % (float(train_n) / float(test_n)))
	print("X Dimensions: ", next(iter(train))[0].shape)
	print("Y Dimensions: ", next(iter(train))[1].shape)
	print("Train:")
	print(get_label_dist(train))
	print("Test:")
	print(get_label_dist(test))

if __name__ == "__main__":

	dataloader_params = {
	    'batch_size': 16,
	    'pin_memory': True
	}

	train_n = None
	test_n = None

	mnist_train = dataloaders.get_mnist_dataloader(True, dataloader_params, train_n)
	mnist_test = dataloaders.get_mnist_dataloader(False, dataloader_params, test_n)

	usps_train = dataloaders.get_usps_dataloader(True, dataloader_params, train_n)
	usps_test = dataloaders.get_usps_dataloader(False, dataloader_params, test_n)

	svhn_train = dataloaders.get_svhn_dataloader(True, dataloader_params, train_n)
	svhn_test = dataloaders.get_svhn_dataloader(False, dataloader_params, test_n)


	datasets = [
		["MNIST", mnist_train, mnist_test],
		["USPS", usps_train, usps_test],
		["SVHN", svhn_train, svhn_test],
	]

	for name, train, test in datasets:
		summarize_dataset(name, train, test)