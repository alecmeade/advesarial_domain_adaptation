import gc
import torch
import numpy as np
import re

def clear_device():
    torch.cuda.empty_cache()
    gc.collect()


def maybe_get_cuda_device():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  

    clear_device()
    return torch.device(dev)  


def compute_metrics(metrics):
    values = []
    for name, metric in metrics.items():
        v = metric.compute()
        if "conf" not in name:
            values.append(str(v))
        else:
            cm_str = str(np.array(v))
            cm_str = re.sub('\s+',', ', cm_str)
            values.append(cm_str)
    return values


def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset()


def update_metrics(metrics, pred, y_batch):
    for name, metric in metrics.items():
        metric.update((pred, y_batch))

# x_batch, y_batch = next(iter(train_loader))
# for i in range(x_batch.shape[0]):
#     plt.figure()
#     plt.title(y_batch[i])
#     img = x_batch[i, :, :, :].squeeze()
#     img = np.transpose(img.numpy(), (1, 2, 0))
#     plt.imshow(img)
#     plt.show()

#     plt.figure()
#     print(img.shape)
#     sns.distplot(img[:, :, 0].reshape(-1))
#     plt.show()
#     if i > 4:
#         break