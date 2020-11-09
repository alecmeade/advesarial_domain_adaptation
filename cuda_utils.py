import torch
import gc


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