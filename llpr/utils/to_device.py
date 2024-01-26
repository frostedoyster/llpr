import torch

def to_device(device, *args):
    result = ()
    for arg in args:
        if isinstance(arg, torch.Tensor):
            result += (arg.to(device),)
        elif isinstance(arg, tuple):
            result += (to_device(device, *arg),)
        elif isinstance(arg, list):
            result += (to_device(device, *arg),)
        else:
            result += (arg,)
    return result
