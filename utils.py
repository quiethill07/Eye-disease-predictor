import argparse
import torch.nn as nn

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v in (0, 1):
            return bool(v)
        raise argparse.ArgumentTypeError('Boolean value expected.')
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError('Boolean value expected.')

    v = v.strip().lower()
    if v in ['true', '1', 'yes', 'y', 't']:
        return True
    elif v in ['false', '0', 'no', 'n', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
