import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.cuda_flag = args.cuda


    def encode(self):
        raise NotImplementedError
