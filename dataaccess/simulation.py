import json
import torch

from torch.autograd import Variable


def load_config():
    with open('../arguments/arguments.json') as f:
        arguments = json.load(f)

    return arguments


def load_data_set(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    return data


def numpy_to_tensor(input_array, output_type, cuda_flag, volatile_flag=False):
    if output_type == 'float':
        input_tensor = Variable(torch.FloatTensor(input_array), volatile=volatile_flag)
    elif output_type == 'int':
        input_tensor = Variable(torch.LongTensor(input_array), volatile=volatile_flag)
    else:
        print('undefined tensor type')

    if cuda_flag:
        input_tensor = input_tensor.cuda()
    return input_tensor