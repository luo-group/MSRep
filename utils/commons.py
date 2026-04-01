import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def convert_easydict_to_dict(obj):
    if isinstance(obj, EasyDict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: convert_easydict_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_easydict_to_dict(v) for v in obj]
    else:
        return obj

def save_config(config, path):
    config = convert_easydict_to_dict(config)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_new_log_dir(root='./logs', prefix='', tag='', timestamp=True):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) if timestamp else ''
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir

# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
def seed_all(seed=42):
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None

# move torch/GPU tensor to numpy/CPU
def toCPU(data):
    return data.cpu().detach()

# count number of free parameters in the network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sec2min_sec(t):
    mins = int(t) // 60
    secs = int(t) % 60
    
    return f'{mins}[min]{secs}[sec]'

def sec2hr_min_sec(t):
    hrs = int(t) // 3600
    mins = (int(t) % 3600) // 60
    secs = (int(t) % 3600) % 60
    
    return f'{hrs}[hr]{mins}[min]{secs}[sec]'

def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices
