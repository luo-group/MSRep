import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict
import json
import matplotlib.pyplot as plt

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

def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices

def get_leveled_acc(pred_labels, gt_labels, label2occurrence, log_dir, levels=[10, 30, 100]):
    assert len(pred_labels) == len(gt_labels), f'{len(pred_labels)} != {len(gt_labels)}'
    label2correct, label2test_num = {}, {}
    n = len(pred_labels)
    for i in range(n):
        if gt_labels[i] not in label2test_num:
            label2test_num[gt_labels[i]] = 0
            label2correct[gt_labels[i]] = 0
        label2test_num[gt_labels[i]] += 1
        if pred_labels[i] == gt_labels[i]:
            label2correct[gt_labels[i]] += 1
    label2acc = {k: label2correct[k] / label2test_num[k] for k in label2test_num}
    occurrence_levels = []
    occurrence_levels.append(f'[{levels[-1]}, +$\infty$)')
    for i in range(len(levels) - 1, 0, -1):
        occurrence_levels.append(f'[{levels[i-1]}, {levels[i]})')
    occurrence_levels.append(f'[0, {levels[0]})')
    print(f'occurrence_levels: {occurrence_levels}')
    level2correct, level2test_num = {level: 0 for level in occurrence_levels}, {level: 0 for level in occurrence_levels}
    for label in label2test_num:
        occurrence = label2occurrence[label]
        for i in range(len(levels)):
            if occurrence >= levels[len(levels) - 1 - i]:
                level2test_num[occurrence_levels[i]] += label2test_num[label]
                level2correct[occurrence_levels[i]] += label2correct[label]
                break
        if occurrence < levels[0]:
            level2test_num[occurrence_levels[-1]] += label2test_num[label]
            level2correct[occurrence_levels[-1]] += label2correct[label]
    level2acc = {level: level2correct[level] / level2test_num[level] for level in level2test_num}
    for level in level2acc:
        print(f'{level}: {level2acc[level]:.4f}')
    if log_dir is not None:
        with open(os.path.join(log_dir, 'level2acc.json'), 'w') as f:
            json.dump(level2acc, f)
            
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=400)
    occurrence_list = sorted(list(label2occurrence.values()), reverse=True)
    axes[0].plot(occurrence_list)
    axes[0].set_xlabel('ECs', fontsize=20)
    axes[0].set_ylabel('occurrence', fontsize=20)
    axes[0].set_title('EC occurrence in train set', fontsize=20)
    
    axes[1].plot(list(level2acc.values()))
    for i, txt in enumerate(list(level2acc.values())):
        axes[1].annotate(f'{txt:.4f}', (i, list(level2acc.values())[i]), fontsize=10)
    axes[1].set_xticks(range(len(occurrence_levels)), occurrence_levels)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('EC occurrence levels', fontsize=20)
    axes[1].set_ylabel('Accuracy', fontsize=20)
    axes[1].set_title('EC Accuracy with descending occurrence', fontsize=20)
    
    plt.tight_layout()
    if log_dir is not None:
        plt.savefig(os.path.join(log_dir, 'level2acc.png'), bbox_inches='tight')
        
    return level2acc

def n_smallest(data, n):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    smallest_values, indices = torch.topk(-data, k=n)
    smallest_values = -smallest_values
    
    return smallest_values, indices        

def maximum_separation(dist_lst, first_grad=True, use_max_grad=False):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        if not len(large_grads[-1]) == 0:
            max_sep_i = large_grads[-1][opt]
        else:
            max_sep_i = 0
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i

def batch_max_sep(smallest_k_dist, smallest_k_indices, ecs_lookup):
    n_query = len(smallest_k_dist)
    pred_ecs = []
    distances = []
    for i in range(n_query):
        ecs = []
        dist = []
        dist_lst = smallest_k_dist[i]
        if not isinstance(dist_lst, list):
            dist_lst = dist_lst.tolist()
        max_sep_i = maximum_separation(dist_lst)
        for j in range(max_sep_i + 1):
            # print(smallest_k_indices[i][j], len(ecs_lookup))
            EC_j = ecs_lookup[smallest_k_indices[i][j]]
            dist_j = dist_lst[j]
            ecs.append(EC_j)
            dist.append(dist_j)
        pred_ecs.append(ecs)
        distances.append(dist)
    
    return pred_ecs, distances