import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datasets.sequence_dataset import MultiLabelSplitDataset
from models.mlp import *
from utils import commons
from utils.losses import NCLoss, pairwise_cosine_distance

torch.set_num_threads(4)

def get_ec2occurrence(data_file, label_file, label_name, label_level):
    print(f'Loading {data_file} for occurrence statistics...')
    data = torch.load(data_file)
    with open(label_file, 'r') as f:
        label_list = json.load(f)
    ec2occurrence = {label: 0 for label in label_list}
    for k, v in data.items():
        for label in v[label_name]:
            ec2occurrence['.'.join(label.split('.')[:label_level])] += 1
    
    return ec2occurrence, label_list

def evaluate(model, val_loader, criterion, device, use_NC=False):
    model.eval()
    all_loss = []
    all_output = []
    if use_NC:
        all_sup_loss, all_nc1_loss, all_nc2_loss, all_max_cosine = [], [], [], []
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output, features = model(data)
            if use_NC:
                loss, (sup_loss, nc1_loss, nc2_loss, max_cosine, means) = criterion(output, label, features)
                all_sup_loss.append(commons.toCPU(sup_loss).item())
                all_nc1_loss.append(commons.toCPU(nc1_loss).item())
                all_nc2_loss.append(commons.toCPU(nc2_loss).item())
                all_max_cosine.append(commons.toCPU(max_cosine).item())
            else:
                loss = criterion(output, label)
            all_loss.append(commons.toCPU(loss).item())
            all_output.append(commons.toCPU(output))
        all_loss = torch.tensor(all_loss)
    model.train()
    
    if use_NC:
        return all_loss.mean().item(), (torch.tensor(all_sup_loss).mean().item(), torch.tensor(all_nc1_loss).mean().item(), torch.tensor(all_nc2_loss).mean().item(), torch.tensor(all_max_cosine).mean().item())
    else:
        return all_loss.mean().item()

from line_profiler import profile
@profile
def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, logger, config, use_NC=False, writer=None, resume=False, all_checkpoints=None):
    model.train()
    n_bad = 0 if not resume else all_checkpoints['n_bad']
    all_loss = []
    all_val_loss = []
    best_val_loss = 1.e10 if not resume else all_checkpoints['best_val_loss']
    epsilon = 1e-4
    if use_NC and config.start_NC_epoch > 0:
        logger.info(f'Not training NC loss until epoch {config.start_NC_epoch}')
        criterion.set_lambda(0, 0)
        criterion.freeze_means()
    for epoch in range(all_checkpoints['epoch'] + 1 if resume else 0, config.num_epochs):
        start_epoch = time.time()
        if use_NC and epoch == config.start_NC_epoch:
            logger.info(f'Start training NC loss at epoch {epoch}')
            criterion.set_lambda(config.lambda1, config.lambda2)
            criterion.unfreeze_means()
            best_val_loss = 1.e10
        if use_NC:
            val_loss, (val_sup_loss, val_nc1_loss, val_nc2_loss, val_max_cosine) = evaluate(model, val_loader, criterion, device, use_NC)
        else:
            val_loss = evaluate(model, val_loader, criterion, device, use_NC)
        if val_loss > best_val_loss - epsilon:
            n_bad += 1
            if n_bad > config.patience:
                logger.info(f'No performance improvement for {config.patience} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! val_loss={val_loss:.4f}')
            n_bad = 0
            best_val_loss = val_loss
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(config.ckpt_dir, 'best_checkpoints.pt'))
            torch.save(criterion.NC1.means.detach().cpu(), os.path.join(config.ckpt_dir, 'best_means.pt'))
        all_val_loss.append(val_loss)
        losses = []
        if use_NC:
            sup_losses, nc1_losses, nc2_losses, max_cosines = [], [], [], []
        for data, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', dynamic_ncols=True):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output, features = model(data)
            if use_NC:
                loss, (sup_loss, nc1_loss, nc2_loss, max_cosine, means) = criterion(output, label, features)
                sup_losses.append(commons.toCPU(sup_loss).item())
                nc1_losses.append(commons.toCPU(nc1_loss).item())
                nc2_losses.append(commons.toCPU(nc2_loss).item())
                max_cosines.append(commons.toCPU(max_cosine).item())
            else:
                loss = criterion(output, label)
            losses.append(commons.toCPU(loss).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = torch.tensor(losses).mean().item()
        all_loss.append(mean_loss)
        lr_scheduler.step(mean_loss)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(config.ckpt_dir, 'last_checkpoints.pt'))
        torch.save(criterion.NC1.means.detach().cpu(), os.path.join(config.ckpt_dir, 'last_means.pt'))
        all_checkpoints = {'epoch': epoch, 'model': state_dict, 'optimizer': optimizer.state_dict(), 'criterion': criterion.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'n_bad': n_bad, 'best_val_loss': best_val_loss}
        torch.save(all_checkpoints, os.path.join(config.ckpt_dir, 'all_checkpoints.pt'))
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}]: loss: {mean_loss:.4f}; val_loss: {val_loss:.4f}; train time: {commons.sec2min_sec(end_epoch - start_epoch)}; lr: {lr_scheduler.get_last_lr()}')
        writer.add_scalar('Train/loss', mean_loss, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)
        if use_NC:
            writer.add_scalar('Val/sup_loss', val_sup_loss, epoch)
            writer.add_scalar('Val/nc1_loss', val_nc1_loss, epoch)
            writer.add_scalar('Val/nc2_loss', val_nc2_loss, epoch)
            writer.add_scalar('Val/max_cosine', val_max_cosine, epoch)
            writer.add_scalar('Train/sup_loss', torch.tensor(sup_losses).mean().item(), epoch)
            writer.add_scalar('Train/nc1_loss', torch.tensor(nc1_losses).mean().item(), epoch)
            writer.add_scalar('Train/nc2_loss', torch.tensor(nc2_losses).mean().item(), epoch)
            writer.add_scalar('Train/max_cosine', torch.tensor(max_cosines).mean().item(), epoch)
    if use_NC:
        torch.save(criterion.NC1.means.detach().cpu(), os.path.join(config.ckpt_dir, 'means.pt'))
    return all_loss, all_val_loss

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset, self).__init__(dataset, indices)
        self.copy_attributes(dataset)

    def copy_attributes(self, dataset):
        for attr in dir(dataset):
            # Make sure we're only copying relevant attributes
            # You might want to exclude methods or system attributes starting with '__'
            if not attr.startswith('__') and not callable(getattr(dataset, attr)):
                setattr(self, attr, getattr(dataset, attr))

def get_args():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('config', type=str, default='configs/train_mlp.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--lambda1', type=float, default=None)
    parser.add_argument('--lambda2', type=float, default=None)
    parser.add_argument('--start_NC_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--nc1', type=str, default=None)
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--random_split_train_val', action='store_true')
    parser.add_argument('--nc_only', action='store_true')
    parser.add_argument('--fixed_means', action='store_true')
    parser.add_argument('--weight_factor', type=float, default=None)
    parser.add_argument('--resume_model_dir', type=str, default=None)
    
    args = parser.parse_args()
    return args


def main():
    start_overall = time.time()
    args = get_args()
    # Load configs
    if args.resume_model_dir is not None:
        print(f'Resuming training from {args.resume_model_dir}')
        config = commons.load_config(os.path.join(args.resume_model_dir, 'config.yml'))
    else:
        config = commons.load_config(args.config)
        config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.train.seed = args.seed if args.seed is not None else config.train.seed
    commons.seed_all(config.train.seed)
    config.train.lambda1 = args.lambda1 if args.lambda1 is not None or not hasattr(config.train, 'lambda1') else config.train.lambda1
    config.train.lambda2 = args.lambda2 if args.lambda2 is not None or not hasattr(config.train, 'lambda2') else config.train.lambda2
    config.train.start_NC_epoch = args.start_NC_epoch if args.start_NC_epoch is not None or not hasattr(config.train, 'start_NC_epoch') else config.train.start_NC_epoch
    config.data.label_level = 4 if not hasattr(config.data, 'label_level') else config.data.label_level
    config.train.lr = args.lr if args.lr is not None or not hasattr(config.train, 'lr') else config.train.lr
    config.train.weight_decay = args.weight_decay if args.weight_decay is not None or not hasattr(config.train, 'weight_decay') else config.train.weight_decay
    config.train.batch_size = args.batch_size if args.batch_size is not None or not hasattr(config.train, 'batch_size') else config.train.batch_size
    config.train.nc1 = args.nc1 if args.nc1 is not None or not hasattr(config.train, 'nc1') else config.train.nc1
    config.train.fixed_means = args.fixed_means if args.fixed_means or not hasattr(config.train, 'fixed_means') else config.train.fixed_means
    config.train.weight_factor = args.weight_factor if args.weight_factor is not None or not hasattr(config.train, 'weight_factor') else config.train.weight_factor

    # Logging
    if args.resume_model_dir is not None:
        log_dir = args.resume_model_dir
    else:   
        log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=not args.no_timestamp)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    config.train.ckpt_dir = ckpt_dir
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = commons.get_logger('train_mlp', log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(f'Resuming training from {args.resume_model_dir}')
    logger.info(args)
    logger.info(config)
    
    # Load dataset
    if args.random_split_train_val:
        logger.info('Randomly split train and validation set')
        all_data = torch.load(config.data.original_train_data_file)
        pids = list(all_data.keys())
        num_train_val = len(pids)
        indices = commons.get_random_indices(num_train_val, seed=config.train.seed)
        train_indices = indices[:int(num_train_val * 0.9)]
        val_indices = indices[int(num_train_val * 0.9):]
        train_pids = [pids[i] for i in train_indices]
        val_pids = [pids[i] for i in val_indices]
        train_data = {pid: all_data[pid] for pid in train_pids}
        val_data = {pid: all_data[pid] for pid in val_pids}
        with open(os.path.join(log_dir, 'train_val_pids.json'), 'w') as f:
            json.dump({'train': train_pids, 'val': val_pids}, f)
        trainset = globals()[config.data.dataset_type](train_data, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
        validset = globals()[config.data.dataset_type](val_data, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
    else:
        trainset = globals()[config.data.dataset_type](config.data.train_data_file, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
        validset = globals()[config.data.dataset_type](config.data.valid_data_file, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
    testset = globals()[config.data.dataset_type](config.data.test_data_file, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
    train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(validset, batch_size=config.train.batch_size, shuffle=False, pin_memory=True)
    # test_loader = DataLoader(testset, batch_size=config.train.batch_size, shuffle=False)
    config.model.out_dim = trainset.num_labels
    logger.info(f'Trainset size: {len(trainset)}; Validset size: {len(validset)}; Testset size: {len(testset)}')
    logger.info(f'Number of labels: {trainset.num_labels}')
    
    # load all checkpoints if resuming
    if args.resume_model_dir is not None:
        all_checkpoints = torch.load(os.path.join(args.resume_model_dir, 'checkpoints', 'all_checkpoints.pt'), map_location=args.device)
    
    # Load model
    model = globals()[config.model.model_type](config.model)
    model.to(args.device)
    if args.resume_model_dir is not None:
        model.load_state_dict(all_checkpoints['model'])
        logger.info(f'Model loaded from {os.path.join(args.resume_model_dir, "checkpoints", "all_checkpoints.pt")}')
        model.to(args.device)
        model.train()
    if args.nc_only:
        # freeze the last layer of the model
        for param in model.layers[-1].parameters():
            param.requires_grad = False
        model.enable_nc_only()
    logger.info(model)
    logger.info(f'Trainable parameters: {commons.count_parameters(model)}')
    
    # get occurrence list
    ec2occurrence, label_list = get_ec2occurrence(config.data.train_data_file if not hasattr(config.data, 'original_train_data_file') else config.data.original_train_data_file, config.data.label_file, label_name=config.data.label_name, label_level=4)
    occurrence_list = [ec2occurrence[ec] for ec in label_list]
    
    # Train
    if config.train.loss == 'NCLoss':
        logger.info('Using NCLoss')
        criterion = NCLoss(sup_criterion=config.train.sup_criterion, lambda1=config.train.lambda1, lambda2=config.train.lambda2, lambda_CE=config.train.lambda_CE, num_classes=config.model.out_dim, feat_dim=config.model.hidden_dims[-1], device=args.device, nc1=config.train.nc1, nc2=config.train.nc2, occurrence_list=occurrence_list, fixed_means=config.train.fixed_means, weight_factor=config.train.weight_factor)
        optimizer = globals()[config.train.optimizer](list(model.parameters()) + list(criterion.parameters()) if not config.train.fixed_means else model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    else:
        criterion = globals()[config.train.loss]()
        optimizer = globals()[config.train.optimizer](model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.train.patience-10, verbose=True)
    
    if args.resume_model_dir is not None:
        optimizer.load_state_dict(all_checkpoints['optimizer'])
        criterion.load_state_dict(all_checkpoints['criterion'])
        lr_scheduler.load_state_dict(all_checkpoints['lr_scheduler'])
        logger.info(f'Optimizer, criterion, lr_scheduler loaded from {os.path.join(args.resume_model_dir, "checkpoints", "all_checkpoints.pt")}')
    
    commons.save_config(config, os.path.join(log_dir, 'config.yml'))
    
    train(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device, logger=logger, config=config.train, use_NC=(config.train.loss == 'NCLoss'), writer=writer, resume=args.resume_model_dir is not None, all_checkpoints=all_checkpoints if args.resume_model_dir is not None else None)

    end_overall = time.time()
    logger.info(f'Elapsed time: {commons.sec2hr_min_sec(end_overall - start_overall)}')
    
    
if __name__ == '__main__':
    main()
