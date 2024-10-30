import sys
sys.path.append('.')

import sys
sys.path.append('.')
import pandas as pd
import os, argparse, json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import commons
from utils.losses import pairwise_cosine_distance
import torch
import numpy as np
from models.mlp import *
from tqdm import tqdm
import faiss
from Bio import SeqIO

def load_model(model_dir, device, logger, ckpt_name='best_checkpoints.pt'):
    train_config = commons.load_config(os.path.join(model_dir, 'config.yml'))
    model = globals()[train_config.model.model_type](train_config.model)
    ckpt = torch.load(os.path.join(model_dir, 'checkpoints', ckpt_name), map_location='cpu')
    model.load_state_dict(ckpt)
    logger.info(f'Model loaded from {os.path.join(model_dir, "checkpoints", ckpt_name)}')
    model.to(device)
    model.eval()
    
    return model

def get_pid_embedding_labels(data_file, model, ont, device, logger, tag, save_path=None, model_dir=None):
    data = torch.load(data_file)
    pids, inputs, labels = zip(*[(k, v['embedding'], v[ont]) for k, v in data.items()])
    inputs = torch.vstack(inputs)
    logger.info(f'Loaded {len(inputs)} sequences from {data_file}')
    batch_size = 1000
    num_batches = int(np.ceil(len(inputs) / batch_size))
    embeddings = []
    if tag == 'lookup' and save_path is not None and os.path.exists(os.path.join(model_dir, save_path)):
        embeddings = torch.load(os.path.join(model_dir, save_path))
        logger.info(f'Loaded embeddings from {os.path.join(model_dir, save_path)}')
        return pids, embeddings, labels
    for i in tqdm(range(num_batches), desc='Embedding', dynamic_ncols=True):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size].to(device)
        with torch.no_grad():
            _, batch_embeddings = model(batch_inputs)
        embeddings.append(batch_embeddings.cpu())
    try:
        embeddings = torch.cat(embeddings, dim=0).to(device)
    except:
        embeddings = torch.cat(embeddings, dim=0)
    logger.info(f'Embedding shape: {embeddings.shape}')
    if tag == 'lookup' and save_path is not None:
        torch.save(embeddings, os.path.join(model_dir, save_path))
        logger.info(f'Saved embeddings to {os.path.join(model_dir, save_path)}')
    
    return pids, embeddings, labels

def get_pid_embedding(data_file, model, device, logger, model_dir=None):
    data = torch.load(data_file)
    pids, inputs = zip(*[(k, v['embedding']) for k, v in data.items()])
    inputs = torch.vstack(inputs)
    logger.info(f'Loaded {len(inputs)} sequences from {data_file}')
    batch_size = 1000
    num_batches = int(np.ceil(len(inputs) / batch_size))
    embeddings = []
    for i in tqdm(range(num_batches), desc='Embedding', dynamic_ncols=True):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size].to(device)
        with torch.no_grad():
            _, batch_embeddings = model(batch_inputs)
        embeddings.append(batch_embeddings.cpu())
    try:
        embeddings = torch.cat(embeddings, dim=0).to(device)
    except:
        embeddings = torch.cat(embeddings, dim=0)
    logger.info(f'Embedding shape: {embeddings.shape}')
    
    return pids, embeddings

def load_database(lookup_database):
    #Build an indexed database
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(lookup_database)
    index.add(lookup_database)

    return(index)

def query(index, queries, k=10):
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k)

    return(D, I)

def infer_lookup_query(lookup_emb, lookup_labels, query_emb, query_pids, topk, logger):
    lookup_emb = lookup_emb.cpu()
    query_emb = query_emb.cpu()
    lookup_emb = lookup_emb / (lookup_emb.norm(dim=1, keepdim=True) + 1e-10)
    query_emb = query_emb / (query_emb.norm(dim=1, keepdim=True) + 1e-10)
    lookup_emb = lookup_emb.cpu().numpy()
    query_emb = query_emb.cpu().numpy()
    lookup_database = load_database(lookup_emb)
    D, smallest_k_indices = query(lookup_database, query_emb, topk)
    pred_labels = []
    n_query = len(query_emb)
    for i in tqdm(range(n_query)):
        pred = []
        for j in smallest_k_indices[i]:
            pred.extend(lookup_labels[j])
        pred_labels.append(pred)
    
    pid2pred = {pid: pred for pid, pred in zip(query_pids, pred_labels)}
    
    return pid2pred

def infer_lookup_query_score(lookup_emb, lookup_labels, query_emb, query_pids, topk, logger):
    lookup_emb = lookup_emb.cpu()
    query_emb = query_emb.cpu()
    lookup_emb = lookup_emb / (lookup_emb.norm(dim=1, keepdim=True) + 1e-10)
    query_emb = query_emb / (query_emb.norm(dim=1, keepdim=True) + 1e-10)
    lookup_emb = lookup_emb.cpu().numpy()
    query_emb = query_emb.cpu().numpy()
    lookup_database = load_database(lookup_emb)
    smallest_k_dist, smallest_k_indices = query(lookup_database, query_emb, topk)
    
    pred_labels = []
    n_query = len(query_emb)
    for i in tqdm(range(n_query)):
        pred = []
        for j in smallest_k_indices[i]:
            pred.extend(lookup_labels[j])
        pred_labels.append(pred)
    
    pid2pred = {pid: pred for pid, pred in zip(query_pids, pred_labels)}
    scores = [dist[0] for dist in smallest_k_dist]
    
    return pid2pred, scores

def save_predictions(pid2pred, save_path):
    pids = list(pid2pred.keys())
    predictions = [';'.join(pid2pred[pid]) for pid in pids]
    df = pd.DataFrame({'Entry': pids, 'Predictions': predictions})
    df.to_csv(save_path, index=False)
    
def save_predictions_score(pid2pred, scores, save_path):
    pids = list(pid2pred.keys())
    predictions = [';'.join(pid2pred[pid]) for pid in pids]
    df = pd.DataFrame({'Entry': pids, 'Predictions': predictions, 'Scores': scores})
    df.to_csv(save_path, index=False)
    
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config', type=str, default='configs/infer_NC_lookup.yml')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--lookup_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument('--ont', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='predictions_lookup.csv')
    parser.add_argument('--ckpt_name', type=str, default='best_checkpoints.pt')
    parser.add_argument('--cache_lookup', action='store_true')
    
    args = parser.parse_args()
    
    return args

def main():
    args = get_args()
    config = commons.load_config(args.config)
    config.model_dir = args.model_dir if args.model_dir is not None else config.model_dir
    config.lookup_data = args.lookup_data if args.lookup_data is not None else config.lookup_data
    config.test_data = args.test_data if args.test_data is not None else config.test_data
    config.label_file = args.label_file if args.label_file is not None else config.label_file
    config.topk = args.topk if args.topk is not None else config.topk
    config.ont = args.ont if args.ont is not None else config.ont
    
    logdir = os.path.join(config.model_dir, 'infer_NC_lookup')
    os.makedirs(logdir, exist_ok=True)
    logger = commons.get_logger('infer', logdir)
    
    model = load_model(config.model_dir, args.device, logger, ckpt_name=args.ckpt_name)
    lookup_pids, lookup_emb, lookup_labels = get_pid_embedding_labels(config.lookup_data, model, config.ont, args.device, logger, tag='lookup' if args.cache_lookup else '', save_path='lookup_emb.pt', model_dir=config.model_dir)
    test_pids, test_emb = get_pid_embedding(config.test_data, model, args.device, logger)
    if args.ont not in ['BP', 'MF', 'CC']:
        pid2pred = infer_lookup_query(lookup_emb, lookup_labels, test_emb, test_pids, config.topk, logger)
        save_predictions(pid2pred, args.output)
    else:
        pid2pred = infer_lookup_query_score(lookup_emb, lookup_labels, test_emb, test_pids, config.topk, logger)
        save_predictions_score(pid2pred, args.output)
        
    logger.info(f'Predictions saved to {args.output}')

if __name__ == '__main__':
    main()