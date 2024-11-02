import sys
sys.path.append('.')

import torch
import pandas as pd
import os, argparse, json
from Bio import SeqIO
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import random

def worker(tmp_fasta_file, cache_dir):
    os.system(f'esm-extract esm1b_t33_650M_UR50S {tmp_fasta_file} {cache_dir} --repr_layers 33 --include mean')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input', type=str, help='csv file', default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--ont', type=str, default=None, choices=['ec', 'gene3D', 'pfam', 'BP', 'MF', 'CC'])
    parser.add_argument('--cache_dir', type=str, default='esm_embeddings')
    parser.add_argument('--devices', nargs='+', default=[0])
    
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    is_labeled = 'Label' in df.columns
    entries = df['Entry'].tolist()
    uncached_entries = []
    for entry in entries:
        cache_path = os.path.join(args.cache_dir, f'{entry}.pt')
        if not os.path.exists(cache_path):
            uncached_entries.append(entry)
    uncached_df = df[df['Entry'].isin(uncached_entries)]
    uncached_entries = uncached_df['Entry'].tolist()
    uncached_sequences = uncached_df['Sequence'].tolist()
    n_jobs = len(args.devices)
    tmp_fasta_files = [os.path.join(os.path.dirname(args.input), os.path.basename(args.input).replace('.csv', f'_{i}.fasta')) for i in range(n_jobs)]
    print(f'Generating {n_jobs} temporary fasta files for {len(uncached_entries)} uncached proteins...')
    indices = list(range(len(uncached_entries)))
    random.shuffle(indices)
    batch_size = int(np.ceil(len(uncached_entries) / n_jobs))
    for i in range(n_jobs):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(uncached_entries))
        with open(tmp_fasta_files[i], 'w') as f:
            for j in range(start, end):
                index = indices[j]
                entry = uncached_entries[index]
                sequence = uncached_sequences[index]
                f.write(f'>{entry}\n{sequence}\n')
    print('Running ESM in parallel...')
    os.makedirs(args.cache_dir, exist_ok=True)
    Parallel(n_jobs=n_jobs)(delayed(worker)(tmp_fasta_files[i], args.cache_dir) for i in range(n_jobs))

    entries = df['Entry'].tolist()
    if is_labeled:
        labels = df['Label'].tolist()
        labels = [label.split(';') for label in labels]
        entry2label = {entry: label for entry, label in zip(entries, labels)}
    data = {}
    print(f'Generating data file for {len(entries)} proteins...')
    for entry in tqdm(entries):
        cache_path = os.path.join(args.cache_dir, f'{entry}.pt')
        emb = torch.load(cache_path)['mean_representations'][33]
        if is_labeled:
            data[entry] = {'embedding': emb, args.ont: entry2label[entry]}
        else:
            data[entry] = {'embedding': emb}
    torch.save(data, args.output)
    for f in tmp_fasta_files:
        os.remove(f)
    
if __name__ == '__main__':
    main()