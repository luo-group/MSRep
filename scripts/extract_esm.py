import sys
sys.path.append('.')

import torch
import pandas as pd
import os, argparse, json
from Bio import SeqIO
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input', type=str, help='csv file', default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--ont', type=str, default=None, choices=['ec', 'gene3D', 'pfam', 'BP', 'MF', 'CC'])
    parser.add_argument('--cache_dir', type=str, default='esm_embeddings')
    
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
    tmp_fasta_file = os.path.join(os.path.dirname(args.input), os.path.basename(args.input).replace('.csv', '.fasta'))
    print(f'Generating temporary fasta file {tmp_fasta_file} for {len(uncached_entries)} uncached proteins...')
    with open(tmp_fasta_file, 'w') as f:
        for entry, sequence in zip(uncached_entries, uncached_sequences):
            f.write(f'>{entry}\n{sequence}\n')
    print('Running ESM...')
    os.makedirs(args.cache_dir, exist_ok=True)
    os.system(f'esm-extract esm1b_t33_650M_UR50S {tmp_fasta_file} {args.cache_dir} --repr_layers 33 --include mean')

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
    
if __name__ == '__main__':
    main()