import sys
sys.path.append('.')
import pandas as pd
import os, argparse, json
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()
    
    df = pd.read_csv(args.input)
    print(df.shape)
    entries = df['Entry'].tolist()
    predictions = df['Predictions'].tolist()
    scores = df['Scores'].tolist()
    n = len(entries)
    new_entries, new_predictions, new_scores = [], [], []
    
    for i in tqdm(range(n)):
        entry = entries[i]
        pred = predictions[i]
        score = scores[i]
        pred = pred.split(';')
        for p in pred:
            new_entries.append(entry)
            new_predictions.append(p)
            new_scores.append(score)
    
    new_df = pd.DataFrame({'Entry': new_entries, 'Predictions': new_predictions, 'Scores': new_scores})
    print(new_df.shape)
    if args.output is not None:
        new_df.to_csv(args.output, index=False)
    else:
        new_df.to_csv(args.input.replace('.csv', '_flatten.csv'), index=False)
    
if __name__ == '__main__':
    main()

