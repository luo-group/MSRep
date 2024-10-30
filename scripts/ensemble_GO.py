import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import os, argparse, json
from utils import commons

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='logs_GO_tune')
    parser.add_argument('--output', type=str, default='ensemble.csv')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--no-timestamp', action='store_true')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    
    # Logging
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=not args.no_timestamp)
    logger = commons.get_logger('ensemble', log_dir)
    logger.info(args)
    logger.info(config)
    commons.save_config(config, os.path.join(log_dir, 'config.yml'))
    
    prediction_files = config.prediction_files
    prediction_dfs = [pd.read_csv(f) for f in prediction_files]
    entry_pred_score = {}
    for df in prediction_dfs:
        for index, row in df.iterrows():
            entry = row['Entry']
            pred = row['Predictions']
            score = row['Scores']
            if entry not in entry_pred_score:
                entry_pred_score[entry] = {}
            if pred not in entry_pred_score[entry]:
                entry_pred_score[entry][pred] = score
            else:
                entry_pred_score[entry][pred] += score
    n_ensemble = len(prediction_dfs)
    new_entries, new_predictions, new_scores = [], [], []
    for entry, pred_score in entry_pred_score.items():
        for pred, score in pred_score.items():
            new_entries.append(entry)
            new_predictions.append(pred)
            new_scores.append(score / n_ensemble)
    new_df = pd.DataFrame({'Entry': new_entries, 'Predictions': new_predictions, 'Scores': new_scores})
    new_df.to_csv(os.path.join(log_dir, args.output), index=False)
    print(f'Ensemble predictions saved to {os.path.join(log_dir, args.output)}')
    
if __name__ == '__main__':
    main()