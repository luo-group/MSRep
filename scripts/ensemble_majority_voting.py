import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import argparse, os
from utils import commons
from collections import Counter

def majority_voting(predictions):
    ensemble_predictions = []
    n_models = len(predictions)
    n_samples = len(predictions[0])
    for i in range(n_samples):
        merged_prediction = []
        voted_prediction = []
        for j in range(n_models):
            merged_prediction.extend(predictions[j][i])
        counter = Counter(merged_prediction)
        for key, value in counter.items():
            if value > n_models / 2:
                voted_prediction.append(key)
        if len(voted_prediction) == 0:
            voted_prediction = predictions[0][i]
        ensemble_predictions.append(voted_prediction)
    
    return ensemble_predictions


def get_args():
    parser = argparse.ArgumentParser(description='Ensemble using majority voting')
    parser.add_argument('config', type=str, help='Path to the config file')
    parser.add_argument('--logdir', type=str, default='logs_ensemble')
    parser.add_argument('-o', '--output', type=str, default='ensemble_majority_voting_S50.csv')
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
    prediction_lists = [df['Predictions'].tolist() for df in prediction_dfs]
    n_models = len(prediction_lists)
    n_samples = len(prediction_lists[0])
    for i in range(n_models):
        for j in range(n_samples):
            prediction_lists[i][j] = prediction_lists[i][j].split(';')
    ground_truth = prediction_dfs[0]['Ground Truth'].tolist()
    entries = prediction_dfs[0]['Entry'].tolist()
    
    ensemble_predictions = majority_voting(prediction_lists)
    ensemble_predictions = [';'.join(p) for p in ensemble_predictions]
    
    ensembled_results = pd.DataFrame({'Entry': entries, 'Predictions': ensemble_predictions, 'Ground Truth': ground_truth})
    ensembled_results.to_csv(args.output, index=False)
    logger.info('Ensemble results saved to {}'.format(args.output))
    
if __name__ == '__main__':
    main()