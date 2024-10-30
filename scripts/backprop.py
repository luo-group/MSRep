import sys
sys.path.append('.')
import os, argparse
import pandas as pd
from goatools.obo_parser import GODag
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

# Function to get ancestors of a GO term
def get_ancestors(go_term, go_dag):
    if go_term not in go_dag:
        return set()
    term = go_dag[go_term]
    ancestors = set(term.get_all_parents())
    return ancestors

def deduplicate_by_max_score(df):
    # Group by 'Entry' and 'Predictions', then keep the row with the max 'Scores'
    deduplicated_df = df.loc[df.groupby(['Entry', 'Predictions'])['Scores'].idxmax()].reset_index(drop=True)
    return deduplicated_df

def get_args():
    parser = argparse.ArgumentParser(description='Go backpropagation')
    parser.add_argument('--go_obo', help='Path to the Gene Ontology OBO file', default='data/GO_new/go-basic.obo')
    parser.add_argument('-i', '--input', help='Path to the input file', required=True)
    parser.add_argument('-o', '--output', help='Path to the output file', default=None)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    go_dag = GODag(args.go_obo)
    
    predictions_df = pd.read_csv(args.input)
    print(f'Original predictions size: {len(predictions_df)}')
    # Store predictions in a dictionary
    pred_dict = defaultdict(lambda: defaultdict(float))
    for index, row in predictions_df.iterrows():
        entry = row['Entry']
        go_term = row['Predictions']
        score = row['Scores']
        pred_dict[entry][go_term] = max(pred_dict[entry][go_term], score)
    # Add ancestor terms to the predictions
    new_predictions = []
    pred_dict_copy = deepcopy(pred_dict)

    for entry, predictions in pred_dict_copy.items():
        current_terms = set(predictions.keys())
        for go_term, score in predictions.items():
            ancestors = get_ancestors(go_term, go_dag)
            for ancestor in ancestors:
                if ancestor not in current_terms:
                    if ancestor not in pred_dict[entry] or score > pred_dict[entry][ancestor]:
                        pred_dict[entry][ancestor] = score
                        new_predictions.append([entry, ancestor, score])
                    current_terms.add(ancestor)

    # Create a new dataframe for the new predictions
    new_predictions_df = pd.DataFrame(new_predictions, columns=['Entry', 'Predictions', 'Scores'])
    new_predictions_df.to_csv('tmp.csv', index=False)

    # Combine the original and new predictions
    combined_df = pd.DataFrame([(entry, term, score) for entry, terms in pred_dict.items() for term, score in terms.items()], columns=['Entry', 'Predictions', 'Scores'])

    print(f'Backpropogated predictions size: {len(combined_df)}')
    
    # Write the updated predictions to a new file
    output_path = os.path.splitext(args.input)[0] + '_backprop.csv' if args.output is None else args.output
    combined_df.to_csv(output_path, index=False)
    print(f'Backpropagated predictions saved to {output_path}')
                  
if __name__ == '__main__':
    main()



