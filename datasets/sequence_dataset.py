import torch
from torch.utils.data import Dataset
import json, logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

class MultiLabelSplitDataset(Dataset):
    def __init__(self, data_file, label_list_file, label_name='ec', label_level=4, logger=None) -> None:
        super().__init__()
        self.logger = logger if logger is not None else get_logger('MultilabelDataset')
        self.raw_data = torch.load(data_file) if type(data_file) == str else data_file
        self.logger.info(f'Loaded {len(self.raw_data)} raw multi-label sequences')
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        self.embeddings, self.labels = [], []
        for pid, data in self.raw_data.items():
            for label in data[label_name]:
                self.embeddings.append(data['embedding'])
                label_idx = self.label2idx[label]
                self.labels.append(label_idx)
        self.labels = torch.tensor(self.labels)
        self.logger.info(f'Processed {len(self.embeddings)} single-label sequences')
        self.logger.info(f'Number of labels: {len(self.label_list)}')
        self.num_labels = len(self.label_list)
        
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
    
    def __len__(self):
        return len(self.embeddings)


