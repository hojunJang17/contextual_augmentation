import pickle
import torch
import typing
from torch.utils.data import Dataset


class Corpus(Dataset):

    def __init__(self, filename:str, token_fn:callable, label_size:int):
        """
        Args:
            filename (str): dataset pickle file
            token_fn (function): function that makes tokens to indices
            label_size (int): number of labels
        """
        with open(filename, 'rb') as f:
            self.corpus = pickle.load(f)
        self.token_fn = token_fn
        self.label_size = label_size

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx:int):
        tokens, labels = map(lambda elm: elm, self.corpus[idx])
        tokens2indices = torch.tensor(self.token_fn(tokens), dtype=torch.int64)
        if len(tokens2indices) > 512:
            tokens2indices[511] = torch.tensor(3)
            tokens2indices = tokens2indices[:511]
        length = torch.tensor(len(tokens2indices))
        return tokens2indices, labels, length