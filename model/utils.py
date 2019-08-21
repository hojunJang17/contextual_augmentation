import typing
from typing import List, Union
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import copy


class Vocab:
    def __init__(self, list_of_tokens: List[str] = None, padding_token='<pad>', unknown_token='<unk>',
                 bos_token='<bos>', eos_token='<eos>', reserved_tokens=None, unknown_token_idx=0):
        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._reserved_tokens = reserved_tokens
        self._special_tokens = []

        for tkn in [self._padding_token, self._bos_token, self._eos_token]:
            if tkn:
                self._special_tokens.append(tkn)

        if self._reserved_tokens:
            self._special_tokens.extend(self._reserved_tokens)
        if self._unknown_token:
            self._special_tokens.insert(unknown_token_idx, self._unknown_token)

        if list_of_tokens:
            self._special_tokens.extend(list(filter(lambda elm: elm not in self._special_tokens, list_of_tokens)))

        self._token_to_idx, self._idx_to_token = self._build(self._special_tokens)
        self._embedding = None

    def to_indices(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, list):
            return [self._token_to_idx[tkn] if tkn in self._token_to_idx else self._token_to_idx[self._unknown_token]
                    for tkn in tokens]
        else:
            return self._token_to_idx[tokens] if tokens in self._token_to_idx else\
                self._token_to_idx[self._unknown_token]

    def to_tokens(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(indices, list):
            return [self._idx_to_token[idx] for idx in indices]
        else:
            return self._idx_to_token[indices]

    def _build(self, list_of_tokens):
        token_to_idx = {tkn: idx for idx, tkn in enumerate(list_of_tokens)}
        idx_to_token = {idx: tkn for idx, tkn in enumerate(list_of_tokens)}
        return token_to_idx, idx_to_token

    def __len__(self):
        return len(self._token_to_idx)

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, array):
        self._embedding = array


def batchify(data):
    """
    collate function
    Args:
        data: dataset

    Returns: token indices, label indices, lengths

    """
    tokens2indices, labels2indices, lengths = zip(*data)
    tokens2indices = pad_sequence(tokens2indices, batch_first=True, padding_value=0)
    lengths = torch.stack(lengths, 0)
    return tokens2indices, torch.tensor(labels2indices), lengths


def masking(input_ids, length, token_vocab):
    """
    random masking process
    Args:
        input_ids (torch.tensor): input array of shape [batch_size, sequence_length]
        length (torch.tensor) : length of input array without padding token, [batch_size]
        token_vocab (Vocab)

    Returns: randomly masked array

    """
    num_masked = ((length-3) * 15) // 100
    masked_idx = [torch.randint(low=1, high=length[i]-2, size=(num_masked[i], 1)) for i in range(len(length))]
    i = torch.randint(10, (len(length), 1))
    result = copy.deepcopy(input_ids)
    for k in range(len(result)):
        if i[k] < 8:
            for idx in masked_idx[k]:
                result[k][idx] = token_vocab.to_indices('[MASK]')
        elif i[k] == 9:
            rand_idx = torch.randint(len(token_vocab), (num_masked[k], 1))
            for m in range(num_masked[k]):
                result[k][masked_idx[k][m]] = rand_idx[m]
    return result


def evaluate(model, data_loader, device):
    """
    function for caculating validation loss
    Args:
        model : train model
        data_loader: data_loader
        device (torch.device): device which model works

    Returns: loss

    """
    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(data_loader), desc='eval_step', total=len(data_loader)):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)
        with torch.no_grad():
            mb_loss = model(x_mb, y_mb, masked_lm_labels=x_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step+1)

    return avg_loss

