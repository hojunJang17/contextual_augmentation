#-*- coding:utf-8 -*-

import json
import torch
import pickle
import copy
from typing import List
from model.net import BertForMaskedLM
from bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertConfig


def restore(text : List[str]):
    """
    making tokenized word list to full string
    Args:
        text (list[str]) : tokenized word list

    Returns:
        full string made from word list
    """
    a = ''
    for i in range(1, len(text)-2):
        if text[i] not in ['[SEP]', '[CLS]', '[PAD]']:
            if i==1 and text[i]=='_':
                continue
            if text[i].endswith('_'):
                a += text[i][:-1]
                a += ' '
            else:
                a += text[i]
    return a

with open('experiment/config.json') as f:
    params = json.loads(f.read())

tokenizer = BertTokenizer.from_pretrained('bert/vocab.korean.rawtext.list', do_lower_case=False)

token_vocab_path = params['filepath'].get('token_vocab')
with open(token_vocab_path, 'rb') as f:
    token_vocab = pickle.load(f)

num_labels = params['training'].get('num_classes')
save_path = params['filepath'].get('ckpt')
ckpt = torch.load(save_path)
config = BertConfig('bert/bert_config.json')
model = BertForMaskedLM(config, num_labels, token_vocab)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

while True:
    print('-'*60)
    try:
        text = input('input  : ')
    except UnicodeDecodeError:
        print("\tError occured")
        continue
    # text = '정말 재미없는 영화인 것 같습니다.'
    if text == '':
        continue
    if text == '..':
        break
    label = torch.tensor([int(input('label  : '))])
    sp = text.split()
    if len(sp) == 1:
        print('output :', text)
        continue
    choice = sp[torch.randint(len(sp), size=(1, 1))]
    print(tokenizer.tokenize(choice))
    tokenized_text = tokenizer.tokenize('[CLS] '+text+' [SEP]')
    masked_index = [tokenized_text.index(elm) for elm in tokenizer.tokenize(choice)]
    for i in masked_index:
        tokenized_text[i] = '[MASK]'

    made_text = []
    for i in range(3):
        made_text += [copy.deepcopy(tokenized_text)]

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [0] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segment_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        predictions = model(tokens_tensor, label, segment_tensors)

    for k in range(len(made_text)):
        predicted_index = torch.argmax(predictions[0, masked_index[0]]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        made_text[k][masked_index[0]] = predicted_token
        predictions[0, masked_index[0]][predicted_index] = -10
        if len(masked_index) > 1:
            for i in masked_index[1:]:
                indexed_token = tokenizer.convert_tokens_to_ids(made_text[k])
                segments_id = [0] * len(made_text[k])
                tokens_tensors = torch.tensor([indexed_token])
                segment_tensor = torch.tensor([segments_id])
                with torch.no_grad():
                    predictions1 = model(tokens_tensors, label, segment_tensor)
                predicted_index1 = torch.argmax(predictions1[0, i]).item()
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index1])[0]
                made_text[k][i] = predicted_token


    print('label  :', label.item())
    for i in range(len(made_text)):
        print('Output :', restore(made_text[i]))
    print()
