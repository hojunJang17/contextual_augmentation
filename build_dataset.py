import pandas as pd
import json
from sklearn.model_selection import train_test_split
from bert.tokenization import BertTokenizer
from tqdm import tqdm
import pickle

# loading dataset
with open('experiment/config.json') as f:
    params = json.loads(f.read())

train_path = params['filepath'].get('train')
val_path = params['filepath'].get('val')
test_path = params['filepath'].get('test')

filepath = 'data/ratings_train.txt'
dataset = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
dataset = dataset.loc[dataset['document'].isna().apply(lambda elm: not elm), :]

document = []

ptr_tokenizer = BertTokenizer.from_pretrained('bert/vocab.korean.rawtext.list', do_lower_case=False)
for i in tqdm(range(len(dataset))):
    try:
        document.append([ptr_tokenizer.tokenize('[CLS] ' + dataset['document'][i] + ' [SEP]'), dataset['label'][i]])
    except KeyError:
        continue

tr, val = train_test_split(document, test_size=0.2, random_state=777)

tst_filepath = 'data/ratings_test.txt'
tst = pd.read_csv(tst_filepath, sep='\t').loc[:, ['document', 'label']]
tst = tst.loc[tst['document'].isna().apply(lambda elm: not elm), :]

test_document = []

for i in tqdm(range(len(tst))):
    try:
        test_document.append([ptr_tokenizer.tokenize('[CLS] ' + tst['document'][i] + ' [SEP]'), tst['label'][i]])
    except KeyError:
        continue

with open(train_path, 'wb') as f:
    pickle.dump(tr, f)
with open(val_path, 'wb') as f:
    pickle.dump(val, f)
with open(test_path, 'wb') as f:
    pickle.dump(test_document, f)