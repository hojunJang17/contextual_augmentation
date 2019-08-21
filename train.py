import json
import pickle
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from pytorch_pretrained_bert import BertConfig
from model.data import Corpus
from model.utils import batchify, evaluate, masking
from model.net import BertForMaskedLM


with open('experiment/config.json') as f:
    params = json.loads(f.read())

# loading token vocab
token_vocab_path = params['filepath'].get('token_vocab')
with open(token_vocab_path, 'rb') as f:
    token_vocab = pickle.load(f)

# loading params
epochs = params['training'].get('epochs')
batch_size = params['training'].get('batch_size')
learning_rate = params['training'].get('learning_rate')
label_size = params['training'].get('num_classes')

# create dataset, dataloader
train_path = params['filepath'].get('train')
val_path = params['filepath'].get('val')

train_data = Corpus(train_path, token_vocab.to_indices, label_size)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4,
                          drop_last=True, collate_fn=batchify)
val_data = Corpus(val_path, token_vocab.to_indices, label_size)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4,
                        drop_last=True, collate_fn=batchify)

# model
config = BertConfig('bert/bert_config.json')
model = BertForMaskedLM(config, label_size, token_vocab)
bert_pretrained = torch.load('bert/pytorch_model.bin')
model.load_state_dict(bert_pretrained, strict=False)

# optimizer
opt = optim.Adam([{"params": model.parameters(), "lr": learning_rate}])

# device
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# train
for epoch in tqdm(range(epochs), desc='epochs'):
    tr_loss = 0
    model.train()
    for step, mb in tqdm(enumerate(train_loader), desc='train_steps', total=len(train_loader)):
        x_mb, y_mb, length = map(lambda elm: elm.to(device), mb)
        opt.zero_grad()
        mb_loss = model(masking(x_mb, length, token_vocab), y_mb, masked_lm_labels=x_mb)
        mb_loss.backward()
        opt.step()

        tr_loss += mb_loss.item()
    else:
        tr_loss /= (step+1)

    val_loss = evaluate(model, val_loader, device)
    tqdm.write('epoch: {}, tr_loss: {:.3f}, val_loss: {:.3f}'.format(epoch+1, tr_loss, val_loss))

model.cpu()
ckpt = {'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict()}
save_path = params['filepath'].get('ckpt')
# save_path = 'non_masking.pth'
torch.save(ckpt, save_path)