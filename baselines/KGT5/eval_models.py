from dataset import T5_Dataset
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from noam_lr_scheduler import NoamLR
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Adafactor
import transformers
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--prefix',type=str, default='temp',
                    help='prefix of file')

parser.add_argument('--checkpoint',type=int,
                    help='number')

parser.add_argument('--dataset',type=str, default='wikidata5m',
                    help='number')
    
args = parser.parse_args()

def removePadding(arr):
    first_pad = (arr == 0).nonzero(as_tuple=True)[0]
    if len(first_pad) == 0:
        return arr
    else:
        last_index = first_pad[0]
        return arr[:last_index]
    

def eval(model, dataset, args=None):
    num_workers = 1
    batch_size = 200
    model.cuda()
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=dataset._collate_without_padding)
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")
    i = 0
    targets = []
    predictions = []
    for steps, batch in enumerate(loader):
        input_ids, attention_mask, labels, labels_attention_mask = batch
        outputs = model.generate(input_ids = input_ids.cuda())
        actual_batch = labels
        predicted_batch = outputs[:, 1:]
        for i in range(len(actual_batch)):
            predict = removePadding(predicted_batch[i])
            actual = removePadding(actual_batch[i])
            predictions.append(predict.cpu().numpy())
            targets.append(actual.cpu().numpy())
            
    correct = 0
    for p, t in zip(predictions, targets):
        p_text = dataset.tokenizedToText(p)
        t_text = dataset.tokenizedToText(t)
        if p_text == t_text:
            correct += 1
    accuracy = correct/len(targets)
    return accuracy    

# args.prefix = 'codex_m_'

# model_prefix = 'model/{}'.format(args.prefix)

# checkpoints = []
# for o in os.listdir('models'):
#     if o.startswith(args.prefix):
#         checkpoints.append('models/{}'.format(o))

# valid_dataset = T5_Dataset('valid', dataset_name='codex-m')


# for c in checkpoints:
#     model = T5ForConditionalGeneration.from_pretrained(c)
#     accuracy = eval(model, valid_dataset)
#     print(c, accuracy)


valid_dataset = T5_Dataset('test', dataset_name=args.dataset)

# checkpoint_location = 'models/trainer_peda/checkpoint-{}'.format(args.checkpoint)
checkpoint_location = 'models/{}_{}.pt'.format(args.prefix, args.checkpoint)
print('Using %s' % checkpoint_location)
model = T5ForConditionalGeneration.from_pretrained(checkpoint_location)
accuracy = eval(model, valid_dataset)
print(accuracy)