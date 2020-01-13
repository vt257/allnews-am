
"""
Usage:
    run.py train --train-src=<file>  [options]
    run.py evaluate [options] MODEL_PATH TEST_SOURCE_FILE

Options:
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --batch-size=<int>                      batch size [default: 32]
    --max-epoch=<int>                       max epoch [default: 30]
    --max-len=<int>                         sentence max size [default: 75]
    --lr=<float>                            learning rate [default: 3e-5]
    --save-to=<file>                        model save path [default: model.bin]
    --train-test-split=<float>              train test split [default: 0.1]
    --full-finetuning                       use full finetuning
"""


import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from typing import List, Tuple, Dict, Set, Union
from seqeval.metrics import f1_score, classification_report
import numpy as np
import sys
from docopt import docopt
import sentence
from tqdm import trange



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(args: Dict):
    MAX_LEN = int(args['--max-len'])
    bs = int(args['--batch-size'])

    model_save_path = args['--save-to']

    dataLoader= sentence.Sentence(args['--train-src'])

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)
    n_gpu = torch.cuda.device_count()

    ##torch.cuda.get_device_name(0)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in dataLoader.sentences]
    print(tokenized_texts[0])

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    tags = pad_sequences([[dataLoader.tag2idx.get(l) for l in lab] for lab in dataLoader.labels],
                         maxlen=MAX_LEN, value=dataLoader.tag2idx["O"], padding="post",
                         dtype="long", truncating="post")

    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    tts = float(args['--train-test-split'])

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=tts)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=tts)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(dataLoader.tag2idx))

    model.cuda();

    FULL_FINETUNING = True if args['--full-finetuning'] else False
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    lr=float(args['--lr'])
    optimizer = Adam(optimizer_grouped_parameters, lr=lr)


    epochs = int(args['--max-epoch'])
    max_grad_norm = 1.0
    hist_valid_scores = []

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        pred_tags = [dataLoader.tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [dataLoader.tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        f1=f1_score(pred_tags, valid_tags)
        print("F1-Score: {}".format(f1))

        is_better = len(hist_valid_scores) == 0 or f1 > max(hist_valid_scores)
        hist_valid_scores.append(f1)
        if is_better:
            model.save(model_save_path)
             # also save the optimizers' state
            torch.save(optimizer.state_dict(), model_save_path + '.optim')

    print('reached maximum number of epochs!', file=sys.stderr)
    exit(0)


def evaluate(args:Dict):
    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)

    dataLoader = sentence.Sentence(args['TEST_SOURCE_FILE'])

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    model=BertForTokenClassification.from_pretrained(args['MODEL_PATH'], num_labels=len(dataLoader.tag2idx))
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in dataLoader.sentences]

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    input_ids_test = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=75, dtype="long", truncating="post", padding="post")

    tags_test = pad_sequences([[dataLoader.tag2idx.get(l) for l in lab] for lab in dataLoader.labels],
                         maxlen=75, value=dataLoader.tag2idx["O"], padding="post",
                         dtype="long", truncating="post")

    attention_masks_test = [[float(i > 0) for i in ii] for ii in input_ids_test]

    te_inputs = torch.tensor(input_ids_test)
    te_tags = torch.tensor(tags_test)
    te_masks = torch.tensor(attention_masks_test)

    test_data = TensorDataset(te_inputs, te_masks, te_tags)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

    model.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    pred_tags = [[dataLoader.tags_vals[p_i] for p_i in p] for p in predictions]
    test_tags = [[dataLoader.tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l]

    tags_test_fin = list()
    for l in tags_test:
        temp_tag = list()
        for l_i in l:
            temp_tag.append(dataLoader.tags_vals[l_i])
        tags_test_fin.append(temp_tag)

    print("Test loss: {}".format(eval_loss / nb_eval_steps))
    print("Test Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Test F1-Score: {}".format(f1_score(tags_test_fin, pred_tags)))

    print(classification_report(tags_test_fin, pred_tags))

    print("Number of Test sentences: ", len(tags_test_fin))

def main():
    """ Main func.
    """
    args = docopt(__doc__)
    # seed the random number generators
    seed = 0
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['test']:
        evaluate(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
