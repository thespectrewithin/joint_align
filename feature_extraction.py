import argparse
import logging
import os
import random
import sys
import time
import numpy as np

import torch
from transformers import BertModel, BertTokenizer
#from pytorch_pretrained_bert.modeling import BertModel
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from utils import NerProcessor

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    
    for (ex_index,example) in enumerate(examples):
        
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("<PAD>")
                    
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        label_ids = [0] + [label_map[lb] for lb in labels] + [0]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_ids))
    return features

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir", default="./", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--data', type=str, help='path of data')
    parser.add_argument('--align_matrix', type=str, help='path of alignment matrices')
    parser.add_argument('--layers', type=str, help='sum of features from layer a to layer b. e.g., 9:13')
    parser.add_argument("--batch_size", default=32, type=int)

    parser.add_argument('--output', type=str, help='output path of extracted features')

    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)
    logging.info("Input args: %r" % args)

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = processor.get_train_examples(args.data)
    all_lengths = [len(tokenizer.tokenize(t.text_a)) for t in train_examples]
    max_seq_len = max(all_lengths) + 2
    logger.info("Max sequence length: %d" % max_seq_len)
    logger.info("Max sequence: %s" % ' '.join(tokenizer.tokenize(train_examples[all_lengths.index(max_seq_len - 2)].text_a)))
    
    if max_seq_len > 512:
        max_seq_len = 512
        logger.info("Max sequence length reset to 512")

    #model = BertModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
    model = BertModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir, output_hidden_states=True)
    model.to(device)

    train_features = convert_examples_to_features(train_examples, label_list, max_seq_len, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    layer_1 = int(args.layers.split(':')[0])
    layer_2 = int(args.layers.split(':')[1])

    if args.align_matrix:
        W = []
        for i in range(layer_1, layer_2):
            temp = np.loadtxt(args.align_matrix + '.' + str(i) + '.map')
            temp = torch.tensor(temp, dtype=torch.float).to(device)
            W.append(temp)

    model.eval()
    to_save = {}

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            #all_encoder_layers, _ = model(input_ids, segment_ids, input_mask)
            all_encoder_layers = model(input_ids, attention_mask=input_mask)[2]

        output = []
        for i, j in enumerate(range(layer_1, layer_2)):
            if args.align_matrix:
                output.append(torch.matmul(all_encoder_layers[j], W[i]))
            else:
                output.append(all_encoder_layers[j])
        output_ = torch.sum(torch.stack(output), dim=0)

        for i in range(len(input_ids)):
            sent_id = i + step * args.batch_size
            layer_output = output_[i,:input_mask[i].to('cpu').sum()]
            to_save[sent_id] = layer_output.detach().cpu().numpy()
                
    torch.save(to_save, args.output + '.pth')

if __name__ == "__main__":
    main()
