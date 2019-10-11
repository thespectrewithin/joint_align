#!/usr/bin/env python

"""Script to convert context embedding with alignment to mapped embeddings

"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from utils import read_align, read_parallel
#from pytorch_pretrained_bert.modeling import BertModel
#from pytorch_pretrained_bert.tokenization import BertTokenizer

def convert_sent_to_input(sents, tokenizer, max_seq_length):
    input_ids = []
    mask = []
    for sent in sents:
        ids = tokenizer.convert_tokens_to_ids(sent)
        mask.append([1] * (len(ids) + 2) + [0] * (max_seq_length - len(ids)))
        input_ids.append([101] + ids + [102] + [0] * (max_seq_length - len(ids)))
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

def convert_words_to_bpe(para, tokenizer):
    
    bpe_para, bpe_table = [], []

    for s in para:
        
        src_sent, tgt_sent = s[0], s[1]
        src_bpe_table, tgt_bpe_table = [], []
        src_sent_bpe, tgt_sent_bpe = [], []
        
        for word in src_sent:
            token = tokenizer.tokenize(word)
            word2bpe_map = []
            for i in range(len(token)):
                word2bpe_map.append(len(src_sent_bpe)+i)
            src_sent_bpe.extend(token)
            src_bpe_table.append(word2bpe_map)

        for word in tgt_sent:
            token = tokenizer.tokenize(word)
            word2bpe_map = []
            for i in range(len(token)):
                word2bpe_map.append(len(tgt_sent_bpe)+i)
            tgt_sent_bpe.extend(token)
            tgt_bpe_table.append(word2bpe_map)
            
        bpe_para.append([src_sent_bpe, tgt_sent_bpe])
        bpe_table.append([src_bpe_table, tgt_bpe_table])
        
    return bpe_para, bpe_table


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--para_file', type=str, required=True,
                        help="Input parallel file")
    parser.add_argument('--align_file', type=str, required=True,
                        help="Input alignment file")
    parser.add_argument('--reverse', type=int, default=0,
                        help="Wheter para and align is reversed")

    parser.add_argument('--align_mode', choices=['word_align', 'sent_avg'],
                        default='word_align', help=" \
                        word_align uses the alignment from align_file \
                        sent_avg uses the averaged embeddign for each sentence")

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir", default="./", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--layer', type=int, default=11,
                        help="layer of bert outputs to be used")

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_length', type=int, default=175)

    parser.add_argument('--output_file', help="Output file path", \
                        default='output')

    args = parser.parse_args(arguments)

    print(args)

    # read parallel
    para = read_parallel(args.para_file, reverse=args.reverse)

    # read alignment
    align_cnt, align_pairs = read_align(args.align_file, reverse=args.reverse)
    print("Read {} aligns".format(align_cnt))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    bpe_para, bpe_table = convert_words_to_bpe(para, tokenizer)

    # filter long/empty sentences
    fltr_para = []
    fltr_align_pairs = []
    fltr_bpe_table = []
    align_cnt = 0
    
    for cnt, [src, trg] in enumerate(bpe_para):
        if len(src) > args.max_seq_length or len(trg) > args.max_seq_length \
            or len(src) == 0 or len(trg) == 0:
            continue
        fltr_para.append([src, trg])
        fltr_align_pairs.append(align_pairs[cnt])
        fltr_bpe_table.append(bpe_table[cnt])
        align_cnt += len(align_pairs[cnt])

    para = fltr_para
    align_pairs = fltr_align_pairs
    print("After filtering {} parallel sentences remains".format(len(para)))

    # print the n-th sentence and alignment
    n = 10
    print("{} ||| {}".format(" ".join(para[n][0]), " ".join(para[n][1])))
    for a in align_pairs[n]:
        print(' '.join([para[n][0][i] for i in fltr_bpe_table[n][0][a[0]]]) + ' : ' + ' '.join([para[n][1][i] for i in fltr_bpe_table[n][1][a[1]]]))

    # bert
    device = torch.device('cuda')

    model = BertModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir, output_hidden_states=True)
    #model = BertModel.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
    model.to(device)

    src_sents = [s[0] for s in para]
    tgt_sents = [s[1] for s in para]
    src_input, src_mask = convert_sent_to_input(src_sents, tokenizer, args.max_seq_length)
    tgt_input, tgt_mask = convert_sent_to_input(tgt_sents, tokenizer, args.max_seq_length)

    src_data = TensorDataset(src_input, src_mask)
    src_sampler = SequentialSampler(src_data)
    src_dataloader = DataLoader(src_data, sampler=src_sampler, batch_size=args.batch_size)

    tgt_data = TensorDataset(tgt_input, tgt_mask)
    tgt_sampler = SequentialSampler(tgt_data)
    tgt_dataloader = DataLoader(tgt_data, sampler=tgt_sampler, batch_size=args.batch_size)

    model.eval()

    src_embed = []
    tgt_embed = []

    for step, batch in enumerate(tqdm(src_dataloader, desc="Iteration")):
        input_ids, input_mask = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        
        with torch.no_grad():
            #all_encoder_layers, _ = model(input_ids, attention_mask=input_mask)
            all_encoder_layers = model(input_ids, attention_mask=input_mask)[2]
            
        src_embed.append(all_encoder_layers[args.layer][:,1:].detach().to('cpu').numpy())

    for step, batch in enumerate(tqdm(tgt_dataloader, desc="Iteration")):
        input_ids, input_mask = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        
        with torch.no_grad():
            #all_encoder_layers, _ = model(input_ids, attention_mask=input_mask
            all_encoder_layers = model(input_ids, attention_mask=input_mask)[2]
            
        tgt_embed.append(all_encoder_layers[args.layer][:,1:].detach().to('cpu').numpy())

    src_embed = np.concatenate(src_embed)
    tgt_embed = np.concatenate(tgt_embed)
    print(src_embed.shape)
    print(tgt_embed.shape)

    feature_size = src_embed.shape[2]
    
    # save the alignments
    if args.align_mode == 'word_align':
        final_res = [np.zeros((align_cnt, feature_size)),
                     np.zeros((align_cnt, feature_size))]
        cnt = 0
    elif args.align_mode == 'sent_avg':
        sent_cnt = len(src_embed)
        final_res = [np.zeros((sent_cnt, feature_size)),
                     np.zeros((sent_cnt, feature_size))]

    for i, pairs in enumerate(align_pairs):
        try:
            if args.align_mode == 'word_align':
                for a in pairs:
                    if len(fltr_bpe_table[i][0][a[0]]) > 0 and len(fltr_bpe_table[i][1][a[1]]) > 0:
                        src_word_avg_embed = np.zeros((1, feature_size))
                        for j in fltr_bpe_table[i][0][a[0]]:
                            src_word_avg_embed += src_embed[i][j,:]
                        final_res[0][cnt,:] = src_word_avg_embed / len(fltr_bpe_table[i][0][a[0]])

                        tgt_word_avg_embed = np.zeros((1, feature_size))
                        for j in fltr_bpe_table[i][1][a[1]]:
                            tgt_word_avg_embed += tgt_embed[i][j,:]

                        final_res[1][cnt,:] = tgt_word_avg_embed / len(fltr_bpe_table[i][1][a[1]])
                        cnt += 1
            elif args.align_mode == 'sent_avg':
                final_res[0][i,:] = np.mean(src_embed[i], axis=0)
                final_res[1][i,:] = np.mean(tgt_embed[i], axis=0)
        except IndexError:
            continue

    # save the final results
    suffix = ['.src', '.trg']
    for j in [0, 1]:
        np.savetxt(args.output_file+suffix[j], final_res[j], delimiter=' ', fmt='%0.6f')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
