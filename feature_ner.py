import argparse
import logging
import random
import time
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from transformers import BertTokenizer

from crf import ChainCRF

from utils import evaluate_ner, NerProcessor, get_features

class lstm_ner(nn.Module):

    def __init__(self, input_size, layer, hidden, n_label, emb_dropout=0.5, word_dropout=0.5):

        super(lstm_ner, self).__init__()
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        self.word_lstm = nn.LSTM(input_size, hidden, layer, batch_first=True, bidirectional=True)
        self.word_dropout = nn.Dropout(word_dropout)
        self.crf = ChainCRF(hidden * 2, n_label)

    def forward(self, feat, word_length, target, gather_input, pad):
        word_input = feat

        if self.emb_dropout.p > 0.0:
            word_input = self.emb_dropout(word_input)
            
        word_length, word_idx = word_length.sort(0, descending=True)
        word_input = word_input[word_idx]

        word_packed_input = pack_padded_sequence(word_input, word_length.cpu().data.numpy(), batch_first=True)
        word_packed_output, _ = self.word_lstm(word_packed_input)
        word_output, _ = pad_packed_sequence(word_packed_output, batch_first=True, total_length=feat.size(1))
        word_output = word_output[torch.from_numpy(np.argsort(word_idx.cpu().data.numpy())).cuda()]

        if self.word_dropout.p > 0.0:
            word_output = self.word_dropout(word_output)

        # we use the 1st bpe corresponding to a word for prediction.
        word_output = torch.gather(torch.cat([word_output, pad], 1), 1, gather_input.unsqueeze(2).expand(gather_input.size(0), gather_input.size(1), word_output.size(2)))
            
        crf_loss = self.crf.loss(word_output, target, target.ne(0).float())
        predict = self.crf.decode(word_output, target.ne(0).float(), 1)
        return crf_loss, predict

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_length, label_id, index):
        self.input_length = input_length
        self.label_id = label_id
        self.input_index =index

def convert_examples_to_features(examples, label_list, max_seq_length, features, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    inputs = []
    input_features = []

    label_max_seq_len = max(len(s.label) for s in examples)
    
    for (ex_index,example) in enumerate(examples):
        
        textlist = example.text_a.split(' ')
        labels = example.label
        tokens = []
        index = [] # for gathering the 1st bpe tokens later
        feature = features[ex_index]
        
        for i, word in enumerate(textlist):
            index.append(len(tokens) + 1)
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            new_index = []
            for ind in range(len(index)):
                if index[ind] <= (max_seq_length - 2):
                    new_index.append(index[ind])
            index = new_index
            labels = labels[0:len(index)]
            s = np.expand_dims(feature[0], axis=0)
            m = feature[1:len(feature)-1]
            e = np.expand_dims(feature[len(feature)-1], axis=0)
            feature = np.concatenate([s, m[0:(max_seq_length-2)], e])

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        label_ids = [label_map[lb] for lb in labels]
        length = len(tokens)
        
        if len(tokens) < max_seq_length:
            fill_size = max_seq_length - len(tokens)
            feature = np.concatenate([feature, np.zeros((fill_size, len(feature[0])))])

        if len(labels) < label_max_seq_len:
            fill_size = label_max_seq_len - len(labels)
            label_ids += [0] * fill_size
            index += [max_seq_length] * fill_size
        
        assert len(feature) == max_seq_length
        assert len(label_ids) == label_max_seq_len
        assert len(index) == label_max_seq_len
        
        inputs.append(InputFeatures(input_length=length, label_id=label_ids, index=index))
        input_features.append(feature)
        
    return inputs, torch.tensor(input_features, dtype=torch.float)

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag for uncased models.")

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    
    parser.add_argument('--train_feat', type=str)
    parser.add_argument('--dev_feat', type=str)
    parser.add_argument('--test_feat', type=str)

    parser.add_argument("--train_batch_size", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=40, type=int)

    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--log_interval', type=int, default=0)
    
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--embed_dropout', type=float, default=0.5)
    parser.add_argument('--word_dropout', type=float, default=0.5)

    parser.add_argument('--log_file', type=str, default=0)

    args = parser.parse_args()

    logging.basicConfig(handlers = [logging.FileHandler(args.log_file + '.log'),
                                    logging.StreamHandler()],
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)
    logging.info("Input args: %r" % args)

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = processor.get_train_examples(args.train_data)
    dev_examples = processor.get_dev_examples(args.dev_data)
    test_examples = processor.get_test_examples(args.test_data)

    train_lengths = [len(tokenizer.tokenize(t.text_a)) for t in train_examples]
    train_max_seq_len = max(train_lengths) + 2
    dev_lengths = [len(tokenizer.tokenize(t.text_a)) for t in dev_examples]
    dev_max_seq_len = max(dev_lengths) + 2
    test_lengths = [len(tokenizer.tokenize(t.text_a)) for t in test_examples]
    test_max_seq_len = max(test_lengths) + 2
    
    logger.info("Train max sequence length: %d" % train_max_seq_len)
    logger.info("Train max sequence: %s" % ' '.join(tokenizer.tokenize(train_examples[train_lengths.index(train_max_seq_len - 2)].text_a)))
    logger.info("Dev max sequence length: %d" % dev_max_seq_len)
    logger.info("Dev max sequence: %s" % ' '.join(tokenizer.tokenize(dev_examples[dev_lengths.index(dev_max_seq_len - 2)].text_a)))
    logger.info("Test max sequence length: %d" % test_max_seq_len)
    logger.info("Test max sequence: %s" % ' '.join(tokenizer.tokenize(test_examples[test_lengths.index(test_max_seq_len - 2)].text_a)))
    
    if train_max_seq_len > 512:
        train_max_seq_len = 512
        logger.info("Train max sequence length reset to 512")
    if dev_max_seq_len > 512:
        dev_max_seq_len = 512
        logger.info("Dev max sequence length reset to 512")
    if test_max_seq_len > 512:
        test_max_seq_len = 512
        logger.info("Test max sequence length reset to 512")

    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

    model = lstm_ner(args.feature_size, args.layer, args.hidden, num_labels, args.embed_dropout, args.word_dropout)
    model.to(device)
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
    vocab_mask = torch.ones(len(label_list))
    vocab_mask[0] = 0
    cross_entropy = CrossEntropyLoss(weight=vocab_mask).to(device)

    logger.info("loading features into memory. might take some time")

    train_features, train_f = convert_examples_to_features(train_examples, label_list, train_max_seq_len, get_features(args.train_feat), tokenizer)
    dev_features, dev_f = convert_examples_to_features(dev_examples, label_list, dev_max_seq_len, get_features(args.dev_feat), tokenizer)
    test_features, test_f = convert_examples_to_features(test_examples, label_list, test_max_seq_len, get_features(args.test_feat), tokenizer)
    
    all_input_lengths = torch.tensor([f.input_length for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_indices = torch.tensor([f.input_index for f in train_features], dtype=torch.long)
    
    train_data = TensorDataset(train_f, all_input_lengths, all_label_ids, all_indices)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    def eval(eval_features, eval_f):

        all_input_lengths = torch.tensor([f.input_length for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_indices = torch.tensor([f.input_index for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(eval_f, all_input_lengths, all_label_ids, all_indices)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        model.eval()

        test_corr = 0
        test_total = 0
        test_pred = []
        test_true = []
        
        label_map = {i : label for i, label in enumerate(label_list)}
        
        for input_feature, input_length, label_ids, indices in tqdm(eval_dataloader, desc="Evaluating"):
            
            input_feature = input_feature.to(device)
            input_length = input_length.to(device)
            label_ids = label_ids.to(device)
            indices = indices.to(device)

            # ensures proper padding when gathering
            pad = torch.zeros((input_feature.size(0), 1, 2*args.hidden)).to(device)

            with torch.no_grad():
                loss, predict = model(input_feature, input_length, label_ids, indices, pad)

            gold = label_ids.contiguous().view(-1)
            num_tokens = gold.data.ne(0).float().sum()
            pred = predict.contiguous().view(-1)
            correct = pred.eq(gold.data).masked_select(gold.ne(0).data).float().sum()
            
            test_corr += correct
            test_total += num_tokens
            label_ids = label_ids.to('cpu').numpy()
            pred = pred.view(input_feature.size(0), -1).to('cpu').numpy()

            for i in range(len(label_ids)):
                temp_1 = []
                temp_2 = []
                for j in range(len(label_ids[i])):
                    if label_ids[i][j] != 0:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[pred[i][j]])
                test_true.append(temp_1)
                test_pred.append(temp_2)

        test_acc = test_corr * 100.0 / test_total

        test_p, test_r, test_f = evaluate_ner(test_pred, test_true)

        model.train()

        return test_acc, test_p, test_r, test_f, test_pred

    best_dev = 0
    best_test = 0
    global_step = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()
    
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        
        start_time = time.time()
        tr_loss = 0
        nb_tr_steps = 0
        total_words = 0
        total_correct = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            
            batch = tuple(t.to(device) for t in batch)
            input_feature, input_length, label_ids, indices = batch
            
            # ensures proper padding when gathering            
            pad = torch.zeros((input_feature.size(0), 1, 2*args.hidden)).to(device)

            loss, predict = model(input_feature, input_length, label_ids, indices, pad)
            loss = loss.mean()

            pred = predict.contiguous().view(-1)
            gold = label_ids.contiguous().view(-1)
            num_tokens = gold.data.ne(0).float().sum()
            correct = pred.eq(gold.data).masked_select(gold.ne(0).data).float().sum()

            tr_loss += loss.item()
            nb_tr_steps += 1
            total_correct += correct
            total_words += num_tokens
            global_step += 1
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.log_interval > 0 and global_step % args.log_interval == 0:
                d_acc, d_p, d_r, d_f, d_preds = eval(dev_features, dev_f)
                t_acc, t_p, t_r, t_f, t_preds = eval(test_features, test_f)
                if d_f > best_dev:
                    best_dev = d_f
                    best_test = t_f
                    logger.info("{:4d}/{:4d} steps |  dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(
                        global_step, int(num_train_optimization_steps), d_acc, d_p, d_r, d_f))
                    logger.info("{:4d}/{:4d} steps | test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(
                        global_step, int(num_train_optimization_steps), t_acc, t_p, t_r, t_f))
                else:
                    logger.info("{:4d}/{:4d} steps |  dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(
                        global_step, int(num_train_optimization_steps), d_acc, d_p, d_r, d_f))
                    logger.info("{:4d}/{:4d} steps | test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(
                        global_step, int(num_train_optimization_steps), t_acc, t_p, t_r, t_f))

        d_acc, d_p, d_r, d_f, d_preds = eval(dev_features, dev_f)
        t_acc, t_p, t_r, t_f, t_preds = eval(test_features, test_f)
        
        logger.info("Epoch {} of {} took {:.4f}s, training loss: {:.4f}, training accuracy: {:.4f}".format(
            epoch+1, int(args.num_train_epochs), time.time() - start_time, tr_loss/nb_tr_steps, total_correct * 100.0/total_words))

        if d_f > best_dev:
            best_dev = d_f
            best_test = t_f
            logger.info(" dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(d_acc, d_p, d_r, d_f))
            logger.info("test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f} ********".format(t_acc, t_p, t_r, t_f))

        else:
            logger.info(" dev acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(d_acc, d_p, d_r, d_f))
            logger.info("test acc: {:.4f} | recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(t_acc, t_p, t_r, t_f))

        if args.optimizer == 'sgd':
            lr = args.learning_rate / (1.0 + (1+ epoch) * 0.05)
            logger.info("learning rate: {:.4f}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    logger.info("Best test F1: {:.4f}".format(best_test))

if __name__ == "__main__":
    main()

