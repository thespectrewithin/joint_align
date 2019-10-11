import argparse
from transformers import BertTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--lang1_data", type=str, required=True,
                    help="path to file that stores lang1 part of the parallel data.")
parser.add_argument("--lang2_data", type=str, required=True,
                    help="path to file that stores lang2 part of the parallel data.")
parser.add_argument("--size", default=30000, type=int,
                    help="size of parallel data needed")

parser.add_argument("--output", type=str, required=True,
                    help="output path for the processed data")

args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

lang1 = []
lang2 = []

for line in open(args.lang1_data):
    lang1.append(line.strip())

for line in open(args.lang2_data):
    lang2.append(line.strip())

assert len(lang1) == len(lang2)

output1 = open(args.output + '.cased', 'w')
output2 = open(args.output + '.uncased', 'w')

size = 0

for i in range(len(lang1)):
    
    sent1 = tokenizer.basic_tokenizer.tokenize(lang1[i])
    sent2 = tokenizer.basic_tokenizer.tokenize(lang2[i])
    
    if len(sent1) < 100 and len(sent2) < 100 and len(sent1) > 0 and len(sent2) > 0:
        output1.write(' '.join(sent1) + ' ||| ' + ' '.join(sent2) + '\n')
        output2.write(' '.join(sent1).lower() + ' ||| ' + ' '.join(sent2).lower() + '\n')
        size += 1
        
    if size >= args.size:
        break

output1.close()
output2.close()
