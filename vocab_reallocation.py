import argparse
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--src_vocab", type=str, required=True)
parser.add_argument("--tgt_vocab", type=str, required=True)
parser.add_argument("--joint_vocab", type=str, required=True)
parser.add_argument("--threshold", type=int, default=95)

parser.add_argument("--input_embedding", type=str, required=True)
parser.add_argument("--dim", type=int, default=300)
parser.add_argument("--vocab_size", type=int, default=400000)

parser.add_argument("--src_only_output_path", type=str, required=True)
parser.add_argument("--tgt_only_output_path", type=str, required=True)
parser.add_argument("--joint_only_output_path", type=str, required=True)

params = parser.parse_args()

src_vocab, src_words_total = read_vocab(params.src_vocab)
tgt_vocab, tgt_words_total = read_vocab(params.tgt_vocab)
joint_vocab, joint_words_total = read_vocab(params.joint_vocab)

corpus_ratio_coeff = src_words_total/tgt_words_total

threshold = params.threshold / 100.0

src_only, tgt_only, joint_only = [], [], []

for word in joint_vocab:
    if word not in src_vocab:
        tgt_only.append(word)
    elif word not in tgt_vocab:
        src_only.append(word)
    else:
        src_count = src_vocab[word]
        tgt_count = tgt_vocab[word] * corpus_ratio_coeff
        total_count = src_count + tgt_count
        if src_count / total_count >= threshold:
            src_only.append(word)
        elif tgt_count / total_count >= threshold:
            tgt_only.append(word)
        else:
            joint_only.append(word)

src_only = set(src_only)
tgt_only = set(tgt_only)
joint_only = set(joint_only)

joint_dict, joint_emb, joint_words = load_embedding(params.input_embedding, params.vocab_size)

src_count, tgt_count, joint_count = 0, 0, 0

for i in range(len(joint_words)):
    if joint_words[i] in src_only:
        src_count += 1
    elif joint_words[i] in tgt_only:
        tgt_count += 1
    elif joint_words[i] in joint_only:
        joint_count += 1

output_src = open(params.src_only_output_path, 'w')
output_tgt = open(params.tgt_only_output_path, 'w')
output_joint = open(params.joint_only_output_path, 'w')

output_src.write(str(src_count) + ' ' + str(params.dim) + '\n')
output_tgt.write(str(tgt_count) + ' ' + str(params.dim) + '\n')
output_joint.write(str(joint_count) + ' ' + str(params.dim) + '\n')


for i in range(len(joint_words)):
    if joint_words[i] in src_only:
        output_src.write(joint_words[i] + ' ' + ' '.join([str(n) for n in joint_emb[i]]) + '\n')
    elif joint_words[i] in tgt_only:
        output_tgt.write(joint_words[i] + ' ' + ' '.join([str(n) for n in joint_emb[i]]) + '\n')
    elif joint_words[i] in joint_only:
        output_joint.write(joint_words[i] + ' ' + ' '.join([str(n) for n in joint_emb[i]]) + '\n')

output_src.close()
output_tgt.close()
output_joint.close()
