import os
import argparse
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--src_vocab", type=str, required=True)
parser.add_argument("--tgt_vocab", type=str, required=True)

parser.add_argument("--input_embedding", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--topk", type=int, default=200000)

parser.add_argument("--add_dico_vocab", type=bool, default=True)
parser.add_argument("--src_lang", type=str, default="")
parser.add_argument("--tgt_lang", type=str, default="")
parser.add_argument("--dico_path", type=str, default="")


params = parser.parse_args()

joint_dict, joint_emb, joint_words = load_embedding(params.input_embedding)

included_word = set()
src_count = 0
for line in open(params.src_vocab):
    word, _ = line.rstrip().split(' ')
    included_word.add(word)
    src_count += 1
    if src_count >= params.topk:
        break

tgt_count = 0
for line in open(params.tgt_vocab):
    word, _ = line.rstrip().split(' ')
    included_word.add(word)
    tgt_count += 1
    if tgt_count >= params.topk:
        break

if params.add_dico_vocab:
    with open(os.path.join(params.dico_path, "%s-%s.0-5000.txt" % (params.src_lang, params.tgt_lang))) as f:
        for line in f:
            included_word.add(line.strip().split()[0])
            included_word.add(line.strip().split()[1])

    with open(os.path.join(params.dico_path, "%s-%s.5000-6500.txt" % (params.src_lang, params.tgt_lang))) as f:
        for line in f:
            included_word.add(line.strip().split()[0])
            included_word.add(line.strip().split()[1])

output_file, total_count = "", 0
for w in included_word:
    if w in joint_dict:
        total_count += 1
        output_file += w + " " + ' '.join([str(v) for v in joint_emb[joint_dict[w]]]) + "\n"

dim = len(joint_emb[0])
output_file = str(total_count) + " " + str(dim) + "\n" + output_file
print("Saving embedding with words:", total_count, "dim:", dim)
with open(params.output_path, "w") as file:
    file.write(output_file)